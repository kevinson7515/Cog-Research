#!/usr/bin/env python3
"""Vision-only SFT for a Co-Sight Qwen3-VL workflow adapter.

This script starts from the base Qwen3-VL checkpoint plus the existing
workflow/tool-call LoRA adapter, merges the workflow LoRA into the base weights,
freezes the merged model, and trains a new LoRA adapter on visual modules only.

The default output is a second-stage vision LoRA adapter. Merge it after the
workflow adapter, in order, before serving.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


IGNORE_INDEX = -100


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() not in {"0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path",
        default="outputs/qwen3-vl-8b-cosight-merged",
        help="Training base. Prefer the workflow-merged model to avoid merging LoRA inside every torchrun rank.",
    )
    parser.add_argument(
        "--adapter_name_or_path",
        default="",
        help="Optional existing workflow PEFT adapter to merge before training. Leave empty when model_name_or_path is already workflow-merged.",
    )
    parser.add_argument("--train_file", default="data/vision_sft/vision_sft_messages.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen3-vl-8b-cosight-vision-lora")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Skip samples above this true multimodal token length. 4096 is the stable default for 3-GPU vision-LoRA runs.",
    )
    parser.add_argument("--skip_overlength", type=parse_bool, default=True)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--dry_run", type=parse_bool, default=False)
    parser.add_argument("--train_on_inputs", type=parse_bool, default=False)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument(
        "--save_strategy",
        choices=["no", "steps", "epoch"],
        default="no",
        help="Vision-LoRA training saves the final adapter explicitly; disabling Trainer checkpoints avoids full-model save paths.",
    )
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", type=parse_bool, default=True)
    parser.add_argument("--fp16", type=parse_bool, default=False)
    parser.add_argument(
        "--gradient_checkpointing",
        type=parse_bool,
        default=False,
        help=(
            "Disabled by default for Qwen3-VL vision-LoRA training. "
            "The visual branch uses dynamic tensors whose metadata can mismatch during checkpoint recomputation."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing_use_reentrant",
        type=parse_bool,
        default=False,
        help="Use non-reentrant checkpointing by default so frozen-base LoRA training still gets gradients.",
    )
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--merge_lora_before_training",
        type=parse_bool,
        default=True,
        help="Merge the workflow LoRA into the base model before training the second-stage vision LoRA.",
    )
    parser.add_argument("--use_vision_lora", type=parse_bool, default=True, help="Train a second-stage LoRA on visual modules.")
    parser.add_argument("--vision_lora_r", type=int, default=32)
    parser.add_argument("--vision_lora_alpha", type=int, default=128)
    parser.add_argument("--vision_lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--vision_lora_module_suffixes",
        default="",
        help="Comma-separated module suffixes under visual modules. Empty means all Linear modules under visual patterns.",
    )
    parser.add_argument(
        "--freeze_non_vision",
        type=parse_bool,
        default=True,
        help="Freeze all parameters before enabling the visual tower/projector patterns.",
    )
    parser.add_argument(
        "--vision_trainable_patterns",
        default="visual,vision_tower,vision_model,vision_encoder,multi_modal_projector,mm_projector,merger",
        help="Comma-separated substrings. Matching parameters are trainable.",
    )
    parser.add_argument(
        "--vision_exclude_patterns",
        default="lm_head,embed_tokens,language_model,model.layers",
        help="Comma-separated substrings excluded even if a trainable pattern matches.",
    )
    parser.add_argument(
        "--enforce_vision_only_trainable",
        type=parse_bool,
        default=True,
        help="Abort if any trainable tensor falls outside the configured visual/module-merger scope.",
    )
    parser.add_argument(
        "--remove_token_type_ids",
        type=parse_bool,
        default=True,
        help="Drop token_type_ids from processor output. Qwen-VL forwards usually do not consume it.",
    )
    parser.add_argument("--max_shard_size", default="4GB")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def load_processor_and_tokenizer(model_path: str):
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True, fix_mistral_regex=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor, tokenizer


def from_pretrained_with_dtype(model_cls: Any, model_path: str, dtype: torch.dtype):
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    try:
        return model_cls.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return model_cls.from_pretrained(model_path, torch_dtype=dtype, **kwargs)


def load_base_model(model_path: str, args: argparse.Namespace):
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model_classes = []
    try:
        from transformers import AutoModelForImageTextToText

        model_classes.append(AutoModelForImageTextToText)
    except Exception:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        model_classes.append(AutoModelForVision2Seq)
    except Exception:
        pass
    from transformers import AutoModelForCausalLM

    model_classes.append(AutoModelForCausalLM)

    last_error = None
    for model_cls in model_classes:
        try:
            return from_pretrained_with_dtype(model_cls, model_path, dtype)
        except Exception as exc:
            last_error = exc

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    raise RuntimeError(
        f"Could not load model for config model_type={getattr(config, 'model_type', None)!r}. "
        f"Last error: {last_error}"
    )


def disable_bnb_dispatchers() -> None:
    try:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False
    except Exception:
        pass
    try:
        import peft.tuners.lora.model as lora_model

        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
    except Exception:
        pass


def load_and_optionally_merge_adapter(model, adapter_path: str, merge: bool):
    if not adapter_path:
        return model, False
    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    disable_bnb_dispatchers()
    from peft import PeftModel

    try:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    except TypeError:
        model = PeftModel.from_pretrained(model, adapter_path)
        for _, param in model.named_parameters():
            param.requires_grad = False

    if merge:
        model = model.merge_and_unload()
        print(f"Loaded and merged workflow LoRA adapter from {adapter.resolve()}")
    else:
        print(f"Loaded frozen workflow LoRA adapter from {adapter.resolve()}")
    return model, True


def pattern_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def configure_trainable_params(model, args: argparse.Namespace) -> Dict[str, Any]:
    include = pattern_list(args.vision_trainable_patterns)
    exclude = pattern_list(args.vision_exclude_patterns)
    if args.freeze_non_vision:
        for param in model.parameters():
            param.requires_grad = False

    trainable_names: List[str] = []
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        should_train = any(pattern in name for pattern in include)
        if should_train and any(pattern in name for pattern in exclude):
            should_train = False
        if should_train:
            param.requires_grad = True
        if param.requires_grad:
            trainable_names.append(name)
            trainable_params += param.numel()

    if not trainable_names:
        raise RuntimeError(
            "No trainable vision parameters matched. "
            f"Patterns were include={include}, exclude={exclude}. "
            "Run with a broader --vision_trainable_patterns after inspecting model.named_parameters()."
        )

    roots: Dict[str, int] = {}
    for name in trainable_names:
        root = ".".join(name.split(".")[:3])
        roots[root] = roots.get(root, 0) + 1

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": round(trainable_params / max(1, total_params), 6),
        "trainable_tensor_count": len(trainable_names),
        "trainable_roots": dict(sorted(roots.items(), key=lambda item: (-item[1], item[0]))[:20]),
        "trainable_name_examples": trainable_names[:20],
    }
    print(json.dumps({"trainable_summary": summary}, ensure_ascii=False, indent=2))
    return summary


def find_vision_lora_targets(model, args: argparse.Namespace) -> List[str]:
    include = pattern_list(args.vision_trainable_patterns)
    exclude = pattern_list(args.vision_exclude_patterns)
    suffixes = pattern_list(args.vision_lora_module_suffixes)
    targets: List[str] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if not any(pattern in name for pattern in include):
            continue
        if any(pattern in name for pattern in exclude):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if suffixes and not any(name.endswith(suffix) or name.split(".")[-1] == suffix for suffix in suffixes):
            continue
        targets.append(name)
    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError(
            "No visual Linear modules matched for LoRA. "
            f"Patterns were include={include}, exclude={exclude}, suffixes={suffixes}."
        )
    print(
        json.dumps(
            {
                "vision_lora_targets": {
                    "count": len(targets),
                    "examples": targets[:50],
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return targets


def apply_vision_lora(model, args: argparse.Namespace):
    from peft import LoraConfig, get_peft_model

    for param in model.parameters():
        param.requires_grad = False

    targets = find_vision_lora_targets(model, args)
    config = LoraConfig(
        r=args.vision_lora_r,
        lora_alpha=args.vision_lora_alpha,
        lora_dropout=args.vision_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def summarize_trainable_params(model) -> Dict[str, Any]:
    trainable_names: List[str] = []
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_names.append(name)
            trainable_params += param.numel()
    roots: Dict[str, int] = {}
    for name in trainable_names:
        root = ".".join(name.split(".")[:5])
        roots[root] = roots.get(root, 0) + 1
    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": round(trainable_params / max(1, total_params), 8),
        "trainable_tensor_count": len(trainable_names),
        "trainable_roots": dict(sorted(roots.items(), key=lambda item: (-item[1], item[0]))[:20]),
        "trainable_name_examples": trainable_names[:20],
    }
    print(json.dumps({"trainable_summary": summary}, ensure_ascii=False, indent=2))
    return summary


def enforce_vision_only_trainable_scope(model, args: argparse.Namespace) -> None:
    include = pattern_list(args.vision_trainable_patterns)
    exclude = pattern_list(args.vision_exclude_patterns)
    violations = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not any(pattern in name for pattern in include) or any(pattern in name for pattern in exclude):
            violations.append(name)

    if violations:
        raise RuntimeError(
            "Vision-only training guard failed: non-vision trainable tensors were found. "
            f"Examples: {violations[:20]}"
        )
    print(
        json.dumps(
            {
                "vision_only_trainable_guard": {
                    "passed": True,
                    "include_patterns": include,
                    "exclude_patterns": exclude,
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def role_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts).strip()
    return str(content or "")


def load_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def normalize_image_content(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            normalized_content = []
            for item in content:
                if not isinstance(item, dict):
                    normalized_content.append(item)
                    continue
                item_type = item.get("type")
                if item_type == "image":
                    image_value = item.get("image")
                    if isinstance(image_value, str):
                        normalized_content.append({**item, "image": load_image(image_value)})
                    else:
                        normalized_content.append(item)
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, str) and os.path.exists(image_url):
                        normalized_content.append({"type": "image", "image": load_image(image_url)})
                    else:
                        normalized_content.append(item)
                else:
                    normalized_content.append(item)
            copied["content"] = normalized_content
        normalized.append(copied)
    return normalized


def render_chat_template(processor, tokenizer, messages: Sequence[Dict[str, Any]], add_generation_prompt: bool) -> str:
    applier = getattr(processor, "apply_chat_template", None)
    if applier is not None:
        return applier(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def encode_with_processor(processor, tokenizer, messages: Sequence[Dict[str, Any]], add_generation_prompt: bool) -> Dict[str, Any]:
    materialized = normalize_image_content(messages)
    applier = getattr(processor, "apply_chat_template", None)
    if applier is not None:
        try:
            encoded = applier(
                materialized,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                return_tensors="pt",
            )
            return dict(encoded)
        except TypeError:
            pass

    text = render_chat_template(processor, tokenizer, messages, add_generation_prompt)
    images = []
    for message in materialized:
        content = message.get("content")
        if isinstance(content, list):
            images.extend(item.get("image") for item in content if isinstance(item, dict) and item.get("type") == "image")
    encoded = processor(text=[text], images=images or None, return_tensors="pt")
    return dict(encoded)


def tensor_to_feature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def ensure_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def squeeze_text_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim >= 2 and value.shape[0] == 1:
        return value[0]
    return value


def encoded_input_len(encoded: Dict[str, Any]) -> int:
    return int(squeeze_text_tensor(ensure_tensor(encoded["input_ids"])).numel())


class VisionSFTDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        processor,
        tokenizer,
        max_seq_length: int,
        train_on_inputs: bool,
        skip_overlength: bool,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_on_inputs = train_on_inputs
        self.records: List[Dict[str, Any]] = []

        skipped_overlength = 0
        skipped_missing_image = 0
        skipped_processor_error = 0
        print("Validating actual multimodal token lengths with the processor...")
        for record in records:
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            image_paths = record.get("images") or []
            missing = [path for path in image_paths if isinstance(path, str) and not os.path.exists(path)]
            if missing:
                skipped_missing_image += 1
                continue
            try:
                encoded = encode_with_processor(processor, tokenizer, messages, add_generation_prompt=False)
                token_count = encoded_input_len(encoded)
                prefix_len = 0
                if not train_on_inputs:
                    prefix_len = self.supervised_prefix_len(messages)
            except Exception as exc:
                skipped_processor_error += 1
                if skipped_processor_error <= 5:
                    print(f"Skipping sample because processor failed: {exc}")
                continue
            if skip_overlength and token_count > max_seq_length:
                skipped_overlength += 1
                continue
            self.records.append(
                {
                    "messages": messages,
                    "meta": record.get("meta", {}),
                    "token_count": token_count,
                    "prefix_len": prefix_len,
                }
            )

        if skipped_missing_image:
            print(f"Skipped {skipped_missing_image} samples with missing local image files.")
        if skipped_overlength:
            print(f"Skipped {skipped_overlength} actual multimodal overlength samples.")
        if skipped_processor_error:
            print(f"Skipped {skipped_processor_error} samples that failed processor encoding.")

    def __len__(self) -> int:
        return len(self.records)

    def supervised_prefix_len(self, messages: Sequence[Dict[str, Any]]) -> int:
        target_index = len(messages) - 1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                target_index = idx
                break
        prompt_messages = list(messages[:target_index])
        encoded = encode_with_processor(self.processor, self.tokenizer, prompt_messages, add_generation_prompt=True)
        return encoded_input_len(encoded)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        messages = record["messages"]
        encoded = encode_with_processor(self.processor, self.tokenizer, messages, add_generation_prompt=False)

        input_ids = squeeze_text_tensor(ensure_tensor(encoded["input_ids"])).to(torch.long)
        attention_mask = squeeze_text_tensor(ensure_tensor(encoded["attention_mask"])).to(torch.long)
        if input_ids.numel() > self.max_seq_length:
            raise ValueError(
                f"Multimodal sample length {input_ids.numel()} exceeds max_seq_length={self.max_seq_length}. "
                "Do not truncate Qwen-VL samples after image processing, because image tokens must match image features. "
                "Keep --skip_overlength true or increase --max_seq_length."
            )

        if self.train_on_inputs:
            labels = input_ids.clone()
        else:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            prefix_len = min(int(record.get("prefix_len", 0) or self.supervised_prefix_len(messages)), input_ids.numel())
            if prefix_len < input_ids.numel():
                labels[prefix_len:] = input_ids[prefix_len:]
            if torch.all(labels == IGNORE_INDEX):
                labels = input_ids.clone()

        feature: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key, value in encoded.items():
            if key in feature:
                continue
            feature[key] = tensor_to_feature(value)
        return feature


@dataclass
class VisionDataCollator:
    tokenizer: Any
    remove_token_type_ids: bool = True

    def pad_1d(self, tensors: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(tensor.numel() for tensor in tensors)
        output = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
        for idx, tensor in enumerate(tensors):
            output[idx, : tensor.numel()] = tensor
        return output

    def collate_extra_tensor(self, key: str, values: Sequence[torch.Tensor]) -> torch.Tensor:
        squeezed = [value.detach().cpu() for value in values]
        if key.endswith("grid_thw") or key in {"pixel_values", "pixel_values_videos"}:
            return torch.cat(squeezed, dim=0)
        if all(value.shape == squeezed[0].shape for value in squeezed):
            return torch.stack(squeezed, dim=0)
        if all(value.ndim == 1 for value in squeezed):
            return self.pad_1d(squeezed, 0)
        return torch.cat(squeezed, dim=0)

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, Any] = {
            "input_ids": self.pad_1d([feature["input_ids"] for feature in features], self.tokenizer.pad_token_id),
            "attention_mask": self.pad_1d([feature["attention_mask"] for feature in features], 0),
            "labels": self.pad_1d([feature["labels"] for feature in features], IGNORE_INDEX),
        }
        extra_keys = sorted(set().union(*(feature.keys() for feature in features)) - set(batch.keys()))
        for key in extra_keys:
            if self.remove_token_type_ids and key == "token_type_ids":
                continue
            values = [feature[key] for feature in features if key in feature and feature[key] is not None]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                batch[key] = self.collate_extra_tensor(key, values)
            else:
                batch[key] = values
        return batch


def dataset_stats(dataset: VisionSFTDataset) -> Dict[str, Any]:
    token_lengths = [int(record.get("token_count", 0)) for record in dataset.records]
    image_counts = []
    answer_chars = []
    for record in dataset.records:
        messages = record.get("messages") or []
        images = record.get("meta", {}).get("image_paths") or []
        image_counts.append(len(images))
        if messages:
            answer_chars.append(len(role_content_text(messages[-1].get("content"))))
    return {
        "samples": len(dataset),
        "min_tokens": min(token_lengths) if token_lengths else 0,
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "avg_tokens": round(sum(token_lengths) / max(1, len(token_lengths)), 2),
        "min_images": min(image_counts) if image_counts else 0,
        "max_images": max(image_counts) if image_counts else 0,
        "avg_images": round(sum(image_counts) / max(1, len(image_counts)), 2),
        "min_answer_chars": min(answer_chars) if answer_chars else 0,
        "max_answer_chars": max(answer_chars) if answer_chars else 0,
        "avg_answer_chars": round(sum(answer_chars) / max(1, len(answer_chars)), 2),
    }


def save_processor(processor, tokenizer, output_dir: str) -> None:
    if processor is not None:
        processor.save_pretrained(output_dir)
    else:
        tokenizer.save_pretrained(output_dir)


def enable_gradient_checkpointing(model, use_reentrant: bool) -> None:
    print(
        "WARNING: gradient checkpointing is enabled for vision-LoRA training. "
        "Qwen3-VL visual modules can fail with recomputed tensor metadata mismatches; "
        "set --gradient_checkpointing false if that happens."
    )
    kwargs = {"use_reentrant": use_reentrant}
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def make_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "none",
        "remove_unused_columns": False,
        "deepspeed": args.deepspeed or None,
        "ddp_find_unused_parameters": False,
    }
    if "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": args.gradient_checkpointing_use_reentrant}
    return TrainingArguments(**kwargs)


def save_final_model(trainer: Trainer, output_dir: str, max_shard_size: str) -> None:
    if getattr(trainer, "is_deepspeed_enabled", False):
        trainer.save_model(output_dir)
        return

    model = trainer.model
    while hasattr(model, "module"):
        module = getattr(model, "module")
        if module is None or module is model:
            break
        model = module
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size=max_shard_size)


def is_dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_rank_zero() -> bool:
    return not is_dist_initialized() or torch.distributed.get_rank() == 0


def wait_for_all_ranks() -> None:
    if is_dist_initialized():
        torch.distributed.barrier()


def maybe_unwrap_model(model):
    while hasattr(model, "module"):
        module = getattr(model, "module")
        if module is None or module is model:
            break
        model = module
    return model


def gather_param_to_rank0(param: torch.nn.Parameter) -> torch.Tensor | None:
    if hasattr(param, "ds_id"):
        try:
            import deepspeed

            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                if is_rank_zero():
                    return param.detach().cpu().clone()
                return None
        except ImportError as exc:
            raise ImportError("Saving ZeRO-3 LoRA parameters requires deepspeed.") from exc

    if is_rank_zero():
        return param.detach().cpu().clone()
    return None


def load_tensor_keys(path: str | Path) -> List[str]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("Reading safetensors adapter files requires safetensors.") from exc
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return list(f.keys())

    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    return list(state_dict.keys())


def inspect_saved_lora_adapter(output_dir: str | Path) -> Tuple[str, int]:
    output_dir = Path(output_dir)
    candidates = [
        output_dir / "adapter_model.safetensors",
        output_dir / "adapter_model.bin",
    ]
    for path in candidates:
        if path.exists():
            keys = load_tensor_keys(path)
            lora_keys = [key for key in keys if "lora_" in key]
            if not lora_keys:
                raise RuntimeError(f"Saved adapter file has no LoRA tensors: {path}")
            return os.fspath(path), len(lora_keys)
    raise FileNotFoundError(f"No adapter weight file found in {output_dir}")


def summarize_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    lora_items = [(name, tensor) for name, tensor in state_dict.items() if "lora_" in name]
    total_numel = 0
    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    zero_tensor_count = 0
    root_counts: Dict[str, int] = {}

    for name, tensor in lora_items:
        detached = tensor.detach().float()
        abs_tensor = detached.abs()
        tensor_max = abs_tensor.max().item() if detached.numel() else 0.0
        if tensor_max == 0.0:
            zero_tensor_count += 1
        max_abs = max(max_abs, tensor_max)
        total_numel += detached.numel()
        sum_abs += abs_tensor.sum().item()
        sum_sq += detached.square().sum().item()

        clean_name = name
        for prefix in ("base_model.model.", "model."):
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix) :]
        root = ".".join(clean_name.split(".")[:4])
        root_counts[root] = root_counts.get(root, 0) + 1

    return {
        "lora_tensor_count": len(lora_items),
        "lora_numel": total_numel,
        "lora_max_abs": max_abs,
        "lora_mean_abs": sum_abs / total_numel if total_numel else 0.0,
        "lora_rms": math.sqrt(sum_sq / total_numel) if total_numel else 0.0,
        "zero_lora_tensor_count": zero_tensor_count,
        "lora_roots": dict(sorted(root_counts.items(), key=lambda item: (-item[1], item[0]))[:20]),
    }


def save_lora_adapter_only(model, output_dir: str, processor, tokenizer) -> None:
    model = maybe_unwrap_model(model)
    if not hasattr(model, "peft_config"):
        raise ValueError("Vision LoRA saving requires a PEFT model. Did you pass --use_vision_lora true?")

    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:
        raise ImportError("Saving a LoRA adapter requires peft.") from exc

    if is_rank_zero():
        print("Saving vision LoRA adapter only...")

    gathered_state_dict: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad or "lora_" in name or "modules_to_save" in name:
            tensor = gather_param_to_rank0(param)
            if tensor is not None:
                gathered_state_dict[name] = tensor

    if is_rank_zero():
        os.makedirs(output_dir, exist_ok=True)
        adapter_state_dict = get_peft_model_state_dict(model, state_dict=gathered_state_dict)
        if not adapter_state_dict:
            raise RuntimeError("No LoRA tensors were gathered; refusing to save an empty adapter.")
        lora_stats = summarize_lora_state_dict(adapter_state_dict)
        model.save_pretrained(output_dir, state_dict=gathered_state_dict)
        saved_adapter_path, saved_adapter_tensors = inspect_saved_lora_adapter(output_dir)
        save_processor(processor, tokenizer, output_dir)
        stats_path = Path(output_dir) / "vision_lora_stats.json"
        stats_path.write_text(json.dumps(lora_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "save_only_lora": True,
                    "output_dir": os.path.abspath(output_dir),
                    "adapter_tensors": len(adapter_state_dict),
                    "saved_adapter_path": saved_adapter_path,
                    "saved_adapter_tensors": saved_adapter_tensors,
                    "lora_stats_file": os.fspath(stats_path),
                    "lora_stats": lora_stats,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    wait_for_all_ranks()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    processor, tokenizer = load_processor_and_tokenizer(args.model_name_or_path)
    records = load_jsonl(args.train_file)
    dataset = VisionSFTDataset(
        records,
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        train_on_inputs=args.train_on_inputs,
        skip_overlength=args.skip_overlength,
    )
    if args.max_train_samples > 0:
        dataset.records = dataset.records[: args.max_train_samples]
    if len(dataset) == 0:
        raise RuntimeError("No vision training samples remain after preprocessing.")

    stats = dataset_stats(dataset)
    print(json.dumps({"dataset_stats": stats}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    if args.use_vision_lora and args.adapter_name_or_path and not args.merge_lora_before_training:
        raise ValueError(
            "--use_vision_lora true requires --merge_lora_before_training true when a workflow adapter is supplied. "
            "This keeps the first-stage workflow adapter active as the training base while saving only the new vision adapter."
        )

    model = load_base_model(args.model_name_or_path, args)
    model, loaded_adapter = load_and_optionally_merge_adapter(
        model,
        args.adapter_name_or_path,
        args.merge_lora_before_training,
    )
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model, args.gradient_checkpointing_use_reentrant)
    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.use_vision_lora:
        model = apply_vision_lora(model, args)
        trainable_summary = summarize_trainable_params(model)
    else:
        trainable_summary = configure_trainable_params(model, args)
    if args.enforce_vision_only_trainable:
        enforce_vision_only_trainable_scope(model, args)

    training_args = make_training_arguments(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=VisionDataCollator(tokenizer, remove_token_type_ids=args.remove_token_type_ids),
    )
    trainer.train()
    if args.use_vision_lora:
        save_lora_adapter_only(trainer.model, args.output_dir, processor, tokenizer)
    else:
        save_final_model(trainer, args.output_dir, args.max_shard_size)
        save_processor(processor, tokenizer, args.output_dir)

    print(
        json.dumps(
            {
                "status": "success",
                "output_dir": os.path.abspath(args.output_dir),
                "train_samples": len(dataset),
                "loaded_workflow_adapter": loaded_adapter,
                "merged_lora_before_training": args.merge_lora_before_training,
                "use_vision_lora": args.use_vision_lora,
                "trainable_params": trainable_summary["trainable_params"],
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler_type": args.lr_scheduler_type,
                "vision_lora_r": args.vision_lora_r,
                "vision_lora_alpha": args.vision_lora_alpha,
                "vision_lora_dropout": args.vision_lora_dropout,
                "enforce_vision_only_trainable": args.enforce_vision_only_trainable,
                "max_shard_size": args.max_shard_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
