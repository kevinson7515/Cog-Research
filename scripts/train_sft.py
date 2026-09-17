#!/usr/bin/env python3
"""SFT train Qwen3-VL/Qwen-style models from Co-Sight trace messages."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
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
    parser.add_argument("--model_name_or_path", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train_file", default="data/sft_traces/sft_messages.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen3-vl-8b-cosight-sft")
    parser.add_argument("--max_seq_length", type=int, default=32768)
    parser.add_argument("--skip_overlength", type=parse_bool, default=False)
    parser.add_argument("--max_train_samples", type=int, default=0, help="Use only the first N samples after filtering; 0 means all.")
    parser.add_argument("--dry_run", type=parse_bool, default=False, help="Only load tokenizer, tokenize dataset, and print stats.")
    parser.add_argument("--train_on_inputs", type=parse_bool, default=False, help="If false, labels are only on assistant spans.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--bf16", type=parse_bool, default=True)
    parser.add_argument("--fp16", type=parse_bool, default=False)
    parser.add_argument("--gradient_checkpointing", type=parse_bool, default=True)
    parser.add_argument("--use_lora", type=parse_bool, default=True)
    parser.add_argument("--disable_bnb", type=parse_bool, default=True, help="Disable PEFT bitsandbytes dispatchers. Keep true unless training with 4bit/8bit quantization.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module suffixes.",
    )
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        processor = None

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return processor, tokenizer


def load_model(model_path: str, args: argparse.Namespace):
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    common_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    model_classes = []
    from transformers import AutoModelForCausalLM
    model_classes.append(AutoModelForCausalLM)
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

    last_error = None
    for model_cls in model_classes:
        try:
            return model_cls.from_pretrained(model_path, **common_kwargs)
        except Exception as exc:
            last_error = exc

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    raise RuntimeError(
        f"Could not load model for config model_type={getattr(config, 'model_type', None)!r}. "
        f"Last error: {last_error}"
    )


def apply_lora(model, args: argparse.Namespace):
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA training requires peft. Install requirements-sft.txt first.") from exc

    if args.disable_bnb:
        # PEFT checks whether bitsandbytes is importable and then imports its
        # CUDA extension dispatchers. Some clusters have a CPU-only or
        # CUDA-mismatched bitsandbytes package installed; LoRA itself does not
        # need bitsandbytes unless 4bit/8bit quantization is used.
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

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def parse_tool_arguments(raw_args: Any) -> Any:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return raw_args
    return raw_args if raw_args is not None else {}


def render_tool_call(tool_call: Dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    payload = {
        "name": str(function.get("name") or tool_call.get("name") or ""),
        "arguments": parse_tool_arguments(function.get("arguments", {})),
    }
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def fallback_message_content(message: Dict[str, Any]) -> str:
    parts: List[str] = []
    content = message.get("content")
    if content:
        parts.append(str(content).strip())
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict):
            parts.append(render_tool_call(tool_call))
    return "\n\n".join(part for part in parts if part).strip()


def tool_call_names(messages: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            name = function.get("name") or tool_call.get("name")
            if name:
                names.append(str(name))
    return names


def render_with_template(tokenizer, messages: List[Dict[str, Any]]) -> str:
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        names = tool_call_names(messages)
        if names and "<tool_call>" not in rendered and not any(name in rendered for name in names):
            raise ValueError("chat template did not render structured tool_calls")
        return rendered
    except Exception:
        chunks = []
        for message in messages:
            role = message["role"]
            if role == "tool" and message.get("name"):
                role = f"tool name={message['name']}"
            chunks.append(f"<|im_start|>{role}\n{fallback_message_content(message)}<|im_end|>")
        return "\n".join(chunks)


def supervised_from_message_index(meta: Dict[str, Any], message_count: int) -> int:
    raw_value = meta.get("supervised_from_message_index", meta.get("supervised_from", 0))
    try:
        value = int(raw_value)
    except Exception:
        value = 0
    return max(0, min(value, max(0, message_count - 1)))


def assistant_char_spans(tokenizer, messages: List[Dict[str, Any]], full_text: str, meta: Dict[str, Any] | None = None) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    previous_text = ""
    meta = meta or {}
    supervised_from = supervised_from_message_index(meta, len(messages))
    for index, message in enumerate(messages):
        current_text = render_with_template(tokenizer, messages[: index + 1])
        if not current_text.startswith(previous_text):
            previous_text = current_text
            continue
        if message.get("role") == "assistant" and index >= supervised_from:
            spans.append((len(previous_text), len(current_text)))
        previous_text = current_text

    if not spans:
        marker = "<tool_call>"
        start = full_text.find(marker)
        if start >= 0:
            spans.append((start, len(full_text)))
    return spans


class TraceSFTDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], tokenizer, max_seq_length: int, train_on_inputs: bool, skip_overlength: bool):
        self.records = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_on_inputs = train_on_inputs

        skipped = 0
        for record in records:
            messages = record.get("messages")
            if not isinstance(messages, list):
                text = record.get("text", "")
                messages = [{"role": "assistant", "content": text}]
            full_text = render_with_template(tokenizer, messages)
            token_count = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
            if skip_overlength and token_count > max_seq_length:
                skipped += 1
                continue
            self.records.append({"messages": messages, "meta": record.get("meta", {})})

        if skipped:
            print(f"Skipped {skipped} overlength samples.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        record = self.records[index]
        messages = record["messages"]
        meta = record.get("meta", {})
        full_text = render_with_template(self.tokenizer, messages)

        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_offsets_mapping=not self.train_on_inputs,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if self.train_on_inputs:
            labels = list(input_ids)
        else:
            labels = [IGNORE_INDEX] * len(input_ids)
            spans = assistant_char_spans(self.tokenizer, messages, full_text, meta)
            offsets = encoded.get("offset_mapping") or []
            for token_index, (start, end) in enumerate(offsets):
                if end <= start:
                    continue
                if any(start < span_end and end > span_start for span_start, span_end in spans):
                    labels[token_index] = input_ids[token_index]

            if all(label == IGNORE_INDEX for label in labels):
                labels = list(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    processor, tokenizer = load_processor_and_tokenizer(args.model_name_or_path)
    records = load_jsonl(args.train_file)
    dataset = TraceSFTDataset(
        records,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        train_on_inputs=args.train_on_inputs,
        skip_overlength=args.skip_overlength,
    )
    if args.max_train_samples > 0:
        dataset.records = dataset.records[: args.max_train_samples]
    if len(dataset) == 0:
        raise RuntimeError("No training samples remain after preprocessing.")

    lengths = []
    supervised_lengths = []
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        lengths.append(len(sample["input_ids"]))
        supervised_lengths.append(sum(1 for label in sample["labels"] if label != IGNORE_INDEX))
    stats = {
        "samples": len(dataset),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "avg_tokens": round(sum(lengths) / len(lengths), 2),
        "min_supervised_tokens": min(supervised_lengths),
        "max_supervised_tokens": max(supervised_lengths),
        "avg_supervised_tokens": round(sum(supervised_lengths) / len(supervised_lengths), 2),
        "max_seq_length": args.max_seq_length,
    }
    print(json.dumps({"dataset_stats": stats}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    model = load_model(args.model_name_or_path, args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    model = apply_lora(model, args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    if processor is not None:
        processor.save_pretrained(args.output_dir)
    else:
        tokenizer.save_pretrained(args.output_dir)

    print(json.dumps({
        "status": "success",
        "output_dir": os.path.abspath(args.output_dir),
        "train_samples": len(dataset),
        "use_lora": args.use_lora,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
