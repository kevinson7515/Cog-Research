#!/usr/bin/env python3
"""Merge Co-Sight workflow LoRA and second-stage vision LoRA in order."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--workflow_adapter_path", default="outputs/qwen3-vl-8b-cosight-lora")
    parser.add_argument("--vision_adapter_path", default="outputs/qwen3-vl-8b-cosight-vision-lora-strong")
    parser.add_argument("--output_dir", default="outputs/qwen3-vl-8b-cosight-merged-vision-strong")
    parser.add_argument("--bf16", action="store_true", help="Deprecated compatibility flag. Prefer --merge_dtype bf16.")
    parser.add_argument(
        "--merge_dtype",
        choices=["fp32", "bf16", "fp16"],
        default=None,
        help="Dtype used while applying LoRA deltas. Default is fp32 unless deprecated --bf16 is set.",
    )
    parser.add_argument(
        "--save_dtype",
        choices=["same", "fp32", "bf16", "fp16"],
        default="bf16",
        help="Dtype used for the saved merged model. Use fp32 to preserve the smallest LoRA deltas for diagnosis.",
    )
    parser.add_argument("--vision_merge_scale", type=float, default=1.0, help="Optional multiplier applied only to the vision LoRA before merge.")
    parser.add_argument("--disable_bnb", action="store_true", default=True)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Where to load and merge the model. Use cuda to avoid CPU-memory OOM when a large GPU is available.",
    )
    parser.add_argument("--max_shard_size", default="4GB")
    parser.add_argument("--allow_missing_adapter_keys", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def selected_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return device_arg


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


def load_with_dtype_fallback(model_cls: Any, model_path: str, kwargs: Dict[str, Any], dtype: torch.dtype):
    try:
        return model_cls.from_pretrained(model_path, trust_remote_code=True, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return model_cls.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype, **kwargs)


def load_base_model(model_path: str, dtype: torch.dtype, device: str):
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
    kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
    if device == "cuda":
        kwargs["device_map"] = {"": "cuda:0"}
    for model_cls in model_classes:
        try:
            return load_with_dtype_fallback(model_cls, model_path, kwargs, dtype)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load base model. Last error: {last_error}")


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


def load_tensor_file(path: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Reading safetensors adapter files requires safetensors.") from exc
        return load_file(os.fspath(path), device="cpu")

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inspect_adapter_checkpoint(adapter_path: str | Path) -> Tuple[str, int]:
    adapter_path = Path(adapter_path)
    candidates = [
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_model.bin",
    ]
    for path in candidates:
        if path.exists():
            keys = load_tensor_keys(path)
            lora_keys = [key for key in keys if "lora_" in key]
            if not lora_keys:
                raise RuntimeError(f"Adapter checkpoint has no LoRA tensors: {path}")
            return os.fspath(path), len(lora_keys)
    raise FileNotFoundError(f"No adapter weight file found in {adapter_path}")


def adapter_scaling(adapter_path: str | Path, scale_multiplier: float) -> float:
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        return scale_multiplier
    config = json.loads(config_path.read_text(encoding="utf-8"))
    r = config.get("r") or 1
    lora_alpha = config.get("lora_alpha") or r
    if isinstance(r, dict):
        r = next(iter(r.values()), 1)
    if isinstance(lora_alpha, dict):
        lora_alpha = next(iter(lora_alpha.values()), r)
    return float(lora_alpha) / max(1.0, float(r)) * scale_multiplier


def summarize_adapter_delta(adapter_weight_file: str | Path, adapter_path: str | Path, scale_multiplier: float) -> Dict[str, Any]:
    state_dict = load_tensor_file(adapter_weight_file)
    scaling = adapter_scaling(adapter_path, scale_multiplier)
    a_tensors = {name: tensor for name, tensor in state_dict.items() if ".lora_A." in name and tensor.ndim == 2}
    total_numel = 0
    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    delta_count = 0
    skipped_without_b = 0

    for a_name, a_tensor in a_tensors.items():
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        b_tensor = state_dict.get(b_name)
        if b_tensor is None or b_tensor.ndim != 2:
            skipped_without_b += 1
            continue
        delta = torch.matmul(b_tensor.float(), a_tensor.float()).mul_(scaling)
        abs_delta = delta.abs()
        total_numel += delta.numel()
        sum_abs += abs_delta.sum().item()
        sum_sq += delta.square().sum().item()
        max_abs = max(max_abs, abs_delta.max().item() if delta.numel() else 0.0)
        delta_count += 1

    return {
        "delta_tensor_count": delta_count,
        "delta_numel": total_numel,
        "delta_max_abs": max_abs,
        "delta_mean_abs": sum_abs / total_numel if total_numel else 0.0,
        "delta_rms": math.sqrt(sum_sq / total_numel) if total_numel else 0.0,
        "scale_multiplier": scale_multiplier,
        "effective_lora_scaling": scaling,
        "skipped_without_lora_b": skipped_without_b,
    }


def scale_lora_before_merge(peft_model, scale_multiplier: float, label: str) -> None:
    if scale_multiplier == 1.0:
        return
    changed = 0
    for module in peft_model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for adapter_name in list(scaling):
                scaling[adapter_name] *= scale_multiplier
                changed += 1
        elif scaling is not None:
            module.scaling = scaling * scale_multiplier
            changed += 1
    print(f"Applied {label} LoRA merge scale multiplier {scale_multiplier} to {changed} LoRA modules.")


def merge_one_adapter(
    model,
    adapter_path: str,
    device: str,
    allow_missing_adapter_keys: bool,
    label: str,
    merge_scale: float = 1.0,
):
    from peft import PeftModel

    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"{label} adapter path does not exist: {adapter_path}")
    weight_file, lora_tensors = inspect_adapter_checkpoint(adapter)
    print(f"{label} adapter checkpoint: {weight_file}")
    print(f"{label} adapter LoRA tensors: {lora_tensors}")
    if label == "vision":
        delta_stats = summarize_adapter_delta(weight_file, adapter, merge_scale)
        print(json.dumps({f"{label}_adapter_delta_stats": delta_stats}, ensure_ascii=False, indent=2))

    kwargs: Dict[str, Any] = {}
    if device == "cuda":
        kwargs["device_map"] = {"": "cuda:0"}

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            peft_model = PeftModel.from_pretrained(model, adapter_path, low_cpu_mem_usage=True, **kwargs)
        except TypeError:
            peft_model = PeftModel.from_pretrained(model, adapter_path, **kwargs)

    missing_key_warnings = [
        warning
        for warning in caught_warnings
        if "missing adapter keys" in str(warning.message).lower()
    ]
    if missing_key_warnings and not allow_missing_adapter_keys:
        raise RuntimeError(
            f"PEFT reported missing adapter keys while loading {label}. "
            "The merged model would likely ignore some LoRA weights, so the merge was aborted."
        )
    for warning in caught_warnings:
        warnings.warn(warning.message, warning.category)

    scale_lora_before_merge(peft_model, merge_scale, label)
    merged = peft_model.merge_and_unload()
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"Merged {label} adapter into the model.")
    return merged


def save_processor(base_model: str, output_dir: str) -> None:
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    except Exception:
        processor = None
    if processor is not None:
        processor.save_pretrained(output_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)


def main() -> None:
    args = parse_args()
    if args.disable_bnb:
        disable_bnb_dispatchers()
    merge_dtype_name = args.merge_dtype or ("bf16" if args.bf16 else "fp32")
    merge_dtype = dtype_from_name(merge_dtype_name)
    save_dtype = merge_dtype if args.save_dtype == "same" else dtype_from_name(args.save_dtype)
    device = selected_device(args.device)

    print(f"Merge device: {device}")
    print(f"Merge dtype: {merge_dtype_name}")
    print(f"Save dtype: {args.save_dtype}")
    print(f"Vision merge scale: {args.vision_merge_scale}")
    print(f"Base model: {args.base_model}")
    print(f"Workflow adapter: {args.workflow_adapter_path}")
    print(f"Vision adapter: {args.vision_adapter_path}")
    print(f"Output dir: {args.output_dir}")

    model = load_base_model(args.base_model, merge_dtype, device)
    model = merge_one_adapter(model, args.workflow_adapter_path, device, args.allow_missing_adapter_keys, "workflow")
    model = merge_one_adapter(
        model,
        args.vision_adapter_path,
        device,
        args.allow_missing_adapter_keys,
        "vision",
        merge_scale=args.vision_merge_scale,
    )
    if save_dtype != merge_dtype:
        print(f"Casting merged model from {merge_dtype} to {save_dtype} before saving.")
        model.to(dtype=save_dtype)
        if device == "cuda":
            torch.cuda.synchronize()
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    save_processor(args.base_model, args.output_dir)

    print(f"Stagewise merged model saved to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
