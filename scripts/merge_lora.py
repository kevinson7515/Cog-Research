#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save a standalone checkpoint."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter_path", default="outputs/qwen3-vl-8b-cosight-lora")
    parser.add_argument("--output_dir", default="outputs/qwen3-vl-8b-cosight-merged")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--disable_bnb", action="store_true", default=True)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Where to load and merge the model. Use cuda to avoid Slurm CPU-memory OOM when a large GPU is available.",
    )
    parser.add_argument("--max_shard_size", default="4GB", help="Shard size used by save_pretrained.")
    parser.add_argument(
        "--allow_missing_adapter_keys",
        action="store_true",
        help="Allow PEFT missing-adapter-key warnings. By default these are treated as merge failures.",
    )
    return parser.parse_args()


def selected_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return device_arg


def load_with_dtype_fallback(model_cls: Any, model_path: str, kwargs: Dict[str, Any], dtype: torch.dtype):
    try:
        return model_cls.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            **kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return model_cls.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            **kwargs,
        )


def load_base_model(model_path: str, dtype: torch.dtype, device: str):
    model_classes = [AutoModelForCausalLM]
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


def load_adapter(model, adapter_path: str, device: str, allow_missing_adapter_keys: bool):
    from peft import PeftModel

    kwargs: Dict[str, Any] = {}
    if device == "cuda":
        kwargs["device_map"] = {"": "cuda:0"}
    weight_file, lora_tensors = inspect_adapter_checkpoint(adapter_path)
    print(f"Adapter checkpoint: {weight_file}")
    print(f"Adapter LoRA tensors: {lora_tensors}")

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
            "PEFT reported missing adapter keys while loading the adapter. "
            "The merged model would likely ignore some or all LoRA weights, so the merge was aborted."
        )
    for warning in caught_warnings:
        warnings.warn(warning.message, warning.category)

    return peft_model


def disable_bnb_dispatchers() -> None:
    # Matching train_sft.py: this LoRA adapter is bf16, not 4bit/8bit.
    # A CUDA-mismatched bitsandbytes package should not block merging.
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


def main() -> None:
    args = parse_args()
    if args.disable_bnb:
        disable_bnb_dispatchers()
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    device = selected_device(args.device)
    print(f"Merge device: {device}")
    print(f"Save max shard size: {args.max_shard_size}")

    model = load_base_model(args.base_model, dtype, device)
    model = load_adapter(model, args.adapter_path, device, args.allow_missing_adapter_keys)
    model = model.merge_and_unload()
    if device == "cuda":
        torch.cuda.synchronize()
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    except Exception:
        processor = None
    if processor is not None:
        processor.save_pretrained(args.output_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
