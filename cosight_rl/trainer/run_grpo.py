"""Launch Co-Sight RL with GRPO-style advantage estimation.

JADE/verl still uses ``verl.trainer.main_ppo`` as the engine entrypoint; this
module keeps the Co-Sight command name aligned with the actual critic-free
``algorithm.adv_estimator=grpo`` setup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from cosight_rl.envs.cosight_env import CoSightRLEnv, CosightEnvConfig, load_jsonl_tasks
from cosight_rl.envs.trajectory import write_trajectories_jsonl, write_unified_buffer_jsonl

from .config import GRPOLaunchConfig, LANGUAGE_LORA_TARGET_MODULES


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value)


def _repair_empty_project_root_expansion(project_root: Path, value: str, arg_name: str) -> str:
    """Repair paths like /outputs/foo produced by an empty ${PROJECT_ROOT}.

    On the training cluster, /outputs is a read-only root path. The intended
    value for Co-Sight jobs is almost always ${PROJECT_ROOT}/outputs/foo. This
    guard catches the common sbatch-export mistake early and keeps checkpoint
    saves inside the project workspace.
    """

    normalized = value.replace("\\", "/")
    if os.name != "nt" and (normalized == "/outputs" or normalized.startswith("/outputs/")):
        repaired = str(project_root / normalized.lstrip("/"))
        print(
            json.dumps(
                {
                    "warning": "repaired_empty_project_root_path",
                    "arg": arg_name,
                    "value": value,
                    "repaired": repaired,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return repaired
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--jade-dir", default="JADE")
    parser.add_argument(
        "--model-path",
        default=os.getenv("COSIGHT_MODEL_PATH", "outputs/qwen3-vl-8b-cosight-merged"),
    )
    parser.add_argument("--train-jsonl", default=os.getenv("COSIGHT_TRAIN_JSONL", "data/quiz_train_rubric.jsonl"))
    parser.add_argument("--val-jsonl", default=os.getenv("COSIGHT_VAL_JSONL", ""))
    parser.add_argument("--val-ratio", type=float, default=float(os.getenv("COSIGHT_VAL_RATIO", "0.1")))
    parser.add_argument("--image-base-dir", default=os.getenv("COSIGHT_IMAGE_BASE_DIR", "data/images"))
    parser.add_argument("--data-dir", default=os.getenv("COSIGHT_DATA_DIR", "data/rl_cosight_rubric"))
    parser.add_argument("--workspace-dir", default=os.getenv("COSIGHT_WORKSPACE_DIR", "work_space/work_space_20260612_134333"))
    parser.add_argument("--trace-dir", default=os.getenv("COSIGHT_TRACE_DIR", "trace"))
    parser.add_argument("--evidence-output-dir", default=os.getenv("COSIGHT_EVIDENCE_DIR", "data/rl_evidence_corpus"))
    parser.add_argument("--evidence-corpus", default=os.getenv("COSIGHT_EVIDENCE_CORPUS", "data/rl_evidence_corpus/task_evidence.json"))
    parser.add_argument("--evidence-max-tasks", type=int, default=int(os.getenv("COSIGHT_EVIDENCE_MAX_TASKS", "0")))
    parser.add_argument("--max-evidence-items", type=int, default=int(os.getenv("COSIGHT_MAX_EVIDENCE_ITEMS", "6")))
    parser.add_argument("--max-evidence-chars", type=int, default=int(os.getenv("COSIGHT_MAX_EVIDENCE_CHARS", "4000")))
    parser.add_argument("--min-evidence-coverage", type=float, default=float(os.getenv("COSIGHT_MIN_EVIDENCE_COVERAGE", "0.75")))
    parser.add_argument("--max-evidence-docs-per-task", type=int, default=int(os.getenv("COSIGHT_MAX_EVIDENCE_DOCS_PER_TASK", "10")))
    parser.add_argument("--max-evidence-doc-chars", type=int, default=int(os.getenv("COSIGHT_MAX_EVIDENCE_DOC_CHARS", "2400")))
    parser.add_argument("--max-trace-evidence-per-task", type=int, default=int(os.getenv("COSIGHT_MAX_TRACE_EVIDENCE_PER_TASK", "4")))
    parser.add_argument("--build-evidence-corpus", action="store_true", default=_env_bool("COSIGHT_BUILD_EVIDENCE_CORPUS", True))
    parser.add_argument("--no-build-evidence-corpus", dest="build_evidence_corpus", action="store_false")
    parser.add_argument("--include-trace-evidence", action="store_true", default=_env_bool("COSIGHT_INCLUDE_TRACE_EVIDENCE", True))
    parser.add_argument("--no-include-trace-evidence", dest="include_trace_evidence", action="store_false")
    parser.add_argument("--include-final-report-evidence", action="store_true", default=_env_bool("COSIGHT_INCLUDE_FINAL_REPORT_EVIDENCE", False))
    parser.add_argument("--include-evidence-in-prompt", action="store_true", default=_env_bool("COSIGHT_INCLUDE_EVIDENCE_IN_PROMPT", True))
    parser.add_argument("--no-include-evidence-in-prompt", dest="include_evidence_in_prompt", action="store_false")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("COSIGHT_OUTPUT_DIR", "outputs/qwen3-vl-8b-rl-evidence-lora"),
    )
    parser.add_argument("--gpus", type=int, default=int(os.getenv("GPU_COUNT", os.getenv("NGPUS_PER_NODE", "1"))))
    parser.add_argument("--max-train-samples", type=int, default=int(os.getenv("COSIGHT_MAX_TRAIN_SAMPLES", "0")))
    parser.add_argument("--max-val-samples", type=int, default=int(os.getenv("COSIGHT_MAX_VAL_SAMPLES", "0")))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.getenv("COSIGHT_MAX_PROMPT_LENGTH", "8192")))
    parser.add_argument("--max-response-length", type=int, default=int(os.getenv("COSIGHT_MAX_RESPONSE_LENGTH", "8192")))
    parser.add_argument("--total-training-steps", type=int, default=int(os.getenv("COSIGHT_TOTAL_TRAINING_STEPS", "100")))
    parser.add_argument("--total-epochs", type=int, default=_optional_env_int("COSIGHT_TOTAL_EPOCHS"))
    parser.add_argument("--save-freq", type=int, default=int(os.getenv("COSIGHT_SAVE_FREQ", "10")))
    parser.add_argument("--test-freq", type=int, default=int(os.getenv("COSIGHT_TEST_FREQ", "20")))
    parser.add_argument("--adv-estimator", default=os.getenv("COSIGHT_ADV_ESTIMATOR", "grpo"))
    parser.add_argument("--lora-rank", type=int, default=int(os.getenv("COSIGHT_LORA_RANK", "16")))
    parser.add_argument("--lora-alpha", type=int, default=int(os.getenv("COSIGHT_LORA_ALPHA", "32")))
    parser.add_argument(
        "--lora-target-modules",
        default=os.getenv("COSIGHT_LORA_TARGET_MODULES", LANGUAGE_LORA_TARGET_MODULES),
    )
    parser.add_argument("--lr", default=os.getenv("COSIGHT_LR", "1e-6"))
    parser.add_argument("--exp-name", default=os.getenv("COSIGHT_EXP_NAME", "qwen3-vl-8b-cosight-grpo-rubric"))
    parser.add_argument("--resume-mode", default=os.getenv("COSIGHT_RESUME_MODE", "disable"), choices=["auto", "disable", "resume_path"])
    parser.add_argument("--ray-num-cpus", type=int, default=int(os.getenv("COSIGHT_RAY_NUM_CPUS", "0")))
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run-trainer", action="store_true", help="Build/run a 1-step trainer smoke command.")
    parser.add_argument("--dry-run-env", action="store_true", help="No-GPU task loader + reward + trajectory smoke test.")
    parser.add_argument("--dry-run-output", default="outputs/cosight_rl_dry_run/trajectories.jsonl")
    args = parser.parse_args()
    if args.total_epochs is None:
        # verl's loop is still epoch-bounded even when trainer.total_training_steps
        # is set. Give it enough epochs and let total_training_steps be the stop.
        args.total_epochs = max(1, args.total_training_steps)
    project_root = Path(args.project_root).expanduser().resolve()
    args.output_dir = _repair_empty_project_root_expansion(project_root, args.output_dir, "--output-dir")
    args.model_path = _repair_empty_project_root_expansion(project_root, args.model_path, "--model-path")
    return args


def _project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _project_path_or_empty(root: Path, value: str) -> str:
    if not value:
        return ""
    return str(_project_path(root, value))


def build_evidence_corpus(args: argparse.Namespace, project_root: Path) -> None:
    if not args.build_evidence_corpus:
        return

    evidence_path = _project_path(project_root, args.evidence_corpus)
    if evidence_path.exists() and not _env_bool("COSIGHT_REBUILD_EVIDENCE_CORPUS", False):
        print(json.dumps({"evidence_corpus": str(evidence_path), "status": "reuse_existing"}, ensure_ascii=False))
        return

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "build_cosight_evidence_corpus.py"),
        "--workspace_dir",
        str(_project_path(project_root, args.workspace_dir)),
        "--trace_dir",
        str(_project_path(project_root, args.trace_dir)),
        "--output_dir",
        str(_project_path(project_root, args.evidence_output_dir)),
        "--max_tasks",
        str(args.evidence_max_tasks),
        "--max_docs_per_task",
        str(args.max_evidence_docs_per_task),
        "--max_chars_per_doc",
        str(args.max_evidence_doc_chars),
        "--max_trace_items_per_task",
        str(args.max_trace_evidence_per_task),
    ]
    if not args.include_trace_evidence:
        cmd.append("--no_include_trace")
    if args.include_final_report_evidence:
        cmd.append("--include_final_reports")
    subprocess.run(cmd, cwd=project_root, check=True)


def prepare_data(args: argparse.Namespace, project_root: Path) -> None:
    data_dir = _project_path(project_root, args.data_dir)
    train_file = data_dir / "train.parquet"
    val_file = data_dir / "val.parquet"
    if args.skip_prepare:
        return
    build_evidence_corpus(args, project_root)
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "prepare_cosight_rl_data.py"),
        "--train_jsonl",
        str(_project_path(project_root, args.train_jsonl)),
        "--val_jsonl",
        _project_path_or_empty(project_root, args.val_jsonl),
        "--output_dir",
        str(data_dir),
        "--image_base_dir",
        str(_project_path(project_root, args.image_base_dir)),
        "--val_ratio",
        str(args.val_ratio),
        "--evidence_corpus",
        str(_project_path(project_root, args.evidence_corpus)),
        "--max_evidence_items",
        str(args.max_evidence_items),
        "--max_evidence_chars",
        str(args.max_evidence_chars),
        "--min_evidence_coverage",
        str(args.min_evidence_coverage if args.include_evidence_in_prompt else 0.0),
        "--jsonl_copy",
    ]
    if args.include_evidence_in_prompt:
        cmd.append("--include_evidence_in_prompt")
    if args.max_train_samples:
        cmd.extend(["--max_train_samples", str(args.max_train_samples)])
    if args.max_val_samples:
        cmd.extend(["--max_val_samples", str(args.max_val_samples)])
    subprocess.run(cmd, cwd=project_root, check=True)


def build_launch_config(args: argparse.Namespace, project_root: Path) -> GRPOLaunchConfig:
    data_dir = _project_path(project_root, args.data_dir)
    return GRPOLaunchConfig(
        project_root=project_root,
        jade_dir=_project_path(project_root, args.jade_dir),
        model_path=_project_path(project_root, args.model_path),
        train_file=data_dir / "train.parquet",
        val_file=data_dir / "val.parquet",
        output_dir=_project_path(project_root, args.output_dir),
        gpus=args.gpus,
        exp_name=args.exp_name,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        total_epochs=args.total_epochs,
        total_training_steps=args.total_training_steps,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        adv_estimator=args.adv_estimator,
        ray_num_cpus=args.ray_num_cpus,
        resume_mode=args.resume_mode,
    )


def build_command(config: GRPOLaunchConfig, dry_run_trainer: bool = False) -> List[str]:
    module = "recipe.transfer_queue.main_ppo" if config.use_transfer_queue else "verl.trainer.main_ppo"
    return [sys.executable, "-m", module, *config.hydra_overrides(dry_run_trainer)]


def sanitize_cuda_allocator_env(env: dict[str, str]) -> None:
    conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if env.get("COSIGHT_KEEP_PYTORCH_CUDA_ALLOC_CONF", "0") == "1":
        return
    if "expandable_segments:True" in conf:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def _usable_executable(path: str | None) -> bool:
    return bool(path) and Path(path).exists() and os.access(path, os.X_OK)


def _select_compiler(env: dict[str, str], env_key: str, override_key: str, binary: str) -> str:
    for candidate in (env.get(override_key), env.get(env_key), shutil.which(binary), f"/usr/bin/{binary}"):
        if _usable_executable(candidate):
            return str(candidate)
    return f"/usr/bin/{binary}"


def sanitize_compiler_env(env: dict[str, str], project_root: Path) -> None:
    cc = _select_compiler(env, "CC", "COSIGHT_CC", "gcc")
    cxx = _select_compiler(env, "CXX", "COSIGHT_CXX", "g++")
    cuda_host_cxx = env.get("COSIGHT_CUDAHOSTCXX") or env.get("CUDAHOSTCXX")
    if not _usable_executable(cuda_host_cxx):
        cuda_host_cxx = cxx

    env["CC"] = cc
    env["CXX"] = cxx
    env["CUDAHOSTCXX"] = str(cuda_host_cxx)
    env["COSIGHT_CC"] = cc
    env["COSIGHT_CXX"] = cxx
    env["COSIGHT_CUDAHOSTCXX"] = str(cuda_host_cxx)
    env.setdefault("TRITON_CACHE_DIR", str(project_root / "outputs" / ".triton_cache"))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(project_root / "outputs" / ".torch_extensions"))
    Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)


def configure_runtime_dirs(env: dict[str, str], project_root: Path) -> None:
    from .config import _first_writable_runtime_dir

    ray_tmp = _first_writable_runtime_dir(project_root, "ray_tmp")
    tmp = _first_writable_runtime_dir(project_root, "tmp")
    ray_tmp.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    env["COSIGHT_RAY_TMPDIR"] = str(ray_tmp)
    env["RAY_TMPDIR"] = str(ray_tmp)
    env["COSIGHT_TMPDIR"] = str(tmp)
    env["TMPDIR"] = str(tmp)


def run_env_dry_run(args: argparse.Namespace, project_root: Path) -> None:
    tasks = load_jsonl_tasks(_project_path(project_root, args.train_jsonl), _project_path(project_root, args.image_base_dir))
    if args.max_train_samples:
        tasks = tasks[: args.max_train_samples]
    else:
        tasks = tasks[:2]
    env = CoSightRLEnv(CosightEnvConfig(workspace_root=_project_path(project_root, "outputs/cosight_rl_dry_run/workspace")))
    trajectories = [env.dry_run_task(task) for task in tasks]
    output = _project_path(project_root, args.dry_run_output)
    write_trajectories_jsonl(trajectories, output)
    unified_output = output.with_suffix(".unified.jsonl")
    write_unified_buffer_jsonl(trajectories, unified_output)
    summary = {
        "tasks": len(trajectories),
        "output": str(output),
        "unified_output": str(unified_output),
        "rewards": [trajectory.final_reward for trajectory in trajectories],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    if args.dry_run_env:
        run_env_dry_run(args, project_root)
        return

    prepare_data(args, project_root)
    if args.prepare_only:
        return

    config = build_launch_config(args, project_root)
    cmd = build_command(config, dry_run_trainer=args.dry_run_trainer)
    env = os.environ.copy()
    pythonpath = [str(project_root), str(config.jade_dir), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath if part)
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    sanitize_cuda_allocator_env(env)
    sanitize_compiler_env(env, project_root)
    configure_runtime_dirs(env, project_root)
    if os.getenv("COSIGHT_KEEP_ROCM_VISIBLE_DEVICES", "0") != "1":
        env.pop("ROCR_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)

    printable = " ".join(shlex.quote(part) for part in cmd)
    if args.print_command or not args.run:
        print(printable)
    if args.run:
        subprocess.run(cmd, cwd=config.jade_dir, env=env, check=True)


if __name__ == "__main__":
    main()
