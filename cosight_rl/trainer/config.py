"""Configuration helpers for Co-Sight critic-free GRPO/RL runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

LANGUAGE_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_LORA_EXCLUDE_MODULES = ".*(visual|vision|blocks|patch_embed|merger).*"


@dataclass(frozen=True)
class RLProfile:
    gpus: int
    train_batch_size: int
    ppo_mini_batch_size: int
    ppo_micro_batch_size_per_gpu: int
    log_prob_micro_batch_size_per_gpu: int
    rollout_tensor_parallel_size: int
    rollout_n: int
    gpu_memory_utilization: float


def build_rl_profile(gpus: int) -> RLProfile:
    """Return conservative defaults for Qwen3-VL-8B LoRA RL."""

    gpus = max(1, min(4, int(gpus)))
    rollout_tp = int(os.getenv("COSIGHT_ROLLOUT_TP", "1"))
    if rollout_tp < 1 or rollout_tp > gpus:
        raise ValueError(
            f"COSIGHT_ROLLOUT_TP must be between 1 and {gpus}, got {rollout_tp}."
        )
    if gpus % rollout_tp != 0:
        raise ValueError(
            f"GPU count {gpus} must be divisible by COSIGHT_ROLLOUT_TP={rollout_tp}."
        )

    train_batch_size = int(os.getenv("COSIGHT_TRAIN_BATCH_SIZE", str(4 * gpus)))
    ppo_mini_batch_size = int(
        os.getenv("COSIGHT_PPO_MINI_BATCH_SIZE", str(min(4 * gpus, train_batch_size)))
    )
    if train_batch_size < 1:
        raise ValueError(f"COSIGHT_TRAIN_BATCH_SIZE must be positive, got {train_batch_size}.")
    if ppo_mini_batch_size < 1:
        raise ValueError(
            f"COSIGHT_PPO_MINI_BATCH_SIZE must be positive, got {ppo_mini_batch_size}."
        )
    if ppo_mini_batch_size > train_batch_size:
        raise ValueError(
            "COSIGHT_PPO_MINI_BATCH_SIZE must be <= COSIGHT_TRAIN_BATCH_SIZE, "
            f"got {ppo_mini_batch_size} > {train_batch_size}."
        )
    ppo_micro_batch_size_per_gpu = int(
        os.getenv("COSIGHT_PPO_MICRO_BATCH_SIZE_PER_GPU", "1")
    )
    log_prob_micro_batch_size_per_gpu = int(
        os.getenv("COSIGHT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "1")
    )
    if ppo_micro_batch_size_per_gpu < 1:
        raise ValueError(
            "COSIGHT_PPO_MICRO_BATCH_SIZE_PER_GPU must be positive, "
            f"got {ppo_micro_batch_size_per_gpu}."
        )
    if log_prob_micro_batch_size_per_gpu < 1:
        raise ValueError(
            "COSIGHT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU must be positive, "
            f"got {log_prob_micro_batch_size_per_gpu}."
        )

    return RLProfile(
        gpus=gpus,
        train_batch_size=train_batch_size,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=ppo_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=log_prob_micro_batch_size_per_gpu,
        rollout_tensor_parallel_size=rollout_tp,
        rollout_n=int(os.getenv("COSIGHT_ROLLOUT_N", "2")),
        gpu_memory_utilization=float(os.getenv("COSIGHT_GPU_MEMORY_UTILIZATION", "0.45")),
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _first_writable_runtime_dir(project_root: Path, suffix: str) -> Path:
    env_name = "COSIGHT_RAY_TMPDIR" if suffix == "ray_tmp" else "COSIGHT_TMPDIR"
    explicit = os.getenv(env_name)
    candidates = [
        Path(explicit) if explicit else None,
        Path(f"/tmp/cosight_{os.getenv('USER', 'u')}_{os.getenv('SLURM_JOB_ID', 'manual')}") / suffix,
        Path(f"/dev/shm/cosight_{os.getenv('USER', 'u')}_{os.getenv('SLURM_JOB_ID', 'manual')}") / suffix,
        project_root / "outputs" / suffix,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(candidate, os.W_OK):
            return candidate
    fallback = project_root / "outputs" / suffix
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _hydra_target_modules(value: str) -> str:
    value = value.strip()
    if value == "all-linear":
        return value
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("LoRA target modules cannot be empty.")
    return "[" + ",".join(modules) + "]"


@dataclass
class GRPOLaunchConfig:
    project_root: Path
    jade_dir: Path
    model_path: Path
    train_file: Path
    val_file: Path
    output_dir: Path
    gpus: int = 1
    exp_name: str = "qwen3-vl-8b-cosight-rl"
    project_name: str = "CoSight-RL"
    max_prompt_length: int = 8192
    max_response_length: int = 8192
    total_epochs: int = 1
    total_training_steps: int = 100
    save_freq: int = 10
    test_freq: int = 10
    lr: str = "1e-6"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: str = LANGUAGE_LORA_TARGET_MODULES
    lora_exclude_modules: str | None = DEFAULT_LORA_EXCLUDE_MODULES
    adv_estimator: str = "grpo"
    logger: str = '["console"]'
    ray_num_cpus: int = 0
    resume_mode: str = "disable"

    @property
    def use_transfer_queue(self) -> bool:
        return _env_bool("COSIGHT_USE_TRANSFER_QUEUE", False)

    @property
    def reward_function_path(self) -> Path:
        return self.project_root / "cosight_rl" / "rewards" / "reward_evaluator.py"

    @property
    def ray_tmp_dir(self) -> Path:
        return _first_writable_runtime_dir(self.project_root, "ray_tmp")

    @property
    def profile(self) -> RLProfile:
        return build_rl_profile(self.gpus)

    def hydra_overrides(self, dry_run_trainer: bool = False) -> List[str]:
        profile = self.profile
        multi_turn = os.getenv("COSIGHT_MULTI_TURN", "false").lower() in {"1", "true", "yes"}
        rollout_backend = os.getenv("COSIGHT_ROLLOUT_BACKEND", "vllm")
        if multi_turn and rollout_backend == "vllm":
            rollout_backend = "sglang"
        total_steps = 1 if dry_run_trainer else self.total_training_steps
        total_epochs = 1 if dry_run_trainer else self.total_epochs
        train_batch_size = 1 if dry_run_trainer else profile.train_batch_size
        ppo_mini_batch_size = (
            min(profile.ppo_mini_batch_size, train_batch_size)
            if dry_run_trainer
            else profile.ppo_mini_batch_size
        )
        ppo_micro_batch_size_per_gpu = min(
            profile.ppo_micro_batch_size_per_gpu,
            ppo_mini_batch_size,
        )
        log_prob_micro_batch_size_per_gpu = min(
            profile.log_prob_micro_batch_size_per_gpu,
            ppo_mini_batch_size,
        )
        rollout_n = 1 if dry_run_trainer else profile.rollout_n
        save_freq = -1 if dry_run_trainer else self.save_freq
        test_freq = -1 if dry_run_trainer else self.test_freq
        dataloader_num_workers = (
            0 if dry_run_trainer else int(os.getenv("COSIGHT_DATALOADER_NUM_WORKERS", "0"))
        )
        ray_num_cpus = self.ray_num_cpus or max(8, min(16, 8 * profile.gpus))
        ray_tmp_dir = self.ray_tmp_dir
        ray_tmp_dir.mkdir(parents=True, exist_ok=True)

        effective_max_prompt_length = self.max_prompt_length
        effective_max_response_length = self.max_response_length
        rollout_gpu_memory_utilization = profile.gpu_memory_utilization
        default_rollout_max_num_seqs = min(8, max(4, train_batch_size * rollout_n))
        rollout_max_num_seqs = int(
            os.getenv("COSIGHT_ROLLOUT_MAX_NUM_SEQS", str(default_rollout_max_num_seqs))
        )
        rollout_enforce_eager = _env_bool("COSIGHT_ROLLOUT_ENFORCE_EAGER", True)
        fsdp_model_dtype = os.getenv("COSIGHT_FSDP_MODEL_DTYPE", "bf16")
        ref_fsdp_model_dtype = os.getenv("COSIGHT_REF_FSDP_MODEL_DTYPE", fsdp_model_dtype)
        disable_vllm_lora = _env_bool("COSIGHT_DISABLE_VLLM_LORA", True)
        disable_torch_compile = _env_bool("COSIGHT_DISABLE_TORCH_COMPILE", True)
        skip_mm_profiling = _env_bool("COSIGHT_VLLM_SKIP_MM_PROFILING", True)
        disable_mm_preprocessor_cache = _env_bool(
            "COSIGHT_DISABLE_MM_PREPROCESSOR_CACHE",
            False,
        )
        enable_prefix_caching = _env_bool("COSIGHT_ENABLE_PREFIX_CACHING", False)
        enable_chunked_prefill = _env_bool("COSIGHT_ENABLE_CHUNKED_PREFILL", True)
        actor_kl_loss_coef = os.getenv(
            "COSIGHT_KL_LOSS_COEF",
            "0.001",
        )
        reward_kl_coef = os.getenv("COSIGHT_REWARD_KL_COEF", "0.001")
        resume_mode = os.getenv("COSIGHT_RESUME_MODE", self.resume_mode).lower()
        if resume_mode not in {"auto", "disable", "resume_path"}:
            raise ValueError("COSIGHT_RESUME_MODE must be one of auto, disable, resume_path.")
        data_truncation = os.getenv("COSIGHT_DATA_TRUNCATION", "left").lower()
        valid_truncation_modes = {"left", "right", "middle", "error"}
        if data_truncation not in valid_truncation_modes:
            raise ValueError(
                "COSIGHT_DATA_TRUNCATION must be one of "
                f"{sorted(valid_truncation_modes)}, got {data_truncation!r}."
            )
        if dry_run_trainer:
            effective_max_prompt_length = min(
                self.max_prompt_length,
                int(os.getenv("COSIGHT_DRY_RUN_MAX_PROMPT_LENGTH", "4096")),
            )
            effective_max_response_length = min(
                self.max_response_length,
                int(os.getenv("COSIGHT_DRY_RUN_MAX_RESPONSE_LENGTH", "2048")),
            )
            rollout_gpu_memory_utilization = float(
                os.getenv("COSIGHT_DRY_RUN_GPU_MEMORY_UTILIZATION", "0.50")
            )
            rollout_max_num_seqs = int(os.getenv("COSIGHT_DRY_RUN_MAX_NUM_SEQS", "4"))
            rollout_enforce_eager = _env_bool("COSIGHT_DRY_RUN_ENFORCE_EAGER", True)
            disable_vllm_lora = _env_bool("COSIGHT_DRY_RUN_DISABLE_VLLM_LORA", disable_vllm_lora)
            skip_mm_profiling = _env_bool("COSIGHT_DRY_RUN_SKIP_MM_PROFILING", True)

        lora_exclude_modules = os.getenv("COSIGHT_LORA_EXCLUDE_MODULES", self.lora_exclude_modules or "")
        use_vllm_lora = self.lora_rank > 0 and rollout_backend == "vllm" and not disable_vllm_lora
        rollout_load_format_setting = os.getenv("COSIGHT_ROLLOUT_LOAD_FORMAT", "auto")
        if rollout_load_format_setting.lower() == "auto":
            rollout_load_format = "safetensors" if use_vllm_lora else "dummy"
        else:
            rollout_load_format = rollout_load_format_setting

        rollout_layered_summon_setting = os.getenv("COSIGHT_ROLLOUT_LAYERED_SUMMON", "auto")
        if rollout_layered_summon_setting.lower() == "auto":
            rollout_layered_summon = use_vllm_lora
        else:
            rollout_layered_summon = _env_bool("COSIGHT_ROLLOUT_LAYERED_SUMMON", use_vllm_lora)

        max_total_len = effective_max_prompt_length + effective_max_response_length
        overrides = [
            "--config-name=transfer_queue_ppo_trainer" if self.use_transfer_queue else "--config-name=ppo_trainer",
            f"data.train_files={self.train_file.as_posix()}",
            f"data.val_files={self.val_file.as_posix()}",
            "data.prompt_key=prompt",
            "data.reward_fn_key=data_source",
            f"data.truncation={data_truncation}",
            "data.filter_overlong_prompts=True",
            "data.trust_remote_code=True",
            f"data.max_prompt_length={effective_max_prompt_length}",
            f"data.max_response_length={effective_max_response_length}",
            f"data.train_batch_size={train_batch_size}",
            f"data.val_batch_size={max(1, min(train_batch_size, 4))}",
            f"data.dataloader_num_workers={dataloader_num_workers}",
            f"algorithm.adv_estimator={self.adv_estimator}",
            "algorithm.use_kl_in_reward=False",
            f"algorithm.kl_ctrl.kl_coef={reward_kl_coef}",
            f"actor_rollout_ref.model.path={self.model_path.as_posix()}",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            f"actor_rollout_ref.model.lora_rank={self.lora_rank}",
            f"actor_rollout_ref.model.lora_alpha={self.lora_alpha}",
            f"actor_rollout_ref.model.target_modules={_hydra_target_modules(self.lora_target_modules)}",
            f"actor_rollout_ref.actor.optim.lr={self.lr}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={ppo_mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="
            f"{ppo_micro_batch_size_per_gpu}",
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={actor_kl_loss_coef}",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            "actor_rollout_ref.actor.entropy_coeff=0",
            "actor_rollout_ref.actor.use_dynamic_bsz=True",
            f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={max_total_len}",
            "actor_rollout_ref.actor.fsdp_config.param_offload=True",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            "actor_rollout_ref.ref.fsdp_config.param_offload=True",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="
            f"{log_prob_micro_batch_size_per_gpu}",
            f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={max_total_len}",
            f"actor_rollout_ref.rollout.name={rollout_backend}",
            f"actor_rollout_ref.rollout.n={rollout_n}",
            "actor_rollout_ref.rollout.dtype=bfloat16",
            "actor_rollout_ref.rollout.temperature=0.8",
            "actor_rollout_ref.rollout.top_p=0.95",
            "actor_rollout_ref.rollout.top_k=-1",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={profile.rollout_tensor_parallel_size}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_gpu_memory_utilization}",
            f"actor_rollout_ref.rollout.disable_lora={disable_vllm_lora}",
            f"actor_rollout_ref.rollout.enforce_eager={rollout_enforce_eager}",
            f"actor_rollout_ref.rollout.max_model_len={max_total_len}",
            f"actor_rollout_ref.rollout.max_num_seqs={rollout_max_num_seqs}",
            f"actor_rollout_ref.rollout.max_num_batched_tokens={max_total_len}",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="
            f"{log_prob_micro_batch_size_per_gpu}",
            f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={max_total_len}",
            f"actor_rollout_ref.rollout.enable_chunked_prefill={enable_chunked_prefill}",
            f"actor_rollout_ref.rollout.enable_prefix_caching={enable_prefix_caching}",
            "actor_rollout_ref.rollout.val_kwargs.temperature=0",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
            "reward_model.reward_manager=naive",
            f"custom_reward_function.path={self.reward_function_path.as_posix()}",
            "custom_reward_function.name=compute_score",
            f"trainer.project_name={self.project_name}",
            f"trainer.experiment_name={self.exp_name}",
            f"trainer.n_gpus_per_node={profile.gpus}",
            "trainer.nnodes=1",
            f"trainer.default_local_dir={self.output_dir.as_posix()}",
            f"trainer.logger={self.logger}",
            f"trainer.total_epochs={total_epochs}",
            f"trainer.total_training_steps={total_steps}",
            f"trainer.save_freq={save_freq}",
            f"trainer.test_freq={test_freq}",
            "trainer.critic_warmup=0",
            "trainer.val_before_train=False",
            f"trainer.resume_mode={resume_mode}",
            "trainer.device=cuda",
            f"ray_kwargs.ray_init.num_cpus={ray_num_cpus}",
            "ray_kwargs.ray_init.include_dashboard=False",
            f"++ray_kwargs.ray_init._temp_dir={ray_tmp_dir.as_posix()}",
        ]
        if self.use_transfer_queue:
            overrides.extend(
                [
                    "++trainer.num_global_batch=1",
                    "++trainer.num_data_storage_units=1",
                    "++trainer.num_data_controllers=1",
                ]
            )
        if lora_exclude_modules:
            overrides.append(f"actor_rollout_ref.model.exclude_modules='{lora_exclude_modules}'")
        if fsdp_model_dtype:
            overrides.append(f"actor_rollout_ref.actor.fsdp_config.model_dtype={fsdp_model_dtype}")
        if ref_fsdp_model_dtype:
            overrides.append(f"actor_rollout_ref.ref.fsdp_config.model_dtype={ref_fsdp_model_dtype}")
        if disable_torch_compile:
            overrides.extend(
                [
                    "actor_rollout_ref.actor.use_torch_compile=False",
                    "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
                    "actor_rollout_ref.ref.use_torch_compile=False",
                    "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False",
                ]
            )
        if rollout_load_format:
            overrides.append(f"actor_rollout_ref.rollout.load_format={rollout_load_format}")
        if rollout_layered_summon:
            overrides.append("actor_rollout_ref.rollout.layered_summon=True")
        if skip_mm_profiling:
            overrides.append("++actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=True")
        if disable_mm_preprocessor_cache:
            overrides.append(
                "++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True"
            )
        if multi_turn:
            tool_config_path = os.getenv("COSIGHT_TOOL_CONFIG_PATH", "")
            if tool_config_path:
                overrides.extend(
                    [
                        "actor_rollout_ref.rollout.multi_turn.enable=True",
                        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
                        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={tool_config_path}",
                        f"actor_rollout_ref.rollout.multi_turn.max_user_turns={os.getenv('COSIGHT_MAX_USER_TURNS', '6')}",
                        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={os.getenv('COSIGHT_MAX_ASSISTANT_TURNS', '8')}",
                        "actor_rollout_ref.rollout.multi_turn.format=hermes",
                    ]
                )
            else:
                raise ValueError("COSIGHT_MULTI_TURN=true requires COSIGHT_TOOL_CONFIG_PATH.")
        return overrides


H20Profile = RLProfile
PPOLaunchConfig = GRPOLaunchConfig
build_h20_profile = build_rl_profile
