#!/usr/bin/env bash

# Shared controls for Conditions B and C. The two launchers may differ only in
# rubric/data mode and output names; all rollout and optimizer controls live
# here so the comparison cannot silently drift.

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  echo "ERROR: PROJECT_ROOT must be set before sourcing trajectory_ablation_common.sh" >&2
  return 2
fi

export COSIGHT_TRAIN_JSONL="${PROJECT_ROOT}/data/quiz_train_rubric.jsonl"
export COSIGHT_VAL_JSONL=""
export COSIGHT_INCLUDE_EVIDENCE_IN_PROMPT="0"
export COSIGHT_MIN_EVIDENCE_COVERAGE="${COSIGHT_MIN_EVIDENCE_COVERAGE:-0.75}"
# The cluster-pinned vLLM 0.11 async external executor is unstable for Qwen3-VL
# multimodal rollouts. Keep the visual observation fixed and hidden behind
# cosight_step.inspect_visual, and train the complete policy-generated
# Plan-and-Act trajectory without invoking the incompatible raw-pixel path.
# Conditions B and C share this path.
export COSIGHT_DISABLE_RAW_IMAGE_ROLLOUT="1"
export COSIGHT_MIN_VISUAL_EVIDENCE_COVERAGE="${COSIGHT_MIN_VISUAL_EVIDENCE_COVERAGE:-0.95}"
# JADE's async rollout server is implemented with vLLM's V1 AsyncLLM API.
# Keep that engine selection consistent, and disable the Qwen3-VL text-only
# deepstack path inside the worker before model load.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export COSIGHT_VLLM_DISABLE_QWEN3VL_DEEPSTACK="${COSIGHT_VLLM_DISABLE_QWEN3VL_DEEPSTACK:-1}"
export COSIGHT_ENABLE_CHUNKED_PREFILL="${COSIGHT_ENABLE_CHUNKED_PREFILL:-0}"
export COSIGHT_DATA_SEED="${COSIGHT_DATA_SEED:-42}"
export COSIGHT_MULTI_TURN="1"

# This cluster's vLLM V1 build cannot consume VERL's in-memory
# TensorLoRARequest. Full-parameter FSDP synchronizes the trainable language
# weights before each rollout and therefore stays on-policy without dynamic
# rollout LoRA. The actor vision tower remains frozen; the raw-image rollout
# path is disabled above because freezing alone does not suppress inference.
export COSIGHT_ROLLOUT_BACKEND="${COSIGHT_ROLLOUT_BACKEND:-vllm}"
export COSIGHT_LORA_RANK="0"
export COSIGHT_FREEZE_VISION_TOWER="1"
export COSIGHT_DISABLE_VLLM_LORA="1"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="false"
export COSIGHT_ROLLOUT_LOAD_FORMAT="safetensors"
# vLLM sleep level 1 keeps checkpoint weights in CPU memory while the actor
# trains. Only trainable policy weights change between rollouts.
export COSIGHT_ROLLOUT_LAYERED_SUMMON="1"

export COSIGHT_TOOL_CONFIG_PATH="${PROJECT_ROOT}/cosight_rl/config/cosight_step_tools.yaml"
export COSIGHT_AGENT_LOOP_CONFIG_PATH="${PROJECT_ROOT}/cosight_rl/config/cosight_plan_act_agent_loop.yaml"
export COSIGHT_DEFAULT_AGENT_LOOP="cosight_plan_act_agent"
export COSIGHT_MAX_ASSISTANT_TURNS="${COSIGHT_MAX_ASSISTANT_TURNS:-20}"
export COSIGHT_MAX_USER_TURNS="${COSIGHT_MAX_USER_TURNS:-20}"
export COSIGHT_MAX_PARALLEL_CALLS="${COSIGHT_MAX_PARALLEL_CALLS:-4}"
export COSIGHT_MAX_TOOL_RESPONSE_LENGTH="${COSIGHT_MAX_TOOL_RESPONSE_LENGTH:-2400}"
export COSIGHT_MAX_RESPONSE_LENGTH="${COSIGHT_MAX_RESPONSE_LENGTH:-16384}"

export COSIGHT_TRAIN_BATCH_SIZE="${COSIGHT_TRAIN_BATCH_SIZE:-6}"
export COSIGHT_PPO_MINI_BATCH_SIZE="${COSIGHT_PPO_MINI_BATCH_SIZE:-6}"
export COSIGHT_ROLLOUT_N="${COSIGHT_ROLLOUT_N:-2}"
export COSIGHT_TOTAL_TRAINING_STEPS="${COSIGHT_TOTAL_TRAINING_STEPS:-100}"
