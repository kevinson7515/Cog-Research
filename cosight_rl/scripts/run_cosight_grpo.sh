#!/bin/bash

#SBATCH --job-name=rl
#SBATCH -o logs/rl_grpo_%j.out
#SBATCH -e logs/rl_grpo_%j.out
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH -D /online1/ycsc_chenkh/hitici_07/Co-Sight-0

set -euo pipefail

DEFAULT_PROJECT_ROOT="/online1/ycsc_chenkh/hitici_07/Co-Sight-0"
if [[ -z "${PROJECT_ROOT:-}" || ! -f "${PROJECT_ROOT}/cosight_rl/scripts/common_slurm_env.sh" ]]; then
  export PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
fi
export COSIGHT_SCRIPT_DIR="${PROJECT_ROOT}/cosight_rl/scripts"

source "/online1/ycsc_chenkh/hitici_07/Co-Sight-0/cosight_rl/scripts/common_slurm_env.sh"

repair_empty_project_root_path() {
  local var_name="$1"
  local value="$2"
  case "${value}" in
    /outputs|/outputs/*)
      echo "WARNING: ${var_name}=${value} looks like PROJECT_ROOT was empty when the path was exported; rewriting to ${PROJECT_ROOT}${value}." >&2
      echo "${PROJECT_ROOT}${value}"
      ;;
    *)
      echo "${value}"
      ;;
  esac
}

GPU_COUNT=${GPU_COUNT:-$(detect_gpu_count)}
export COSIGHT_ADV_ESTIMATOR="${COSIGHT_ADV_ESTIMATOR:-grpo}"
export COSIGHT_USE_TRANSFER_QUEUE="${COSIGHT_USE_TRANSFER_QUEUE:-0}"
export COSIGHT_FSDP_MODEL_DTYPE="${COSIGHT_FSDP_MODEL_DTYPE:-bf16}"
export COSIGHT_REF_FSDP_MODEL_DTYPE="${COSIGHT_REF_FSDP_MODEL_DTYPE:-${COSIGHT_FSDP_MODEL_DTYPE}}"
export COSIGHT_GPU_MEMORY_UTILIZATION="${COSIGHT_GPU_MEMORY_UTILIZATION:-0.45}"
export COSIGHT_ENABLE_PREFIX_CACHING="${COSIGHT_ENABLE_PREFIX_CACHING:-0}"
export COSIGHT_DISABLE_VLLM_LORA="${COSIGHT_DISABLE_VLLM_LORA:-1}"
export COSIGHT_ROLLOUT_LOAD_FORMAT="${COSIGHT_ROLLOUT_LOAD_FORMAT:-auto}"
export COSIGHT_ROLLOUT_LAYERED_SUMMON="${COSIGHT_ROLLOUT_LAYERED_SUMMON:-auto}"
export COSIGHT_ROLLOUT_ENFORCE_EAGER="${COSIGHT_ROLLOUT_ENFORCE_EAGER:-1}"
export COSIGHT_ROLLOUT_MAX_NUM_SEQS="${COSIGHT_ROLLOUT_MAX_NUM_SEQS:-8}"
export COSIGHT_PPO_MICRO_BATCH_SIZE_PER_GPU="${COSIGHT_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export COSIGHT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${COSIGHT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export COSIGHT_DISABLE_TORCH_COMPILE="${COSIGHT_DISABLE_TORCH_COMPILE:-1}"
export COSIGHT_DISABLE_MM_PREPROCESSOR_CACHE="${COSIGHT_DISABLE_MM_PREPROCESSOR_CACHE:-1}"
export COSIGHT_RAY_NUM_CPUS="${COSIGHT_RAY_NUM_CPUS:-8}"
export COSIGHT_DATALOADER_NUM_WORKERS="${COSIGHT_DATALOADER_NUM_WORKERS:-0}"
export COSIGHT_VAL_RATIO="${COSIGHT_VAL_RATIO:-0.1}"
export COSIGHT_TEST_FREQ="${COSIGHT_TEST_FREQ:-20}"
export COSIGHT_SAVE_FREQ="${COSIGHT_SAVE_FREQ:-10}"
export COSIGHT_LOG_PROGRESS="${COSIGHT_LOG_PROGRESS:-1}"
export COSIGHT_RESUME_MODE="${COSIGHT_RESUME_MODE:-disable}"
export COSIGHT_TOTAL_TRAINING_STEPS="${COSIGHT_TOTAL_TRAINING_STEPS:-100}"
export COSIGHT_TOTAL_EPOCHS="${COSIGHT_TOTAL_EPOCHS:-${COSIGHT_TOTAL_TRAINING_STEPS}}"
export COSIGHT_MAX_TRAIN_SAMPLES="${COSIGHT_MAX_TRAIN_SAMPLES:-100}"
export COSIGHT_MAX_VAL_SAMPLES="${COSIGHT_MAX_VAL_SAMPLES:-20}"
export COSIGHT_BUILD_EVIDENCE_CORPUS="${COSIGHT_BUILD_EVIDENCE_CORPUS:-1}"
export COSIGHT_REBUILD_EVIDENCE_CORPUS="${COSIGHT_REBUILD_EVIDENCE_CORPUS:-0}"
export COSIGHT_INCLUDE_EVIDENCE_IN_PROMPT="${COSIGHT_INCLUDE_EVIDENCE_IN_PROMPT:-1}"
export COSIGHT_INCLUDE_TRACE_EVIDENCE="${COSIGHT_INCLUDE_TRACE_EVIDENCE:-1}"
export COSIGHT_INCLUDE_FINAL_REPORT_EVIDENCE="${COSIGHT_INCLUDE_FINAL_REPORT_EVIDENCE:-0}"
export COSIGHT_EVIDENCE_MAX_TASKS="${COSIGHT_EVIDENCE_MAX_TASKS:-0}"
export COSIGHT_MAX_EVIDENCE_ITEMS="${COSIGHT_MAX_EVIDENCE_ITEMS:-6}"
export COSIGHT_MAX_EVIDENCE_CHARS="${COSIGHT_MAX_EVIDENCE_CHARS:-4000}"
export COSIGHT_MIN_EVIDENCE_COVERAGE="${COSIGHT_MIN_EVIDENCE_COVERAGE:-0.75}"
export COSIGHT_MAX_EVIDENCE_DOCS_PER_TASK="${COSIGHT_MAX_EVIDENCE_DOCS_PER_TASK:-10}"
export COSIGHT_MAX_EVIDENCE_DOC_CHARS="${COSIGHT_MAX_EVIDENCE_DOC_CHARS:-2400}"
export COSIGHT_MAX_TRACE_EVIDENCE_PER_TASK="${COSIGHT_MAX_TRACE_EVIDENCE_PER_TASK:-4}"

warn_legacy_env() {
  local legacy_name="$1"
  local namespaced_name="$2"
  if [[ -n "${!legacy_name:-}" && -z "${!namespaced_name:-}" ]]; then
    echo "Ignoring inherited ${legacy_name}=${!legacy_name}; use ${namespaced_name} to override this script." >&2
  fi
}

pick_default_model_path() {
  local candidates=(
    "${PROJECT_ROOT}/outputs/qwen3-vl-8b-cosight-merged"
    "${PROJECT_ROOT}/outputs/qwen3-vl-8b-cosight-merged-vision"
    "${PROJECT_ROOT}/outputs/qwen3-vl-8b-cosight-stagewise-merged"
    "${PROJECT_ROOT}/outputs/qwen3-vl-8b-sft"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" && -f "${candidate}/config.json" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "${candidates[0]}"
}

DEFAULT_MODEL_PATH="$(pick_default_model_path)"
warn_legacy_env MODEL_PATH COSIGHT_MODEL_PATH
warn_legacy_env TRAIN_JSONL COSIGHT_TRAIN_JSONL
warn_legacy_env VAL_JSONL COSIGHT_VAL_JSONL
warn_legacy_env IMAGE_BASE_DIR COSIGHT_IMAGE_BASE_DIR
warn_legacy_env DATA_DIR COSIGHT_DATA_DIR
warn_legacy_env OUTPUT_DIR COSIGHT_OUTPUT_DIR
MODEL_PATH=${COSIGHT_MODEL_PATH:-"${DEFAULT_MODEL_PATH}"}
TRAIN_JSONL=${COSIGHT_TRAIN_JSONL:-"${PROJECT_ROOT}/data/quiz_train_rubric.jsonl"}
VAL_JSONL=${COSIGHT_VAL_JSONL:-""}
IMAGE_BASE_DIR=${COSIGHT_IMAGE_BASE_DIR:-"${PROJECT_ROOT}/data/images"}
DATA_DIR=${COSIGHT_DATA_DIR:-"${PROJECT_ROOT}/data/rl_cosight_rubric"}
OUTPUT_DIR=${COSIGHT_OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/qwen3-vl-8b-rl-evidence-lora"}
WORKSPACE_DIR=${COSIGHT_WORKSPACE_DIR:-"${PROJECT_ROOT}/work_space/work_space_20260612_134333"}
TRACE_DIR=${COSIGHT_TRACE_DIR:-"${PROJECT_ROOT}/trace"}
EVIDENCE_DIR=${COSIGHT_EVIDENCE_DIR:-"${PROJECT_ROOT}/data/rl_evidence_corpus"}
EVIDENCE_CORPUS=${COSIGHT_EVIDENCE_CORPUS:-"${EVIDENCE_DIR}/task_evidence.json"}

MODEL_PATH="$(repair_empty_project_root_path MODEL_PATH "${MODEL_PATH}")"
OUTPUT_DIR="$(repair_empty_project_root_path OUTPUT_DIR "${OUTPUT_DIR}")"

if [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* || "${MODEL_PATH}" == ../* ]]; then
  if [[ ! -d "${MODEL_PATH}" || ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: COSIGHT_MODEL_PATH is not a valid local HuggingFace model directory: ${MODEL_PATH}" >&2
    echo "Expected to find ${MODEL_PATH}/config.json." >&2
    echo "Set COSIGHT_MODEL_PATH to an existing merged/SFT model directory, for example:" >&2
    echo "  sbatch --export=ALL,COSIGHT_MODEL_PATH=${PROJECT_ROOT}/outputs/qwen3-vl-8b-cosight-merged cosight_rl/scripts/run_cosight_grpo.sh" >&2
    exit 2
  fi
fi

if [[ "${GPU_COUNT}" -lt 3 ]]; then
  echo "Warning: GPU_COUNT=${GPU_COUNT}. Qwen3-VL-8B GRPO on A100 usually needs 3 GPUs; 2-GPU runs may die during FSDP/LoRA initialization." >&2
fi

echo "CoSight GRPO runtime: GPU_COUNT=${GPU_COUNT} MODEL_PATH=${MODEL_PATH} TRAIN_JSONL=${TRAIN_JSONL} VAL_JSONL=${VAL_JSONL:-<train_split>} DATA_DIR=${DATA_DIR} OUTPUT_DIR=${OUTPUT_DIR} COSIGHT_VAL_RATIO=${COSIGHT_VAL_RATIO} COSIGHT_TOTAL_TRAINING_STEPS=${COSIGHT_TOTAL_TRAINING_STEPS} COSIGHT_TOTAL_EPOCHS=${COSIGHT_TOTAL_EPOCHS} COSIGHT_RESUME_MODE=${COSIGHT_RESUME_MODE} COSIGHT_MAX_TRAIN_SAMPLES=${COSIGHT_MAX_TRAIN_SAMPLES} COSIGHT_MAX_VAL_SAMPLES=${COSIGHT_MAX_VAL_SAMPLES} COSIGHT_TEST_FREQ=${COSIGHT_TEST_FREQ} COSIGHT_SAVE_FREQ=${COSIGHT_SAVE_FREQ} COSIGHT_LOG_PROGRESS=${COSIGHT_LOG_PROGRESS} COSIGHT_BUILD_EVIDENCE_CORPUS=${COSIGHT_BUILD_EVIDENCE_CORPUS} COSIGHT_REBUILD_EVIDENCE_CORPUS=${COSIGHT_REBUILD_EVIDENCE_CORPUS} EVIDENCE_CORPUS=${EVIDENCE_CORPUS} COSIGHT_INCLUDE_EVIDENCE_IN_PROMPT=${COSIGHT_INCLUDE_EVIDENCE_IN_PROMPT} COSIGHT_MAX_EVIDENCE_ITEMS=${COSIGHT_MAX_EVIDENCE_ITEMS} COSIGHT_MAX_EVIDENCE_CHARS=${COSIGHT_MAX_EVIDENCE_CHARS} COSIGHT_MIN_EVIDENCE_COVERAGE=${COSIGHT_MIN_EVIDENCE_COVERAGE} COSIGHT_INCLUDE_TRACE_EVIDENCE=${COSIGHT_INCLUDE_TRACE_EVIDENCE} COSIGHT_USE_TRANSFER_QUEUE=${COSIGHT_USE_TRANSFER_QUEUE} COSIGHT_FSDP_MODEL_DTYPE=${COSIGHT_FSDP_MODEL_DTYPE} COSIGHT_REF_FSDP_MODEL_DTYPE=${COSIGHT_REF_FSDP_MODEL_DTYPE} COSIGHT_GPU_MEMORY_UTILIZATION=${COSIGHT_GPU_MEMORY_UTILIZATION} COSIGHT_ROLLOUT_TP=${COSIGHT_ROLLOUT_TP:-1} COSIGHT_ROLLOUT_N=${COSIGHT_ROLLOUT_N:-2} COSIGHT_ROLLOUT_MAX_NUM_SEQS=${COSIGHT_ROLLOUT_MAX_NUM_SEQS} COSIGHT_ROLLOUT_ENFORCE_EAGER=${COSIGHT_ROLLOUT_ENFORCE_EAGER} COSIGHT_PPO_MICRO_BATCH_SIZE_PER_GPU=${COSIGHT_PPO_MICRO_BATCH_SIZE_PER_GPU} COSIGHT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${COSIGHT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} COSIGHT_DISABLE_TORCH_COMPILE=${COSIGHT_DISABLE_TORCH_COMPILE} COSIGHT_DISABLE_VLLM_LORA=${COSIGHT_DISABLE_VLLM_LORA} COSIGHT_ENABLE_PREFIX_CACHING=${COSIGHT_ENABLE_PREFIX_CACHING} COSIGHT_DISABLE_MM_PREPROCESSOR_CACHE=${COSIGHT_DISABLE_MM_PREPROCESSOR_CACHE} COSIGHT_ROLLOUT_LOAD_FORMAT=${COSIGHT_ROLLOUT_LOAD_FORMAT} COSIGHT_ROLLOUT_LAYERED_SUMMON=${COSIGHT_ROLLOUT_LAYERED_SUMMON} COSIGHT_RAY_NUM_CPUS=${COSIGHT_RAY_NUM_CPUS} COSIGHT_DATALOADER_NUM_WORKERS=${COSIGHT_DATALOADER_NUM_WORKERS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} OMP_NUM_THREADS=${OMP_NUM_THREADS} ulimit_nproc=$(ulimit -u)"
nvidia-smi || true

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  run_in_runtime_env python -m cosight_rl.trainer.run_grpo \
    --project-root "${PROJECT_ROOT}" \
    --dry-run-env \
    --train-jsonl "${TRAIN_JSONL}" \
    --image-base-dir "${IMAGE_BASE_DIR}" \
    --max-train-samples "${COSIGHT_DRY_RUN_SAMPLES:-2}" \
    "$@"
  exit 0
fi

if [[ "${1:-}" == "--dry-run-trainer" ]]; then
  shift
  run_in_runtime_env python -m cosight_rl.trainer.run_grpo \
    --project-root "${PROJECT_ROOT}" \
    --run \
    --dry-run-trainer \
    --gpus "${GPU_COUNT}" \
    --model-path "${MODEL_PATH}" \
    --train-jsonl "${TRAIN_JSONL}" \
    --val-jsonl "${VAL_JSONL}" \
    --val-ratio "${COSIGHT_VAL_RATIO}" \
    --image-base-dir "${IMAGE_BASE_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --resume-mode "${COSIGHT_RESUME_MODE}" \
    --workspace-dir "${WORKSPACE_DIR}" \
    --trace-dir "${TRACE_DIR}" \
    --evidence-output-dir "${EVIDENCE_DIR}" \
    --evidence-corpus "${EVIDENCE_CORPUS}" \
    --evidence-max-tasks "${COSIGHT_EVIDENCE_MAX_TASKS}" \
    --max-evidence-items "${COSIGHT_MAX_EVIDENCE_ITEMS}" \
    --max-evidence-chars "${COSIGHT_MAX_EVIDENCE_CHARS}" \
    --min-evidence-coverage "${COSIGHT_MIN_EVIDENCE_COVERAGE}" \
    --max-evidence-docs-per-task "${COSIGHT_MAX_EVIDENCE_DOCS_PER_TASK}" \
    --max-evidence-doc-chars "${COSIGHT_MAX_EVIDENCE_DOC_CHARS}" \
    --max-trace-evidence-per-task "${COSIGHT_MAX_TRACE_EVIDENCE_PER_TASK}" \
    --save-freq "${COSIGHT_SAVE_FREQ}" \
    --test-freq "${COSIGHT_TEST_FREQ}" \
    --max-train-samples "${COSIGHT_DRY_RUN_SAMPLES:-2}" \
    --max-val-samples "${COSIGHT_DRY_RUN_SAMPLES:-2}" \
    "$@"
  exit 0
fi

run_in_runtime_env python -m cosight_rl.trainer.run_grpo \
  --project-root "${PROJECT_ROOT}" \
  --run \
  --gpus "${GPU_COUNT}" \
  --model-path "${MODEL_PATH}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --val-ratio "${COSIGHT_VAL_RATIO}" \
  --image-base-dir "${IMAGE_BASE_DIR}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume-mode "${COSIGHT_RESUME_MODE}" \
  --workspace-dir "${WORKSPACE_DIR}" \
  --trace-dir "${TRACE_DIR}" \
  --evidence-output-dir "${EVIDENCE_DIR}" \
  --evidence-corpus "${EVIDENCE_CORPUS}" \
  --evidence-max-tasks "${COSIGHT_EVIDENCE_MAX_TASKS}" \
  --max-evidence-items "${COSIGHT_MAX_EVIDENCE_ITEMS}" \
  --max-evidence-chars "${COSIGHT_MAX_EVIDENCE_CHARS}" \
  --min-evidence-coverage "${COSIGHT_MIN_EVIDENCE_COVERAGE}" \
  --max-evidence-docs-per-task "${COSIGHT_MAX_EVIDENCE_DOCS_PER_TASK}" \
  --max-evidence-doc-chars "${COSIGHT_MAX_EVIDENCE_DOC_CHARS}" \
  --max-trace-evidence-per-task "${COSIGHT_MAX_TRACE_EVIDENCE_PER_TASK}" \
  --max-train-samples "${COSIGHT_MAX_TRAIN_SAMPLES}" \
  --max-val-samples "${COSIGHT_MAX_VAL_SAMPLES}" \
  --save-freq "${COSIGHT_SAVE_FREQ}" \
  --test-freq "${COSIGHT_TEST_FREQ}" \
  "$@"
