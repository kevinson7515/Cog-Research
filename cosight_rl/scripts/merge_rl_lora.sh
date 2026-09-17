#!/bin/bash

#SBATCH --job-name=cosight-rl-merge
#SBATCH -o logs/rl_merge_%j.out
#SBATCH -e logs/rl_merge_%j.out
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-${0}}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/common_slurm_env.sh" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/cosight_rl/scripts/common_slurm_env.sh" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}/cosight_rl/scripts" && pwd)"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/common_slurm_env.sh" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
  elif [[ -n "${PROJECT_ROOT:-}" && -f "${PROJECT_ROOT}/cosight_rl/scripts/common_slurm_env.sh" ]]; then
    SCRIPT_DIR="$(cd "${PROJECT_ROOT}/cosight_rl/scripts" && pwd)"
  fi
fi
if [[ ! -f "${SCRIPT_DIR}/common_slurm_env.sh" ]]; then
  echo "Could not locate common_slurm_env.sh. Submit from the project root or set PROJECT_ROOT." >&2
  exit 1
fi
export COSIGHT_SCRIPT_DIR="${SCRIPT_DIR}"
source "${SCRIPT_DIR}/common_slurm_env.sh"

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

pick_default_base_model() {
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

BASE_MODEL=${BASE_MODEL:-"$(pick_default_base_model)"}
RL_CKPT_DIR=${RL_CKPT_DIR:-"${PROJECT_ROOT}/outputs/qwen3-vl-8b-rl-evidence-lora"}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/qwen3-vl-8b-rl-evidence-merged"}
RL_LORA_PATH=${RL_LORA_PATH:-""}

BASE_MODEL="$(repair_empty_project_root_path BASE_MODEL "${BASE_MODEL}")"
RL_CKPT_DIR="$(repair_empty_project_root_path RL_CKPT_DIR "${RL_CKPT_DIR}")"
OUTPUT_DIR="$(repair_empty_project_root_path OUTPUT_DIR "${OUTPUT_DIR}")"
RL_LORA_PATH="$(repair_empty_project_root_path RL_LORA_PATH "${RL_LORA_PATH}")"

if [[ ! -d "${BASE_MODEL}" || ! -f "${BASE_MODEL}/config.json" ]]; then
  echo "Base model is not a valid local HuggingFace model directory: ${BASE_MODEL}" >&2
  echo "Set BASE_MODEL to the merged/SFT model used for RL training." >&2
  exit 1
fi

if [[ -z "${RL_LORA_PATH}" ]]; then
  if [[ -f "${RL_CKPT_DIR}/adapter_config.json" ]]; then
    RL_LORA_PATH="${RL_CKPT_DIR}"
  elif [[ -f "${RL_CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    LATEST_STEP="$(tr -dc '0-9' < "${RL_CKPT_DIR}/latest_checkpointed_iteration.txt" || true)"
    if [[ -n "${LATEST_STEP}" && -f "${RL_CKPT_DIR}/global_step_${LATEST_STEP}/actor/lora_adapter/adapter_config.json" ]]; then
      RL_LORA_PATH="${RL_CKPT_DIR}/global_step_${LATEST_STEP}/actor/lora_adapter"
    fi
  fi
fi

if [[ -z "${RL_LORA_PATH}" ]]; then
  if [[ -f "${RL_CKPT_DIR}/adapter_config.json" ]]; then
    RL_LORA_PATH="${RL_CKPT_DIR}"
  else
    RL_LORA_PATH=$(find "${RL_CKPT_DIR}" -type f -name adapter_config.json -path "*/lora_adapter/*" -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)
  fi
fi

if [[ -z "${RL_LORA_PATH}" || ! -f "${RL_LORA_PATH}/adapter_config.json" ]]; then
  echo "Could not find a LoRA adapter. Set RL_LORA_PATH or check RL_CKPT_DIR=${RL_CKPT_DIR}" >&2
  exit 1
fi

echo "Merging RL LoRA:"
echo "  base_model=${BASE_MODEL}"
echo "  adapter_path=${RL_LORA_PATH}"
echo "  output_dir=${OUTPUT_DIR}"

run_in_runtime_env python scripts/merge_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${RL_LORA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --bf16 \
  --disable_bnb
