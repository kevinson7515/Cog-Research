#!/bin/bash

#SBATCH --job-name=cosight-rl-audit
#SBATCH -o logs/rl_audit_%j.out
#SBATCH -e logs/rl_audit_%j.out
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

DEFAULT_WORKSPACE="${PROJECT_ROOT}/work_space/work_space_20260621_075554"
if [[ ! -d "${DEFAULT_WORKSPACE}" ]]; then
  DEFAULT_WORKSPACE="$(find "${PROJECT_ROOT}/work_space" -maxdepth 1 -type d -name 'work_space_*' 2>/dev/null | sort | tail -n 1 || true)"
fi
WORKSPACE=${WORKSPACE:-"${DEFAULT_WORKSPACE}"}
LOGS=${LOGS:-"${PROJECT_ROOT}/logs"}
OUTPUT=${OUTPUT:-"${PROJECT_ROOT}/outputs/reward_audit_$(date +%Y%m%d_%H%M%S).jsonl"}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-"${OUTPUT}.summary.json"}
METADATA_JSONL=${METADATA_JSONL:-""}
MIN_REPORT_CHARS=${MIN_REPORT_CHARS:-200}

cmd=(
  python -m cosight_rl.rewards.audit_rewards
  --workspace "${WORKSPACE}"
  --logs "${LOGS}"
  --output "${OUTPUT}"
  --summary-output "${SUMMARY_OUTPUT}"
  --min-report-chars "${MIN_REPORT_CHARS}"
)

if [[ -n "${METADATA_JSONL}" ]]; then
  cmd+=(--metadata-jsonl "${METADATA_JSONL}")
fi

run_in_runtime_env "${cmd[@]}" "$@"
