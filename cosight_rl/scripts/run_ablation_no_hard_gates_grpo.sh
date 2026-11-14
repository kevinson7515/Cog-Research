#!/bin/bash

#SBATCH --job-name=rl-no-hard-gates
#SBATCH -o logs/rl_no_hard_gates_%j.out
#SBATCH -e logs/rl_no_hard_gates_%j.out
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH -D /online1/ycsc_chenkh/hitici_07/Cog-Research

set -euo pipefail

# Reuse the full baseline launcher: same SFT initialization, data, rollout,
# optimizer and update budget. Only the report failure-control bundle changes.
DEFAULT_PROJECT_ROOT="/online1/ycsc_chenkh/hitici_07/Cog-Research"
if [[ -z "${PROJECT_ROOT:-}" || ! -f "${PROJECT_ROOT}/cosight_rl/scripts/run_cosight_grpo.sh" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/cosight_rl/scripts/run_cosight_grpo.sh" ]]; then
    export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  else
    CANDIDATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    if [[ -f "${CANDIDATE_ROOT}/cosight_rl/scripts/run_cosight_grpo.sh" ]]; then
      export PROJECT_ROOT="${CANDIDATE_ROOT}"
    else
      export PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
    fi
  fi
fi

# Hard Gates here means BOTH additive penalties and report hard-error caps.
# Keep all five component scores, including H_report's repetition sensitivity.
export COSIGHT_ENABLE_HARD_GATES="0"
export COSIGHT_EXP_NAME="${COSIGHT_EXP_NAME:-qwen3-vl-8b-cosight-grpo-no-hard-gates}"
export COSIGHT_OUTPUT_DIR="${COSIGHT_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen3-vl-8b-rl-no-hard-gates-lora}"

exec bash "${PROJECT_ROOT}/cosight_rl/scripts/run_cosight_grpo.sh" "$@"
