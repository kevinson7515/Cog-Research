#!/bin/bash

#SBATCH --job-name=abl-step-e2e
#SBATCH -o logs/abl_step_e2e_%j.out
#SBATCH -e logs/abl_step_e2e_%j.out
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH -D /online1/ycsc_chenkh/hitici_07/Cog-Research

set -euo pipefail

DEFAULT_PROJECT_ROOT="/online1/ycsc_chenkh/hitici_07/Cog-Research"
if [[ -z "${PROJECT_ROOT:-}" || ! -f "${PROJECT_ROOT}/cosight_rl/scripts/run_cosight_grpo.sh" ]]; then
  export PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
fi

# Condition C: hidden teacher step rubric + the same end-to-end Plan-and-Act
# policy trajectory and frozen-visual-observation path used by Condition B.
export COSIGHT_EXPERIMENT="step_trajectory"
export COSIGHT_STEP_RUBRIC_JSONL="${PROJECT_ROOT}/data/quiz_train_step_rubric.jsonl"
export COSIGHT_DATA_DIR="${PROJECT_ROOT}/data/rl_ablation_step_trajectory"
export COSIGHT_OUTPUT_DIR="${PROJECT_ROOT}/outputs/qwen3-vl-8b-ablation-step-e2e-full"
export COSIGHT_EXP_NAME="ablation-step-only-e2e-grpo-full"
export COSIGHT_REBUILD_STEP_RUBRIC="${COSIGHT_REBUILD_STEP_RUBRIC:-0}"

source "${PROJECT_ROOT}/cosight_rl/scripts/trajectory_ablation_common.sh"
exec bash "${PROJECT_ROOT}/cosight_rl/scripts/run_cosight_grpo.sh" "$@"
