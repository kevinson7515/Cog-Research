#!/bin/bash

#SBATCH --job-name=abl-dim-e2e
#SBATCH -o logs/abl_dimension_e2e_%j.out
#SBATCH -e logs/abl_dimension_e2e_%j.out
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

# Condition B: current dimension-guided reward + end-to-end Plan-and-Act policy
# trajectory. On this cluster, the shared launcher exposes frozen visual
# observations through a tool instead of invoking vLLM 0.11's broken raw-pixel
# path. The submitted report is scored by exactly the same reward evaluator as
# Condition A; no extra process bonus is added.
export COSIGHT_EXPERIMENT="dimension_trajectory"
export COSIGHT_DIMENSION_RUBRIC_JSONL="${PROJECT_ROOT}/data/quiz_train_dimension_e2e_rubric.jsonl"
export COSIGHT_DATA_DIR="${PROJECT_ROOT}/data/rl_ablation_dimension_trajectory_trace_rubric"
export COSIGHT_OUTPUT_DIR="${PROJECT_ROOT}/outputs/qwen3-vl-8b-ablation-dimension-e2e-trace-rubric-full"
export COSIGHT_EXP_NAME="ablation-dimension-e2e-trace-rubric-grpo-full"
# Rebuild the hidden five-dimensional rubric from the restored legacy rubric
# plus planner/actor traces. The builder excludes all workspace reports and
# validates that task content/order is unchanged.
export COSIGHT_REBUILD_DIMENSION_RUBRIC="${COSIGHT_REBUILD_DIMENSION_RUBRIC:-1}"
# Do not repeat the previous 100-step no-op run if terminal reports are still
# unreachable. Initial validation must contain at least one non-zero reward.
export COSIGHT_ABORT_ZERO_REWARD_PREFLIGHT="${COSIGHT_ABORT_ZERO_REWARD_PREFLIGHT:-1}"

source "${PROJECT_ROOT}/cosight_rl/scripts/trajectory_ablation_common.sh"
exec bash "${PROJECT_ROOT}/cosight_rl/scripts/run_cosight_grpo.sh" "$@"
