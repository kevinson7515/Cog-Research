#!/bin/bash

#SBATCH --job-name=cosight-rl-grpo
#SBATCH -o logs/rl_grpo_%j.out
#SBATCH -e logs/rl_grpo_%j.out
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH -D /online1/ycsc_chenkh/hitici_07/Co-Sight-0

set -euo pipefail

DEFAULT_PROJECT_ROOT="/online1/ycsc_chenkh/hitici_07/Co-Sight-0"
if [[ -z "${PROJECT_ROOT:-}" || ! -f "${PROJECT_ROOT}/cosight_rl/scripts/common_slurm_env.sh" ]]; then
  export PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
fi
SCRIPT_DIR="${PROJECT_ROOT}/cosight_rl/scripts"

echo "run_cosight_ppo.sh is a compatibility wrapper; forwarding to run_cosight_grpo.sh."
exec bash "${SCRIPT_DIR}/run_cosight_grpo.sh" "$@"
