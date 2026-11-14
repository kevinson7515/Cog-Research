#!/usr/bin/env bash
#SBATCH --job-name=e6a
#SBATCH -o logs/e6a_%j.out
#SBATCH -e logs/e6a_%j.out
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --nodes=1

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export PROJECT_ROOT
source "${PROJECT_ROOT}/cosight_rl/scripts/common_slurm_env.sh"
if [[ "${1:-}" == resume ]]; then
  shift
  run_in_runtime_env python -m cosight_rl.trainer.e6a_resume "$@"
  exit
fi
run_in_runtime_env python -m cosight_rl.trainer.e6a "$@"
