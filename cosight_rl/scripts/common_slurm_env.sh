#!/usr/bin/env bash

# Shared Slurm/Singularity runtime for Co-Sight RL scripts.
# Source this file after `set -euo pipefail`.

if command -v module >/dev/null 2>&1; then
  module load amd/cmake/3.31.8 || true
  module load amd/gcc_compiler/13.2.0 || true
  module load amd/singularity/3.10.4 || true
fi

if [[ -n "${COSIGHT_SCRIPT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd "${COSIGHT_SCRIPT_DIR}" && pwd)"
else
  SCRIPT_PATH="${BASH_SOURCE[0]:-${0}}"
  SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
fi
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_PROJECT_ROOT="${LOCAL_PROJECT_ROOT}"

if [[ -n "${PROJECT_ROOT:-}" && ! -f "${PROJECT_ROOT}/cosight_rl/scripts/common_slurm_env.sh" ]]; then
  echo "Ignoring invalid PROJECT_ROOT=${PROJECT_ROOT}; expected a Co-Sight checkout." >&2
  unset PROJECT_ROOT
fi

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -d "${DEFAULT_PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
  else
    PROJECT_ROOT="${LOCAL_PROJECT_ROOT}"
  fi
fi
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SIF_IMAGE="${SIF_IMAGE:-/online1/hpc_containers/base_hpc_glibc_235.sif}"
CONDA_BIND="${CONDA_BIND:-/online1/hpc_containers/conda/}"
PROJECT_BIND="${PROJECT_BIND:-/online1/ycsc_chenkh/hitici_07}"
CONDA_ENV="${CONDA_ENV:-/home/export/base/ycsc_chenkh/hitici_07/online1/anaconda3/envs/co-sight}"

HTTP_PROXY_VALUE="${HTTP_PROXY_VALUE:-http://174.0.250.13:3128}"
HTTPS_PROXY_VALUE="${HTTPS_PROXY_VALUE:-http://174.0.250.13:3128}"
NO_PROXY_VALUE="${NO_PROXY_VALUE:-localhost,127.0.0.1,0.0.0.0}"

export http_proxy="${HTTP_PROXY_VALUE}"
export https_proxy="${HTTPS_PROXY_VALUE}"
export HTTP_PROXY="${HTTP_PROXY_VALUE}"
export HTTPS_PROXY="${HTTPS_PROXY_VALUE}"
export no_proxy="${NO_PROXY_VALUE}"
export NO_PROXY="${NO_PROXY_VALUE}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ "${COSIGHT_KEEP_PYTORCH_CUDA_ALLOC_CONF:-0}" != "1" && "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
if [[ -z "${NCCL_RAS_ADDR:-}" && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
  export NCCL_RAS_ADDR="localhost:$((20000 + (SLURM_JOB_ID % 20000)))"
fi
if [[ -z "${NCCL_SOCKET_MAGIC:-}" && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
  export NCCL_SOCKET_MAGIC="${SLURM_JOB_ID}"
fi
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

select_writable_runtime_dir() {
  local suffix="$1"
  local candidate
  for candidate in \
    "${COSIGHT_RUNTIME_ROOT:-}" \
    "/tmp/cosight_${USER:-u}_${SLURM_JOB_ID:-manual}" \
    "/dev/shm/cosight_${USER:-u}_${SLURM_JOB_ID:-manual}" \
    "${PROJECT_ROOT}/outputs/${suffix}"; do
    if [[ -n "${candidate}" ]] && mkdir -p "${candidate}" 2>/dev/null && [[ -w "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

export COSIGHT_RAY_TMPDIR="${COSIGHT_RAY_TMPDIR:-$(select_writable_runtime_dir ray_tmp)}"
export RAY_TMPDIR="${RAY_TMPDIR:-${COSIGHT_RAY_TMPDIR}}"
export TMPDIR="${COSIGHT_TMPDIR:-$(select_writable_runtime_dir tmp)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export COSIGHT_CC="${COSIGHT_CC:-/usr/bin/gcc}"
export COSIGHT_CXX="${COSIGHT_CXX:-/usr/bin/g++}"
export COSIGHT_CUDAHOSTCXX="${COSIGHT_CUDAHOSTCXX:-${COSIGHT_CXX}}"
export CC="${COSIGHT_CC}"
export CXX="${COSIGHT_CXX}"
export CUDAHOSTCXX="${COSIGHT_CUDAHOSTCXX}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_ROOT}/outputs/.triton_cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PROJECT_ROOT}/outputs/.torch_extensions}"

# These cluster jobs target NVIDIA CUDA GPUs. Some shared images/modules may leak
# ROCm visibility variables; verl refuses to run when ROCR/HIP and CUDA
# visibility variables are both set.
if [[ "${COSIGHT_KEEP_ROCM_VISIBLE_DEVICES:-0}" != "1" ]]; then
  unset ROCR_VISIBLE_DEVICES
  unset HIP_VISIBLE_DEVICES
fi

mkdir -p logs outputs "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${COSIGHT_RAY_TMPDIR}" "${TMPDIR}"

detect_gpu_count() {
  if [[ -n "${GPU_COUNT:-}" ]]; then
    echo "${GPU_COUNT}"
  elif [[ "${SLURM_GPUS_PER_NODE:-}" =~ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
    echo "${SLURM_GPUS_ON_NODE}"
  elif [[ "${SLURM_GPUS_ON_NODE:-}" =~ ([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local devices="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -z "${devices}" ]]; then
      echo "1"
    else
      awk -F',' '{print NF}' <<< "${devices}"
    fi
  else
    echo "${NGPUS_PER_NODE:-1}"
  fi
}

use_singularity_runtime() {
  case "${COSIGHT_USE_SINGULARITY:-auto}" in
    1|true|TRUE|yes|YES) return 0 ;;
    0|false|FALSE|no|NO) return 1 ;;
  esac
  command -v singularity >/dev/null 2>&1 && [[ -f "${SIF_IMAGE}" ]]
}

run_in_runtime_env() {
  local quoted_cmd
  local quoted_project_root
  local quoted_conda_env
  printf -v quoted_cmd "%q " "$@"
  printf -v quoted_project_root "%q" "${PROJECT_ROOT}"
  printf -v quoted_conda_env "%q" "${CONDA_ENV}"

  if use_singularity_runtime; then
    singularity exec --nv \
      -B "${CONDA_BIND}" \
      -B "${PROJECT_BIND}" \
      "${SIF_IMAGE}" \
      bash -lc "
        source activate ${quoted_conda_env}
        cd ${quoted_project_root}
        unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
        export http_proxy='${HTTP_PROXY_VALUE}'
        export https_proxy='${HTTPS_PROXY_VALUE}'
        export HTTP_PROXY='${HTTP_PROXY_VALUE}'
        export HTTPS_PROXY='${HTTPS_PROXY_VALUE}'
        export no_proxy='${NO_PROXY_VALUE}'
        export NO_PROXY='${NO_PROXY_VALUE}'
        export TOKENIZERS_PARALLELISM='${TOKENIZERS_PARALLELISM}'
        export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN='${VLLM_ALLOW_LONG_MAX_MODEL_LEN}'
        export VLLM_ALLOW_RUNTIME_LORA_UPDATING='${VLLM_ALLOW_RUNTIME_LORA_UPDATING}'
        export VLLM_LOGGING_LEVEL='${VLLM_LOGGING_LEVEL}'
        export NCCL_DEBUG='${NCCL_DEBUG}'
        export CUDA_DEVICE_MAX_CONNECTIONS='${CUDA_DEVICE_MAX_CONNECTIONS}'
        export NCCL_CUMEM_ENABLE='${NCCL_CUMEM_ENABLE}'
        export NCCL_RAS_ENABLE='${NCCL_RAS_ENABLE}'
        if [[ -n '${NCCL_RAS_ADDR:-}' ]]; then
          export NCCL_RAS_ADDR='${NCCL_RAS_ADDR:-}'
        else
          unset NCCL_RAS_ADDR
        fi
        export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO='${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO}'
        if [[ -n '${NCCL_SOCKET_MAGIC:-}' ]]; then
          export NCCL_SOCKET_MAGIC='${NCCL_SOCKET_MAGIC:-}'
        else
          unset NCCL_SOCKET_MAGIC
        fi
        export PYTHONUNBUFFERED='${PYTHONUNBUFFERED}'
        export PYTHONPATH='${PROJECT_ROOT}'\${PYTHONPATH:+:\${PYTHONPATH}}
        export COSIGHT_RAY_TMPDIR='${COSIGHT_RAY_TMPDIR}'
        export RAY_TMPDIR='${RAY_TMPDIR}'
        export TMPDIR='${TMPDIR}'
        export HYDRA_FULL_ERROR='${HYDRA_FULL_ERROR}'
        export OMP_NUM_THREADS='${OMP_NUM_THREADS}'
        export MKL_NUM_THREADS='${MKL_NUM_THREADS}'
        export OPENBLAS_NUM_THREADS='${OPENBLAS_NUM_THREADS}'
        export NUMEXPR_NUM_THREADS='${NUMEXPR_NUM_THREADS}'
        export BLIS_NUM_THREADS='${BLIS_NUM_THREADS}'
        export VECLIB_MAXIMUM_THREADS='${VECLIB_MAXIMUM_THREADS}'
        export RAYON_NUM_THREADS='${RAYON_NUM_THREADS}'
        export COSIGHT_CC='${COSIGHT_CC}'
        export COSIGHT_CXX='${COSIGHT_CXX}'
        export COSIGHT_CUDAHOSTCXX='${COSIGHT_CUDAHOSTCXX}'
        export CC='${COSIGHT_CC}'
        export CXX='${COSIGHT_CXX}'
        export CUDAHOSTCXX='${COSIGHT_CUDAHOSTCXX}'
        export TRITON_CACHE_DIR='${TRITON_CACHE_DIR}'
        export TORCH_EXTENSIONS_DIR='${TORCH_EXTENSIONS_DIR}'
        pass_cosight_env() {
          local name=\"\$1\"
          local value=\"\$2\"
          if [[ -n \"\${value}\" ]]; then
            export \"\${name}=\${value}\"
          else
            unset \"\${name}\"
          fi
        }
        pass_cosight_env COSIGHT_ADV_ESTIMATOR '${COSIGHT_ADV_ESTIMATOR:-}'
        pass_cosight_env COSIGHT_USE_TRANSFER_QUEUE '${COSIGHT_USE_TRANSFER_QUEUE:-}'
        pass_cosight_env COSIGHT_MODEL_PATH '${COSIGHT_MODEL_PATH:-}'
        pass_cosight_env COSIGHT_TRAIN_JSONL '${COSIGHT_TRAIN_JSONL:-}'
        pass_cosight_env COSIGHT_VAL_JSONL '${COSIGHT_VAL_JSONL:-}'
        pass_cosight_env COSIGHT_IMAGE_BASE_DIR '${COSIGHT_IMAGE_BASE_DIR:-}'
        pass_cosight_env COSIGHT_DATA_DIR '${COSIGHT_DATA_DIR:-}'
        pass_cosight_env COSIGHT_TRAIN_BATCH_SIZE '${COSIGHT_TRAIN_BATCH_SIZE:-}'
        pass_cosight_env COSIGHT_PPO_MINI_BATCH_SIZE '${COSIGHT_PPO_MINI_BATCH_SIZE:-}'
        pass_cosight_env COSIGHT_PPO_MICRO_BATCH_SIZE_PER_GPU '${COSIGHT_PPO_MICRO_BATCH_SIZE_PER_GPU:-}'
        pass_cosight_env COSIGHT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU '${COSIGHT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-}'
        pass_cosight_env COSIGHT_ROLLOUT_TP '${COSIGHT_ROLLOUT_TP:-}'
        pass_cosight_env COSIGHT_ROLLOUT_N '${COSIGHT_ROLLOUT_N:-}'
        pass_cosight_env COSIGHT_ROLLOUT_MAX_NUM_SEQS '${COSIGHT_ROLLOUT_MAX_NUM_SEQS:-}'
        pass_cosight_env COSIGHT_DATALOADER_NUM_WORKERS '${COSIGHT_DATALOADER_NUM_WORKERS:-}'
        pass_cosight_env COSIGHT_FSDP_MODEL_DTYPE '${COSIGHT_FSDP_MODEL_DTYPE:-}'
        pass_cosight_env COSIGHT_REF_FSDP_MODEL_DTYPE '${COSIGHT_REF_FSDP_MODEL_DTYPE:-}'
        pass_cosight_env COSIGHT_DISABLE_TORCH_COMPILE '${COSIGHT_DISABLE_TORCH_COMPILE:-}'
        pass_cosight_env COSIGHT_KL_LOSS_COEF '${COSIGHT_KL_LOSS_COEF:-}'
        pass_cosight_env COSIGHT_GPU_MEMORY_UTILIZATION '${COSIGHT_GPU_MEMORY_UTILIZATION:-}'
        pass_cosight_env COSIGHT_DRY_RUN_GPU_MEMORY_UTILIZATION '${COSIGHT_DRY_RUN_GPU_MEMORY_UTILIZATION:-}'
        pass_cosight_env COSIGHT_DISABLE_MM_PREPROCESSOR_CACHE '${COSIGHT_DISABLE_MM_PREPROCESSOR_CACHE:-}'
        pass_cosight_env COSIGHT_ENABLE_CHUNKED_PREFILL '${COSIGHT_ENABLE_CHUNKED_PREFILL:-}'
        pass_cosight_env COSIGHT_DISABLE_VLLM_LORA '${COSIGHT_DISABLE_VLLM_LORA:-}'
        pass_cosight_env COSIGHT_VLLM_SKIP_MM_PROFILING '${COSIGHT_VLLM_SKIP_MM_PROFILING:-}'
        pass_cosight_env COSIGHT_ROLLOUT_ENFORCE_EAGER '${COSIGHT_ROLLOUT_ENFORCE_EAGER:-}'
        pass_cosight_env COSIGHT_DRY_RUN_MAX_NUM_SEQS '${COSIGHT_DRY_RUN_MAX_NUM_SEQS:-}'
        pass_cosight_env COSIGHT_DRY_RUN_MAX_PROMPT_LENGTH '${COSIGHT_DRY_RUN_MAX_PROMPT_LENGTH:-}'
        pass_cosight_env COSIGHT_DRY_RUN_MAX_RESPONSE_LENGTH '${COSIGHT_DRY_RUN_MAX_RESPONSE_LENGTH:-}'
        pass_cosight_env COSIGHT_DRY_RUN_ENFORCE_EAGER '${COSIGHT_DRY_RUN_ENFORCE_EAGER:-}'
        pass_cosight_env COSIGHT_DRY_RUN_SKIP_MM_PROFILING '${COSIGHT_DRY_RUN_SKIP_MM_PROFILING:-}'
        pass_cosight_env COSIGHT_DRY_RUN_DISABLE_VLLM_LORA '${COSIGHT_DRY_RUN_DISABLE_VLLM_LORA:-}'
        pass_cosight_env COSIGHT_ENABLE_PREFIX_CACHING '${COSIGHT_ENABLE_PREFIX_CACHING:-}'
        pass_cosight_env COSIGHT_ROLLOUT_LOAD_FORMAT '${COSIGHT_ROLLOUT_LOAD_FORMAT:-}'
        pass_cosight_env COSIGHT_ROLLOUT_LAYERED_SUMMON '${COSIGHT_ROLLOUT_LAYERED_SUMMON:-}'
        pass_cosight_env COSIGHT_RAY_NUM_CPUS '${COSIGHT_RAY_NUM_CPUS:-}'
        pass_cosight_env COSIGHT_MAX_PROMPT_LENGTH '${COSIGHT_MAX_PROMPT_LENGTH:-}'
        pass_cosight_env COSIGHT_MAX_RESPONSE_LENGTH '${COSIGHT_MAX_RESPONSE_LENGTH:-}'
        pass_cosight_env COSIGHT_TOTAL_TRAINING_STEPS '${COSIGHT_TOTAL_TRAINING_STEPS:-}'
        pass_cosight_env COSIGHT_TOTAL_EPOCHS '${COSIGHT_TOTAL_EPOCHS:-}'
        pass_cosight_env COSIGHT_LORA_RANK '${COSIGHT_LORA_RANK:-}'
        pass_cosight_env COSIGHT_LORA_ALPHA '${COSIGHT_LORA_ALPHA:-}'
        pass_cosight_env COSIGHT_LORA_TARGET_MODULES '${COSIGHT_LORA_TARGET_MODULES:-}'
        pass_cosight_env COSIGHT_LR '${COSIGHT_LR:-}'
        pass_cosight_env COSIGHT_EXP_NAME '${COSIGHT_EXP_NAME:-}'
        pass_cosight_env COSIGHT_OUTPUT_DIR '${COSIGHT_OUTPUT_DIR:-}'
        pass_cosight_env COSIGHT_DATA_TRUNCATION '${COSIGHT_DATA_TRUNCATION:-}'
        if [[ '${COSIGHT_KEEP_ROCM_VISIBLE_DEVICES:-0}' != '1' ]]; then
          unset ROCR_VISIBLE_DEVICES
          unset HIP_VISIBLE_DEVICES
        fi
        ${quoted_cmd}
      "
  else
    "$@"
  fi
}
