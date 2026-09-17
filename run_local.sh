#!/bin/bash

#SBATCH --job-name=cog-research-mmdr
#SBATCH --output=/home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/mmdeepresearch_bench_1_%j.out
#SBATCH --error=/home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/mmdeepresearch_bench_1_%j.out
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

set -euo pipefail

# -----------------------------------------------------------------------------
# 1. Module environment
# -----------------------------------------------------------------------------
# vLLM is only being executed, not compiled. Avoid loading CMake/GCC modules,
# because they may inject host libraries into LD_LIBRARY_PATH and interfere with
# the CUDA libraries mounted into Singularity by --nv.
module purge
module load amd/singularity/3.10.4

# -----------------------------------------------------------------------------
# 2. Paths and service configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT="/home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research"
SIF_IMAGE="/online1/hpc_containers/base_hpc_glibc_235.sif"
CONDA_BIND="/online1/hpc_containers/conda/"
PROJECT_BIND="/online1/ycsc_chenkh/hitici_07"
CONDA_ENV="/home/export/base/ycsc_chenkh/hitici_07/online1/anaconda3/envs/cog-research"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/outputs/qwen3-vl-8b-ablation-without-sft-merged}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b-ablation-without-sft-merged}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
REQUESTED_VLLM_PORT="${VLLM_PORT:-8011}"
VLLM_PORT="${REQUESTED_VLLM_PORT}"
VLLM_PORT_AUTO="${VLLM_PORT_AUTO:-true}"
VLLM_PORT_FALLBACK_START="${VLLM_PORT_FALLBACK_START:-18001}"
VLLM_PORT_FALLBACK_END="${VLLM_PORT_FALLBACK_END:-18100}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.95}"

port_is_free() {
  local port="$1"
  local py_bin=""

  py_bin="$(command -v python3 || command -v python || true)"
  if [[ -n "${py_bin}" ]]; then
    "${py_bin}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        sys.exit(1)

sys.exit(0)
PY
    return $?
  fi

  if command -v ss >/dev/null 2>&1; then
    ! ss -ltn | awk '{print $4}' | grep -Eq "[:.]${port}$"
    return $?
  fi

  if command -v nc >/dev/null 2>&1; then
    ! nc -z 127.0.0.1 "${port}" >/dev/null 2>&1
    return $?
  fi

  return 0
}

find_free_port() {
  local start="$1"
  local end="$2"
  local port=""

  for port in $(seq "${start}" "${end}"); do
    if port_is_free "${port}"; then
      echo "${port}"
      return 0
    fi
  done

  return 1
}

if [[ "${VLLM_PORT_AUTO}" = "true" ]] || [[ "${VLLM_PORT_AUTO}" = "1" ]]; then
  if ! port_is_free "${REQUESTED_VLLM_PORT}"; then
    echo "Requested vLLM port ${REQUESTED_VLLM_PORT} is already in use."
    EXISTING_MODELS_JSON="$(curl -f -s --noproxy "*" \
      "http://127.0.0.1:${REQUESTED_VLLM_PORT}/v1/models" 2>/dev/null || true)"
    if [[ -n "${EXISTING_MODELS_JSON}" ]]; then
      echo "Existing /v1/models on port ${REQUESTED_VLLM_PORT}:"
      echo "${EXISTING_MODELS_JSON}"
    fi

    if ! SELECTED_VLLM_PORT="$(find_free_port "${VLLM_PORT_FALLBACK_START}" "${VLLM_PORT_FALLBACK_END}")"; then
      echo "ERROR: no free vLLM port found in ${VLLM_PORT_FALLBACK_START}-${VLLM_PORT_FALLBACK_END}." >&2
      exit 1
    fi

    VLLM_PORT="${SELECTED_VLLM_PORT}"
    echo "Using fallback vLLM port ${VLLM_PORT}."
  fi
else
  if ! port_is_free "${REQUESTED_VLLM_PORT}"; then
    echo "ERROR: requested vLLM port ${REQUESTED_VLLM_PORT} is already in use." >&2
    echo "Set VLLM_PORT to a free port, or keep VLLM_PORT_AUTO=true to auto-select one." >&2
    exit 1
  fi
fi

cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
VLLM_LOG="${LOG_DIR}/vllm_${VLLM_PORT}.log"
: > "${VLLM_LOG}"

# Export variables required inside the Singularity container.
export PROJECT_ROOT CONDA_ENV MODEL_PATH SERVED_MODEL_NAME
export VLLM_HOST VLLM_PORT VLLM_TP_SIZE VLLM_MAX_MODEL_LEN
export VLLM_GPU_MEMORY_UTIL

# -----------------------------------------------------------------------------
# 3. Preserve Slurm GPU binding
# -----------------------------------------------------------------------------
# Do not set CUDA_VISIBLE_DEVICES=0 manually. Slurm may expose the allocated GPU
# through a logical index, UUID, or cgroup device restriction. Preserve whatever
# the scheduler supplied.
echo "===== Slurm/GPU allocation ====="
echo "hostname=${HOSTNAME:-$(hostname)}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "WARNING: CUDA_VISIBLE_DEVICES is unset."
  echo "The script will rely on Slurm cgroups and Singularity --nv."
fi

# Remove explicit container overrides inherited from the submitting shell.
# CUDA_VISIBLE_DEVICES itself is still inherited normally by Singularity.
unset SINGULARITYENV_CUDA_VISIBLE_DEVICES || true
unset APPTAINERENV_CUDA_VISIBLE_DEVICES || true

# -----------------------------------------------------------------------------
# 4. Application environment
# -----------------------------------------------------------------------------
export http_proxy="${HTTP_PROXY_VALUE:-http://174.0.250.13:3128}"
export https_proxy="${HTTP_PROXY_VALUE:-http://174.0.250.13:3128}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="${no_proxy}"

LOCAL_API_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

export API_KEY="${API_KEY:-EMPTY}"
export API_BASE_URL="${LOCAL_API_BASE_URL}"
export MODEL_NAME="${SERVED_MODEL_NAME}"
export PROXY=""

export PLAN_API_KEY="${PLAN_API_KEY:-${API_KEY}}"
export PLAN_API_BASE_URL="${LOCAL_API_BASE_URL}"
export PLAN_MODEL_NAME="${SERVED_MODEL_NAME}"
export PLAN_PROXY=""

export ACT_API_KEY="${ACT_API_KEY:-${API_KEY}}"
export ACT_API_BASE_URL="${LOCAL_API_BASE_URL}"
export ACT_MODEL_NAME="${SERVED_MODEL_NAME}"
export ACT_PROXY=""

export TOOL_API_KEY="${TOOL_API_KEY:-${API_KEY}}"
export TOOL_API_BASE_URL="${LOCAL_API_BASE_URL}"
export TOOL_MODEL_NAME="${SERVED_MODEL_NAME}"
export TOOL_PROXY=""

export VISION_API_KEY="${VISION_API_KEY:-${API_KEY}}"
export VISION_API_BASE_URL="${LOCAL_API_BASE_URL}"
export VISION_MODEL_NAME="${SERVED_MODEL_NAME}"
export VISION_PROXY=""

export CREDIBILITY_API_KEY="${CREDIBILITY_API_KEY:-${API_KEY}}"
export CREDIBILITY_API_BASE_URL="${LOCAL_API_BASE_URL}"
export CREDIBILITY_MODEL_NAME="${SERVED_MODEL_NAME}"
export CREDIBILITY_PROXY=""

export BROWSER_API_KEY="${BROWSER_API_KEY:-${API_KEY}}"
export BROWSER_API_BASE_URL="${LOCAL_API_BASE_URL}"
export BROWSER_MODEL_NAME="${SERVED_MODEL_NAME}"
export BROWSER_PROXY=""

export QUIZ_FILE_PATH="${QUIZ_FILE_PATH:-${PROJECT_ROOT}/data/quiz_repare.jsonl}"
export QUIZ_IMAGE_BASE_DIR="${QUIZ_IMAGE_BASE_DIR:-${PROJECT_ROOT}/data/images}"

export MAX_TOKENS="${MAX_TOKENS:-12288}"
export PLAN_MAX_TOKENS="${PLAN_MAX_TOKENS:-12288}"
export ACT_MAX_TOKENS="${ACT_MAX_TOKENS:-12288}"
export TOOL_MAX_TOKENS="${TOOL_MAX_TOKENS:-12288}"
export VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-8192}"
export VISION_TOOL_MAX_TOKENS="${VISION_TOOL_MAX_TOKENS:-4096}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.12}"
export VISION_TOOL_MAX_RESPONSE_CHARS="${VISION_TOOL_MAX_RESPONSE_CHARS:-20000}"
export MAX_TOOL_CONTENT_LENGTH="${MAX_TOOL_CONTENT_LENGTH:-50000}"
export TOOL_CALL_SAFETY_RATIO="${TOOL_CALL_SAFETY_RATIO:-0.8}"
export TOOL_CALL_CHARS_PER_TOKEN="${TOOL_CALL_CHARS_PER_TOKEN:-0.6}"
export TOOL_CALL_MAX_ARG_CHARS="${TOOL_CALL_MAX_ARG_CHARS:-10000}"
export TOOL_CALL_RETRY_LIMIT="${TOOL_CALL_RETRY_LIMIT:-4}"
export STEP_NOTE_MAX_CHARS="${STEP_NOTE_MAX_CHARS:-4096}"
export TOOL_CALL_RECORD_MAX_CHARS="${TOOL_CALL_RECORD_MAX_CHARS:-4096}"
export TOOL_RESULT_RECORD_MAX_CHARS="${TOOL_RESULT_RECORD_MAX_CHARS:-2048}"
export ENABLE_CONTEXT_COMPRESSION="${ENABLE_CONTEXT_COMPRESSION:-true}"
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-60000}"
export REPORT_MAX_CONTENT_LENGTH="${REPORT_MAX_CONTENT_LENGTH:-60000}"
export REPORT_MAX_FILE_PREVIEW_LENGTH="${REPORT_MAX_FILE_PREVIEW_LENGTH:-8000}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-1200}"

# -----------------------------------------------------------------------------
# 5. Clean up a stale process owned by the current user on the requested port
# -----------------------------------------------------------------------------
echo "Checking port ${VLLM_PORT}..."

if command -v fuser >/dev/null 2>&1; then
  # fuser can normally kill only processes the current user is allowed to kill.
  fuser -k "${VLLM_PORT}/tcp" 2>/dev/null || true
fi

sleep 2

# -----------------------------------------------------------------------------
# 6. Start vLLM
# -----------------------------------------------------------------------------
VLLM_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT

  if [[ -n "${VLLM_PID}" ]] &&
     kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM process ${VLLM_PID}..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi

  exit "${exit_code}"
}

trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "Starting vLLM model server:"
echo "  model=${SERVED_MODEL_NAME}"
echo "  model_path=${MODEL_PATH}"
echo "  address=${VLLM_HOST}:${VLLM_PORT}"

if [[ ! -d "${MODEL_PATH}" ]] || [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: merged model directory is not valid: ${MODEL_PATH}" >&2
  echo "Expected to find ${MODEL_PATH}/config.json." >&2
  exit 1
fi

singularity exec --nv \
  -B "${CONDA_BIND}" \
  -B "${PROJECT_BIND}" \
  -B "/online1/public" \
  "${SIF_IMAGE}" \
  bash -s > "${VLLM_LOG}" 2>&1 <<'VLLM_CONTAINER' &
set -euo pipefail

unset SSL_CERT_FILE
unset SSL_CERT_DIR
unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE

export PATH="${CONDA_ENV}/bin:${PATH}"

export VLLM_USE_FLASHINFER_SAMPLER=0

cd "${PROJECT_ROOT}"

echo "===== Container GPU diagnostics ====="
echo "hostname=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "python=$(command -v python)"

nvidia-smi -L
ls -l /dev/nvidia* || true

# A real CUDA allocation is required. Merely seeing the GPU in nvidia-smi is
# insufficient. Abort before starting vLLM when PyTorch cannot create a CUDA
# context or execute a tensor operation.
python - <<'PY'
import os
import sys

import torch

print("torch_version:", torch.__version__)
print("torch_cuda_version:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count:", torch.cuda.device_count())
print("cuda_available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable inside the Singularity container"
    )

# Force CUDA context creation and execute an actual GPU operation.
torch.cuda.init()

print("device_name:", torch.cuda.get_device_name(0))
print("device_capability:", torch.cuda.get_device_capability(0))

x = torch.ones(4, device="cuda")
print("cuda_tensor_test:", x)

sys.stdout.flush()
PY

echo "===== Starting vLLM ====="

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${VLLM_TP_SIZE}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --trust-remote-code \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTIL}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
VLLM_CONTAINER

VLLM_PID=$!

echo "vLLM launcher PID=${VLLM_PID}"
echo "vLLM log=${VLLM_LOG}"

# -----------------------------------------------------------------------------
# 7. Health check
# -----------------------------------------------------------------------------
echo "Waiting for vLLM to become ready..."

COUNT=0
MAX_RETRIES="${VLLM_MAX_RETRIES:-180}"
LAST_MODELS_JSON=""

model_is_served() {
  local payload="$1"
  MODELS_JSON="${payload}" SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" python - <<'PY'
import json
import os
import sys

payload = os.environ.get("MODELS_JSON", "")
wanted = os.environ["SERVED_MODEL_NAME"]

try:
    data = json.loads(payload)
except Exception:
    sys.exit(1)

ids = [
    item.get("id")
    for item in data.get("data", [])
    if isinstance(item, dict)
]

if wanted in ids:
    sys.exit(0)

sys.exit(1)
PY
}

while true; do
  MODELS_JSON="$(curl -f -s --noproxy "*" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"

  if [[ -n "${MODELS_JSON}" ]]; then
    LAST_MODELS_JSON="${MODELS_JSON}"
  fi

  if [[ -n "${MODELS_JSON}" ]] && model_is_served "${MODELS_JSON}"; then
    break
  fi

  # Do not wait for the full timeout when vLLM has already crashed.
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    VLLM_STATUS=0
    wait "${VLLM_PID}" || VLLM_STATUS=$?

    echo "ERROR: vLLM exited before becoming ready."
    echo "Exit status: ${VLLM_STATUS}"
    echo "===== Last 200 lines of ${VLLM_LOG} ====="

    tail -n 200 "${VLLM_LOG}" || true
    exit 1
  fi

  echo "Waiting for vLLM on port ${VLLM_PORT} to serve model ${SERVED_MODEL_NAME}..."
  echo "Attempt ${COUNT}/${MAX_RETRIES}"

  sleep 10
  COUNT=$((COUNT + 1))

  if [[ "${COUNT}" -ge "${MAX_RETRIES}" ]]; then
    echo "ERROR: vLLM failed to serve the expected model within the configured timeout."
    if [[ -n "${LAST_MODELS_JSON}" ]]; then
      echo "Last /v1/models response:"
      echo "${LAST_MODELS_JSON}"
    fi
    echo "===== Last 200 lines of ${VLLM_LOG} ====="

    tail -n 200 "${VLLM_LOG}" || true
    exit 1
  fi
done

echo "vLLM is ready and serving ${SERVED_MODEL_NAME}."

# -----------------------------------------------------------------------------
# 8. Run cog-research
# -----------------------------------------------------------------------------
echo "Starting cog-research..."

singularity exec --nv \
  -B "${CONDA_BIND}" \
  -B "${PROJECT_BIND}" \
  -B "/online1/public" \
  "${SIF_IMAGE}" \
  bash -s <<'COSIGHT_CONTAINER'
set -euo pipefail

unset SSL_CERT_FILE
unset SSL_CERT_DIR
unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE

export PATH="${CONDA_ENV}/bin:${PATH}"

# Allow runtime exports to take precedence over values in .env.
export DOTENV_OVERRIDE=false

cd "${PROJECT_ROOT}"

exec python CogResearch.py
COSIGHT_CONTAINER

echo "Job finished."
