#!/bin/bash

#SBATCH --job-name=cog-research-merged
#SBATCH -o /home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/merged_%j.out
#SBATCH -e /home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/merged_%j.out
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

set -euo pipefail

module load amd/cmake/3.31.8
module load amd/gcc_compiler/13.2.0
module load amd/singularity/3.10.4

PROJECT_ROOT="${PROJECT_ROOT:-/home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research}"
cd "${PROJECT_ROOT}"

SIF_IMAGE="${SIF_IMAGE:-/online1/hpc_containers/base_hpc_glibc_235.sif}"
CONDA_BIND="${CONDA_BIND:-/online1/hpc_containers/conda/}"
PROJECT_BIND="${PROJECT_BIND:-/online1/ycsc_chenkh/hitici_07}"
CONDA_ENV="${CONDA_ENV:-/home/export/base/ycsc_chenkh/hitici_07/online1/anaconda3/envs/cog-research}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/outputs/qwen3-vl-8b-sft-merged}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b-sft-merged}"

VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.95}"

QUIZ_FILE_PATH="${QUIZ_FILE_PATH:-${PROJECT_ROOT}/data/quiz.jsonl}"
QUIZ_IMAGE_BASE_DIR="${QUIZ_IMAGE_BASE_DIR:-${PROJECT_ROOT}/data/images}"
COSIGHT_TRACE_DIR="${COSIGHT_TRACE_DIR:-${PROJECT_ROOT}/trace_merged}"

HTTP_PROXY_VALUE="${HTTP_PROXY_VALUE:-http://174.0.250.13:3128}"
HTTPS_PROXY_VALUE="${HTTPS_PROXY_VALUE:-http://174.0.250.13:3128}"
NO_PROXY_VALUE="${NO_PROXY_VALUE:-localhost,127.0.0.1,0.0.0.0}"
PROXY_VALUE="${PROXY_VALUE:-${HTTP_PROXY_VALUE}}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}" "${COSIGHT_TRACE_DIR}"
VLLM_LOG="${LOG_DIR}/vllm_merged_${VLLM_PORT}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export http_proxy="${HTTP_PROXY_VALUE}"
export https_proxy="${HTTPS_PROXY_VALUE}"
export HTTP_PROXY="${HTTP_PROXY_VALUE}"
export HTTPS_PROXY="${HTTPS_PROXY_VALUE}"
export no_proxy="${NO_PROXY_VALUE}"
export NO_PROXY="${NO_PROXY_VALUE}"

LOCAL_API_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
LOCAL_API_KEY="${LOCAL_API_KEY:-EMPTY}"

# Keep all non-vision roles on the merged model served by local vLLM.
export API_KEY="${API_KEY:-${LOCAL_API_KEY}}"
export API_BASE_URL="${LOCAL_API_BASE_URL}"
export MODEL_NAME="${SERVED_MODEL_NAME}"
export PLAN_API_KEY="${PLAN_API_KEY:-${LOCAL_API_KEY}}"
export PLAN_API_BASE_URL="${LOCAL_API_BASE_URL}"
export PLAN_MODEL_NAME="${SERVED_MODEL_NAME}"
export ACT_API_KEY="${ACT_API_KEY:-${LOCAL_API_KEY}}"
export ACT_API_BASE_URL="${LOCAL_API_BASE_URL}"
export ACT_MODEL_NAME="${SERVED_MODEL_NAME}"
export TOOL_API_KEY="${TOOL_API_KEY:-${LOCAL_API_KEY}}"
export TOOL_API_BASE_URL="${LOCAL_API_BASE_URL}"
export TOOL_MODEL_NAME="${SERVED_MODEL_NAME}"
export CREDIBILITY_API_KEY="${CREDIBILITY_API_KEY:-${LOCAL_API_KEY}}"
export CREDIBILITY_API_BASE_URL="${LOCAL_API_BASE_URL}"
export CREDIBILITY_MODEL_NAME="${SERVED_MODEL_NAME}"
export BROWSER_API_KEY="${BROWSER_API_KEY:-${LOCAL_API_KEY}}"
export BROWSER_API_BASE_URL="${LOCAL_API_BASE_URL}"
export BROWSER_MODEL_NAME="${SERVED_MODEL_NAME}"

# The OpenAI/httpx client in llm.py uses explicit per-role proxy settings.
# Local vLLM roles must not use the cluster HTTP proxy, otherwise localhost
# requests are sent to Squid and fail with ERR_ACCESS_DENIED.
export PROXY=""
export PLAN_PROXY=""
export ACT_PROXY=""
export TOOL_PROXY=""
export CREDIBILITY_PROXY=""
export BROWSER_PROXY=""
# Vision remains external. VISION_API_KEY / VISION_API_BASE_URL /
# VISION_MODEL_NAME / VISION_PROXY can come from .env or VISION_* overrides.

export QUIZ_FILE_PATH
export QUIZ_IMAGE_BASE_DIR
export COSIGHT_TRACE_DIR

export MAX_TOKENS="${MAX_TOKENS:-12288}"
export PLAN_MAX_TOKENS="${PLAN_MAX_TOKENS:-12288}"
export ACT_MAX_TOKENS="${ACT_MAX_TOKENS:-12288}"
export TOOL_MAX_TOKENS="${TOOL_MAX_TOKENS:-12288}"
export VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-8192}"
export CREDIBILITY_MAX_TOKENS="${CREDIBILITY_MAX_TOKENS:-8192}"
export BROWSER_MAX_TOKENS="${BROWSER_MAX_TOKENS:-8192}"
export VISION_TOOL_MAX_TOKENS="${VISION_TOOL_MAX_TOKENS:-4096}"
export VISION_TOOL_MAX_RESPONSE_CHARS="${VISION_TOOL_MAX_RESPONSE_CHARS:-20000}"
export MAX_TOOL_CONTENT_LENGTH="${MAX_TOOL_CONTENT_LENGTH:-50000}"
export TOOL_CALL_SAFETY_RATIO="${TOOL_CALL_SAFETY_RATIO:-0.8}"
export TOOL_CALL_CHARS_PER_TOKEN="${TOOL_CALL_CHARS_PER_TOKEN:-0.6}"
export TOOL_CALL_MAX_ARG_CHARS="${TOOL_CALL_MAX_ARG_CHARS:-6000}"
export TOOL_CALL_RETRY_LIMIT="${TOOL_CALL_RETRY_LIMIT:-4}"
export STEP_NOTE_MAX_CHARS="${STEP_NOTE_MAX_CHARS:-4096}"
export TOOL_CALL_RECORD_MAX_CHARS="${TOOL_CALL_RECORD_MAX_CHARS:-4096}"
export TOOL_RESULT_RECORD_MAX_CHARS="${TOOL_RESULT_RECORD_MAX_CHARS:-2048}"
export ENABLE_CONTEXT_COMPRESSION="${ENABLE_CONTEXT_COMPRESSION:-true}"
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-60000}"
export REPORT_MAX_CONTENT_LENGTH="${REPORT_MAX_CONTENT_LENGTH:-60000}"
export REPORT_MAX_FILE_PREVIEW_LENGTH="${REPORT_MAX_FILE_PREVIEW_LENGTH:-8000}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-900}"

echo "Testing cog-research with merged SFT model."
echo "Project root: ${PROJECT_ROOT}"
echo "Merged model path: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Quiz file: ${QUIZ_FILE_PATH}"
echo "Image dir: ${QUIZ_IMAGE_BASE_DIR}"
echo "Trace dir: ${COSIGHT_TRACE_DIR}"
echo "Local API: ${LOCAL_API_BASE_URL}"
echo "Vision model is external and will be loaded from .env or VISION_* env vars."

echo "Cleaning up potential zombie processes on port ${VLLM_PORT}..."
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${VLLM_PORT}/tcp" || true
fi
sleep 2

echo "Starting merged model server on ${VLLM_HOST}:${VLLM_PORT}..."
singularity exec --nv \
  -B "${CONDA_BIND}" \
  -B "${PROJECT_BIND}" \
  "${SIF_IMAGE}" \
  bash -c "
    set -euo pipefail
    source activate '${CONDA_ENV}'
    unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
    python -m vllm.entrypoints.openai.api_server \
      --model '${MODEL_PATH}' \
      --served-model-name '${SERVED_MODEL_NAME}' \
      --tensor-parallel-size '${VLLM_TP_SIZE}' \
      --host '${VLLM_HOST}' \
      --port '${VLLM_PORT}' \
      --trust-remote-code \
      --max-model-len '${VLLM_MAX_MODEL_LEN}' \
      --gpu-memory-utilization '${VLLM_GPU_MEMORY_UTIL}' \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_xml
  " > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "Cleaning up vLLM process..."
  kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM to be ready. Check ${VLLM_LOG} if this takes too long."
COUNT=0
MAX_RETRIES="${MAX_RETRIES:-180}"
while ! curl -f -s --noproxy "*" "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null; do
  echo "Waiting for vLLM on ${VLLM_PORT}... (${COUNT}/${MAX_RETRIES})"
  sleep 10
  COUNT=$((COUNT + 1))
  if [ "${COUNT}" -ge "${MAX_RETRIES}" ]; then
    echo "Error: vLLM failed to start within timeout. Check ${VLLM_LOG}"
    exit 1
  fi
done
echo "vLLM is ready."

singularity exec --nv \
  -B "${CONDA_BIND}" \
  -B "${PROJECT_BIND}" \
  "${SIF_IMAGE}" \
  bash -c "
    set -euo pipefail
    cd '${PROJECT_ROOT}'
    source activate '${CONDA_ENV}'

    unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
    export DOTENV_OVERRIDE=false

    python - <<'PY'
from config.config import (
    get_plan_model_config,
    get_act_model_config,
    get_tool_model_config,
    get_vision_model_config,
    get_credibility_model_config,
    get_browser_model_config,
)

configs = {
    'plan': get_plan_model_config(),
    'act': get_act_model_config(),
    'tool': get_tool_model_config(),
    'vision': get_vision_model_config(),
    'credibility': get_credibility_model_config(),
    'browser': get_browser_model_config(),
}

print('Resolved model config:')
for name, cfg in configs.items():
    print(
        f\"  {name}: model={cfg.get('model')}, \"
        f\"base_url={cfg.get('base_url')}, \"
        f\"proxy={cfg.get('proxy')}, \"
        f\"api_key_set={bool(cfg.get('api_key')) and cfg.get('api_key') != 'EMPTY'}\"
    )
PY

    python CoSight.py
  "

echo "Merged-model test finished."
echo "Trace output: ${COSIGHT_TRACE_DIR}"
