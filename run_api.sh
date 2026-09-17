#!/bin/bash

#SBATCH --job-name=cog-research
#SBATCH -o /home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/long_reason_sample.out
#SBATCH -e /home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/long_reason_sample.out
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

QUIZ_FILE_PATH="${QUIZ_FILE_PATH:-${PROJECT_ROOT}/data/quiz_train.jsonl}"
QUIZ_IMAGE_BASE_DIR="${QUIZ_IMAGE_BASE_DIR:-${PROJECT_ROOT}/data/images}"
CogResearch_TRACE_DIR="${CogResearch_TRACE_DIR:-${PROJECT_ROOT}/trace}"

HTTP_PROXY_VALUE="${HTTP_PROXY_VALUE:-http://174.0.250.13:3128}"
HTTPS_PROXY_VALUE="${HTTPS_PROXY_VALUE:-http://174.0.250.13:3128}"
NO_PROXY_VALUE="${NO_PROXY_VALUE:-localhost,127.0.0.1,0.0.0.0}"
PROXY_VALUE="${PROXY_VALUE:-${HTTP_PROXY_VALUE}}"

# API-mode non-secret model routing. Keys are still read from .env.
# Override these with API_RUN_BASE_URL/API_RUN_MODEL_NAME if needed.
API_RUN_BASE_URL="${API_RUN_BASE_URL:-https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1}"
API_RUN_MODEL_NAME="${API_RUN_MODEL_NAME:-qwen3.7-plus}"
VISION_RUN_MODEL_NAME="${VISION_RUN_MODEL_NAME:-qwen3.7-plus}"

# Keep exported endpoint/model values authoritative. With DOTENV_OVERRIDE=false,
# python-dotenv fills missing API keys from .env but does not overwrite these
# API-mode routing values with stale local-vLLM settings.
DOTENV_OVERRIDE_VALUE="${DOTENV_OVERRIDE:-false}"
THINKING_MODE_VALUE="${THINKING_MODE:-false}"

# Runtime defaults.
MAX_TOKENS="${MAX_TOKENS:-12288}"
PLAN_MAX_TOKENS="${PLAN_MAX_TOKENS:-12288}"
ACT_MAX_TOKENS="${ACT_MAX_TOKENS:-12288}"
TOOL_MAX_TOKENS="${TOOL_MAX_TOKENS:-12288}"
VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-8192}"
CREDIBILITY_MAX_TOKENS="${CREDIBILITY_MAX_TOKENS:-8192}"
BROWSER_MAX_TOKENS="${BROWSER_MAX_TOKENS:-8192}"
VISION_TOOL_MAX_TOKENS="${VISION_TOOL_MAX_TOKENS:-4096}"
VISION_TOOL_MAX_RESPONSE_CHARS="${VISION_TOOL_MAX_RESPONSE_CHARS:-20000}"
MAX_TOOL_CONTENT_LENGTH="${MAX_TOOL_CONTENT_LENGTH:-50000}"
TOOL_CALL_SAFETY_RATIO="${TOOL_CALL_SAFETY_RATIO:-0.8}"
TOOL_CALL_CHARS_PER_TOKEN="${TOOL_CALL_CHARS_PER_TOKEN:-0.6}"
TOOL_CALL_MAX_ARG_CHARS="${TOOL_CALL_MAX_ARG_CHARS:-6000}"
TOOL_CALL_RETRY_LIMIT="${TOOL_CALL_RETRY_LIMIT:-4}"
STEP_NOTE_MAX_CHARS="${STEP_NOTE_MAX_CHARS:-4096}"
TOOL_CALL_RECORD_MAX_CHARS="${TOOL_CALL_RECORD_MAX_CHARS:-4096}"
TOOL_RESULT_RECORD_MAX_CHARS="${TOOL_RESULT_RECORD_MAX_CHARS:-2048}"
ENABLE_CONTEXT_COMPRESSION="${ENABLE_CONTEXT_COMPRESSION:-true}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-60000}"
REPORT_MAX_CONTENT_LENGTH="${REPORT_MAX_CONTENT_LENGTH:-60000}"
REPORT_MAX_FILE_PREVIEW_LENGTH="${REPORT_MAX_FILE_PREVIEW_LENGTH:-8000}"
LLM_TIMEOUT="${LLM_TIMEOUT:-900}"

mkdir -p "${PROJECT_ROOT}/logs" "${CogResearch_TRACE_DIR}"

echo "Starting cog-research trajectory generation in API mode."
echo "Project root: ${PROJECT_ROOT}"
echo "Quiz file: ${QUIZ_FILE_PATH}"
echo "Image dir: ${QUIZ_IMAGE_BASE_DIR}"
echo "Trace dir: ${CogResearch_TRACE_DIR}"
echo "API base URL: ${API_RUN_BASE_URL}"
echo "Planner/actor/tool model: ${API_RUN_MODEL_NAME}"
echo "Vision model: ${VISION_RUN_MODEL_NAME}"
echo "API keys will be loaded from .env and will not be printed."

singularity exec \
  -B "${CONDA_BIND}" \
  -B "${PROJECT_BIND}" \
  "${SIF_IMAGE}" \
  bash -lc "
    set -euo pipefail
    cd '${PROJECT_ROOT}'
    source activate '${CONDA_ENV}'

    unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE

    export DOTENV_OVERRIDE='${DOTENV_OVERRIDE_VALUE}'
    export THINKING_MODE='${THINKING_MODE_VALUE}'
    export PLAN_THINKING_MODE='${THINKING_MODE_VALUE}'
    export ACT_THINKING_MODE='${THINKING_MODE_VALUE}'
    export TOOL_THINKING_MODE='${THINKING_MODE_VALUE}'
    export VISION_THINKING_MODE='${THINKING_MODE_VALUE}'
    export CREDIBILITY_THINKING_MODE='${THINKING_MODE_VALUE}'
    export BROWSER_THINKING_MODE='${THINKING_MODE_VALUE}'

    export http_proxy='${HTTP_PROXY_VALUE}'
    export https_proxy='${HTTPS_PROXY_VALUE}'
    export HTTP_PROXY='${HTTP_PROXY_VALUE}'
    export HTTPS_PROXY='${HTTPS_PROXY_VALUE}'
    export no_proxy='${NO_PROXY_VALUE}'
    export NO_PROXY='${NO_PROXY_VALUE}'
    export PROXY='${PROXY_VALUE}'
    export PLAN_PROXY='${PROXY_VALUE}'
    export ACT_PROXY='${PROXY_VALUE}'
    export TOOL_PROXY='${PROXY_VALUE}'
    export VISION_PROXY='${PROXY_VALUE}'
    export CREDIBILITY_PROXY='${PROXY_VALUE}'
    export BROWSER_PROXY='${PROXY_VALUE}'

    export API_BASE_URL='${API_RUN_BASE_URL}'
    export MODEL_NAME='${API_RUN_MODEL_NAME}'
    export PLAN_API_BASE_URL='${API_RUN_BASE_URL}'
    export PLAN_MODEL_NAME='${API_RUN_MODEL_NAME}'
    export ACT_API_BASE_URL='${API_RUN_BASE_URL}'
    export ACT_MODEL_NAME='${API_RUN_MODEL_NAME}'
    export TOOL_API_BASE_URL='${API_RUN_BASE_URL}'
    export TOOL_MODEL_NAME='${API_RUN_MODEL_NAME}'
    export CREDIBILITY_API_BASE_URL='${API_RUN_BASE_URL}'
    export CREDIBILITY_MODEL_NAME='${API_RUN_MODEL_NAME}'
    export BROWSER_API_BASE_URL='${API_RUN_BASE_URL}'
    export BROWSER_MODEL_NAME='${API_RUN_MODEL_NAME}'
    export VISION_API_BASE_URL='${API_RUN_BASE_URL}'
    export VISION_MODEL_NAME='${VISION_RUN_MODEL_NAME}'

    export QUIZ_FILE_PATH='${QUIZ_FILE_PATH}'
    export QUIZ_IMAGE_BASE_DIR='${QUIZ_IMAGE_BASE_DIR}'
    export CogResearch_TRACE_DIR='${CogResearch_TRACE_DIR}'

    export MAX_TOKENS='${MAX_TOKENS}'
    export PLAN_MAX_TOKENS='${PLAN_MAX_TOKENS}'
    export ACT_MAX_TOKENS='${ACT_MAX_TOKENS}'
    export TOOL_MAX_TOKENS='${TOOL_MAX_TOKENS}'
    export VISION_MAX_TOKENS='${VISION_MAX_TOKENS}'
    export CREDIBILITY_MAX_TOKENS='${CREDIBILITY_MAX_TOKENS}'
    export BROWSER_MAX_TOKENS='${BROWSER_MAX_TOKENS}'
    export VISION_TOOL_MAX_TOKENS='${VISION_TOOL_MAX_TOKENS}'
    export VISION_TOOL_MAX_RESPONSE_CHARS='${VISION_TOOL_MAX_RESPONSE_CHARS}'
    export MAX_TOOL_CONTENT_LENGTH='${MAX_TOOL_CONTENT_LENGTH}'
    export TOOL_CALL_SAFETY_RATIO='${TOOL_CALL_SAFETY_RATIO}'
    export TOOL_CALL_CHARS_PER_TOKEN='${TOOL_CALL_CHARS_PER_TOKEN}'
    export TOOL_CALL_MAX_ARG_CHARS='${TOOL_CALL_MAX_ARG_CHARS}'
    export TOOL_CALL_RETRY_LIMIT='${TOOL_CALL_RETRY_LIMIT}'
    export STEP_NOTE_MAX_CHARS='${STEP_NOTE_MAX_CHARS}'
    export TOOL_CALL_RECORD_MAX_CHARS='${TOOL_CALL_RECORD_MAX_CHARS}'
    export TOOL_RESULT_RECORD_MAX_CHARS='${TOOL_RESULT_RECORD_MAX_CHARS}'
    export ENABLE_CONTEXT_COMPRESSION='${ENABLE_CONTEXT_COMPRESSION}'
    export MAX_CONTEXT_TOKENS='${MAX_CONTEXT_TOKENS}'
    export REPORT_MAX_CONTENT_LENGTH='${REPORT_MAX_CONTENT_LENGTH}'
    export REPORT_MAX_FILE_PREVIEW_LENGTH='${REPORT_MAX_FILE_PREVIEW_LENGTH}'
    export LLM_TIMEOUT='${LLM_TIMEOUT}'

    python - <<'PY'
import sys
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

bad = []
print('Resolved API model config:')
for name, cfg in configs.items():
    base_url = cfg.get('base_url') or ''
    model = cfg.get('model') or ''
    proxy = cfg.get('proxy') or ''
    thinking = cfg.get('thinking_mode')
    has_key = bool(cfg.get('api_key')) and cfg.get('api_key') != 'EMPTY'
    print(f'  {name}: model={model}, base_url={base_url}, proxy={proxy}, thinking_mode={thinking}, api_key_set={has_key}')
    if '127.0.0.1' in base_url or 'localhost' in base_url:
        bad.append(f'{name} still points to local base_url: {base_url}')
    if name in {'plan', 'act', 'tool'} and model == 'Qwen3-VL-8B-Instruct':
        bad.append(f'{name} still points to local model: {model}')
    if not has_key:
        bad.append(f'{name} api_key is missing or EMPTY')

if bad:
    print('API preflight failed:', file=sys.stderr)
    for item in bad:
        print(f'  - {item}', file=sys.stderr)
    sys.exit(2)
PY

    python CogResearch.py
  "

echo "API trajectory generation finished."
echo "Trace output: ${CogResearch_TRACE_DIR}"
