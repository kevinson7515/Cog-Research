#!/bin/bash

#SBATCH --job-name=cog-research # 指定作业名
#SBATCH -o /home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/long_reason_sample.out # 指定输出文件名;默认为slurm-[jobID].out
#SBATCH -e /home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/long_reason_sample.out # 指定程序错误输出文件名
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:1 # 指定作业使用卡的总数为 1
#SBATCH --nodes=1 # 确保在同一个节点上运行，避免网络延迟

# 1. 加载模块
set -euo pipefail
module load amd/cmake/3.31.8
module load amd/gcc_compiler/13.2.0
module load amd/singularity/3.10.4

cd /home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research

# ---------- Config (paths are server paths; keep as-is by default) ----------
SIF_IMAGE="/online1/hpc_containers/base_hpc_glibc_235.sif"
CONDA_BIND="/online1/hpc_containers/conda/"
PROJECT_BIND="/online1/ycsc_chenkh/hitici_07"
CONDA_ENV="/home/export/base/ycsc_chenkh/hitici_07/online1/anaconda3/envs/cog-research"

MODEL_PATH="/home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/models/Qwen3-VL-8B-Instruct"
SERVED_MODEL_NAME="Qwen3-VL-8B-Instruct"

VLLM_HOST="0.0.0.0"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.95}"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
VLLM_LOG="${LOG_DIR}/vllm_${VLLM_PORT}.log"

# 2. 统一配置环境变量（Singularity 会自动继承这些变量到容器内）
export CUDA_VISIBLE_DEVICES=0
export http_proxy="http://174.0.250.13:3128"
export https_proxy="http://174.0.250.13:3128"
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export VLLM_PORT
export SERVED_MODEL_NAME
LOCAL_API_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export API_KEY="${API_KEY:-EMPTY}"
export API_BASE_URL="${API_BASE_URL:-${LOCAL_API_BASE_URL}}"
export MODEL_NAME="${MODEL_NAME:-${SERVED_MODEL_NAME}}"
export PROXY="http://174.0.250.13:3128"
# Keep planner/actor/tool models pinned to the local vLLM server.
# Vision-specific API settings are intentionally not exported here; they are read from .env.
export PLAN_API_KEY="${PLAN_API_KEY:-${API_KEY}}"
export PLAN_API_BASE_URL="${PLAN_API_BASE_URL:-${LOCAL_API_BASE_URL}}"
export PLAN_MODEL_NAME="${PLAN_MODEL_NAME:-${SERVED_MODEL_NAME}}"
export ACT_API_KEY="${ACT_API_KEY:-${API_KEY}}"
export ACT_API_BASE_URL="${ACT_API_BASE_URL:-${LOCAL_API_BASE_URL}}"
export ACT_MODEL_NAME="${ACT_MODEL_NAME:-${SERVED_MODEL_NAME}}"
export TOOL_API_KEY="${TOOL_API_KEY:-${API_KEY}}"
export TOOL_API_BASE_URL="${TOOL_API_BASE_URL:-${LOCAL_API_BASE_URL}}"
export TOOL_MODEL_NAME="${TOOL_MODEL_NAME:-${SERVED_MODEL_NAME}}"
export MAX_TOKENS="${MAX_TOKENS:-12288}"
export PLAN_MAX_TOKENS="${PLAN_MAX_TOKENS:-12288}"
export ACT_MAX_TOKENS="${ACT_MAX_TOKENS:-12288}"
export TOOL_MAX_TOKENS="${TOOL_MAX_TOKENS:-12288}"
export VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-8192}"
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

# 3. 清理端口
echo "Cleaning up potential zombie processes on port ${VLLM_PORT}..."
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${VLLM_PORT}/tcp" || true
fi
sleep 2

# 4. 启动 vLLM 模型服务
echo "Starting Model Server (${SERVED_MODEL_NAME}) on ${VLLM_HOST}:${VLLM_PORT}..."
singularity exec --nv \
  -B "${CONDA_BIND}" \
  -B "${PROJECT_BIND}" \
  "${SIF_IMAGE}" \
  bash -c "
    source activate ${CONDA_ENV}
    # Some clusters set SSL_CERT_FILE to a non-existent path inside containers,
    # which breaks httpx/OpenAI client initialization even for http:// base_url.
    unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
    python -m vllm.entrypoints.openai.api_server \
      --model ${MODEL_PATH} \
      --served-model-name ${SERVED_MODEL_NAME} \
      --tensor-parallel-size ${VLLM_TP_SIZE} \
      --host ${VLLM_HOST} \
      --port ${VLLM_PORT} \
      --trust-remote-code \
      --max-model-len ${VLLM_MAX_MODEL_LEN} \
      --gpu-memory-utilization ${VLLM_GPU_MEMORY_UTIL} \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_xml
  " > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "Cleaning up vLLM process..."
  kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# 5. 健康检查：耐心等待大模型加载完毕
echo "Waiting for vLLM to be ready (this may take 15-30 minutes due to CUDA graph compilation)..."
COUNT=0
MAX_RETRIES=180 # 增加到最多等 30 分钟
while ! curl -f -s --noproxy "*" "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null; do
  echo "Waiting for vLLM on ${VLLM_PORT}... (${COUNT}/${MAX_RETRIES})"
  sleep 10
  COUNT=$((COUNT + 1))
  if [ "${COUNT}" -ge "${MAX_RETRIES}" ]; then
    echo "Error: vLLM failed to start within timeout. Check ${VLLM_LOG}"
    exit 1
  fi
done
echo "vLLM is UP and Ready!"

# 6. 执行主程序
echo "Starting cog-research..."
singularity exec --nv \
  -B "${CONDA_BIND}" \
  -B "${PROJECT_BIND}" \
  "${SIF_IMAGE}" \
  bash -c "
    source activate ${CONDA_ENV}
    unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE

    # Allow runtime exports to win over .env (see config/config.py)
    export DOTENV_OVERRIDE=false

    python CogResearch.py
  "

echo "Job Finished."
