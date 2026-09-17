#!/bin/bash

#SBATCH --job-name=workflow
#SBATCH -o /home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/sft_%j.out
#SBATCH -e /home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/logs/sft_%j.out
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --gres=gpu:3
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

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3-VL-8B-Instruct}"
TRACE_DIR="${TRACE_DIR:-${PROJECT_ROOT}/trace}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${PROJECT_ROOT}/data/sft_traces}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/qwen3-vl-8b-sft-lora}"

# 32k context can OOM even on 95GB GPUs during backward for Qwen3-VL-8B.
# Override MAX_SEQ_LENGTH at submission time if you want to try a longer run.
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SAVE_STEPS="${SAVE_STEPS:-100}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
INCLUDE_TRACES="${INCLUDE_TRACES:-planner,actor}"
SKIP_OVERLENGTH="${SKIP_OVERLENGTH:-true}"
SFT_SAMPLE_MODE="${SFT_SAMPLE_MODE:-turn}"
SFT_CONTEXT_MESSAGES="${SFT_CONTEXT_MESSAGES:-8}"
SFT_MAX_TOOL_RESULT_CHARS="${SFT_MAX_TOOL_RESULT_CHARS:-4000}"
SFT_MAX_ARG_CHARS="${SFT_MAX_ARG_CHARS:-6000}"
SFT_STRIP_TOOL_PREAMBLE="${SFT_STRIP_TOOL_PREAMBLE:-true}"
SFT_TOOL_ONLY="${SFT_TOOL_ONLY:-true}"
SFT_ADD_WORKFLOW_SAMPLES="${SFT_ADD_WORKFLOW_SAMPLES:-true}"
SFT_WORKFLOW_CONTEXT_MESSAGES="${SFT_WORKFLOW_CONTEXT_MESSAGES:-14}"
SFT_WORKFLOW_MAX_TARGET_MESSAGES="${SFT_WORKFLOW_MAX_TARGET_MESSAGES:-24}"
# Exact workflow duplication amplified long report/save templates in the current
# SFT corpus. Keep one copy by default; opt in only after a deduplicated audit.
SFT_WORKFLOW_REPEAT="${SFT_WORKFLOW_REPEAT:-1}"
SFT_WORKFLOW_ANCHOR_TOOLS="${SFT_WORKFLOW_ANCHOR_TOOLS:-file_saver,generate_markdown_report}"
SFT_WORKFLOW_TERMINAL_TOOLS="${SFT_WORKFLOW_TERMINAL_TOOLS:-mark_step}"
SFT_WORKFLOW_PREVIOUS_TOOLS="${SFT_WORKFLOW_PREVIOUS_TOOLS:-ask_question_about_image,ask_question_about_video,audio_recognition,serper_search,image_search,fetch_website_content,fetch_website_content_with_images,fetch_website_images_only,file_read,extract_document_content,execute_code,file_find_in_content}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/config/deepspeed_zero3_bf16.json}"
MASTER_PORT="${MASTER_PORT:-29501}"
SFT_DRY_RUN="${SFT_DRY_RUN:-false}"
HTTP_PROXY_VALUE="${HTTP_PROXY_VALUE:-http://174.0.250.13:3128}"
HTTPS_PROXY_VALUE="${HTTPS_PROXY_VALUE:-http://174.0.250.13:3128}"

mkdir -p logs "${SFT_DATA_DIR}" "${SFT_OUTPUT_DIR}"

echo "Preparing SFT data from ${TRACE_DIR}..."
singularity exec --nv \
  -B "${CONDA_BIND}" \
  -B "${PROJECT_BIND}" \
  "${SIF_IMAGE}" \
  bash -c "
    source activate ${CONDA_ENV}
    unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
    export http_proxy='${HTTP_PROXY_VALUE}'
    export https_proxy='${HTTPS_PROXY_VALUE}'
    export HTTP_PROXY='${HTTP_PROXY_VALUE}'
    export HTTPS_PROXY='${HTTPS_PROXY_VALUE}'
    python scripts/prepare_sft_from_traces.py \
      --trace_dir '${TRACE_DIR}' \
      --output_dir '${SFT_DATA_DIR}' \
      --include '${INCLUDE_TRACES}' \
      --sample_mode '${SFT_SAMPLE_MODE}' \
      --context_messages '${SFT_CONTEXT_MESSAGES}' \
      --max_tool_result_chars '${SFT_MAX_TOOL_RESULT_CHARS}' \
      --max_arg_chars '${SFT_MAX_ARG_CHARS}' \
      --strip_tool_preamble '${SFT_STRIP_TOOL_PREAMBLE}' \
      --tool_only '${SFT_TOOL_ONLY}' \
      --add_workflow_samples '${SFT_ADD_WORKFLOW_SAMPLES}' \
      --workflow_context_messages '${SFT_WORKFLOW_CONTEXT_MESSAGES}' \
      --workflow_max_target_messages '${SFT_WORKFLOW_MAX_TARGET_MESSAGES}' \
      --workflow_repeat '${SFT_WORKFLOW_REPEAT}' \
      --workflow_anchor_tools '${SFT_WORKFLOW_ANCHOR_TOOLS}' \
      --workflow_terminal_tools '${SFT_WORKFLOW_TERMINAL_TOOLS}' \
      --workflow_previous_tools '${SFT_WORKFLOW_PREVIOUS_TOOLS}' \
      --dedupe \
      --shuffle
  "

echo "Starting SFT training..."
singularity exec --nv \
  -B "${CONDA_BIND}" \
  -B "${PROJECT_BIND}" \
  "${SIF_IMAGE}" \
  bash -c "
    source activate ${CONDA_ENV}
    unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
    export http_proxy='${HTTP_PROXY_VALUE}'
    export https_proxy='${HTTPS_PROXY_VALUE}'
    export HTTP_PROXY='${HTTP_PROXY_VALUE}'
    export HTTPS_PROXY='${HTTPS_PROXY_VALUE}'
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
    torchrun \
      --nnodes 1 \
      --nproc_per_node '${NPROC_PER_NODE}' \
      --master_addr 127.0.0.1 \
      --master_port '${MASTER_PORT}' \
      scripts/train_sft.py \
      --model_name_or_path '${MODEL_PATH}' \
      --train_file '${SFT_DATA_DIR}/sft_messages.jsonl' \
      --output_dir '${SFT_OUTPUT_DIR}' \
      --max_seq_length '${MAX_SEQ_LENGTH}' \
      --dry_run '${SFT_DRY_RUN}' \
      --skip_overlength '${SKIP_OVERLENGTH}' \
      --num_train_epochs '${EPOCHS}' \
      --learning_rate '${LR}' \
      --per_device_train_batch_size '${BATCH_SIZE}' \
      --gradient_accumulation_steps '${GRAD_ACCUM}' \
      --save_steps '${SAVE_STEPS}' \
      --use_lora true \
      --disable_bnb true \
      --lora_r '${LORA_R}' \
      --lora_alpha '${LORA_ALPHA}' \
      --deepspeed '${DEEPSPEED_CONFIG}' \
      --bf16 true \
      --gradient_checkpointing true
  "

echo "SFT finished. Output: ${SFT_OUTPUT_DIR}"
