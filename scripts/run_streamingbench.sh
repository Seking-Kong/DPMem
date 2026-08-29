#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_NAME="${RUN_NAME:-dpmem_streamingbench}"
CKPT_PATH="${QWEN3_VL_CKPT_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
TASK_CSV="${STREAMINGBENCH_TASK_CSV:-data/StreamingBench/StreamingBench/Real_Time_Visual_Understanding.csv}"
VIDEO_DIR="${STREAMINGBENCH_VIDEO_DIR:-data/StreamingBench/Real-Time Visual Understanding}"
RESULT_DIR="${RESULT_DIR:-outputs/streamingbench}"

NUM_CHUNKS="${NUM_CHUNKS:-1}"
GPUS_PER_WORKER="${GPUS_PER_WORKER:-1}"
FPS="${FPS:-1}"
FRAMES_PER_CHUNK="${FRAMES_PER_CHUNK:-40}"
DROP_METHOD="${DROP_METHOD:-feature}"
DROP_THRESHOLD="${DROP_THRESHOLD:-0.5}"
DROP_MODE="${DROP_MODE:-both}"
MAX_NEW_TOKENS_INTERMEDIATE="${MAX_NEW_TOKENS_INTERMEDIATE:-256}"
MAX_NEW_TOKENS_FINAL="${MAX_NEW_TOKENS_FINAL:-64}"

python scripts/streamingbench_dpmem.py \
    --run_name "$RUN_NAME" \
    --ckpt_path "$CKPT_PATH" \
    --task_csv "$TASK_CSV" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --num_chunks "$NUM_CHUNKS" \
    --gpus_per_worker "$GPUS_PER_WORKER" \
    --fps "$FPS" \
    --frames_per_chunk "$FRAMES_PER_CHUNK" \
    --drop_method "$DROP_METHOD" \
    --drop_threshold "$DROP_THRESHOLD" \
    --drop_mode "$DROP_MODE" \
    --max_new_tokens_intermediate "$MAX_NEW_TOKENS_INTERMEDIATE" \
    --max_new_tokens_final "$MAX_NEW_TOKENS_FINAL"
