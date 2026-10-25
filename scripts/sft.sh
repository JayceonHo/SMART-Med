#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/dataset/MedVR-3K"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-$REPO_ROOT/output/sft/}"
SFT_DATASET="${SFT_DATASET:-sft.jsonl}"
SFT_EPOCHS="${SFT_EPOCHS:-5}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
NPROC_PER_NODE="$NPROC_PER_NODE" \
swift sft \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --dataset "./$SFT_DATASET" \
    --load_from_cache_file true \
    --output_dir "$SFT_OUTPUT_DIR" \
    --split_dataset_ratio 0.05 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs "$SFT_EPOCHS" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 6 \
    --attn_impl sdpa \
    --padding_free false \
    --learning_rate 1e-4 \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --packing false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 6 \
    --eval_steps 20 \
    --save_steps 20 \
    --save_total_limit 5 \
    --logging_steps 25 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    "$@"
