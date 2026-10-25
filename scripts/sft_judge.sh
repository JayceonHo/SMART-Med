#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/dataset/MedVR-3K"

# Use four processes by default; set CUDA_VISIBLE_DEVICES at launch time to choose GPUs.
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
NPROC_PER_NODE="$NPROC_PER_NODE" \
swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset './judge.jsonl' \
    --load_from_cache_file true \
    --output_dir "$REPO_ROOT/output/judge_augmented/" \
    --split_dataset_ratio 0.1 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 6 \
    --attn_impl sdpa \
    --padding_free false \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
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
    --logging_steps 10 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    "$@"
