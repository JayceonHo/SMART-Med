#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/dataset/MedVR-3K"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-$REPO_ROOT/output/grpo/}"
GRPO_DATASET="${GRPO_DATASET:-$REPO_ROOT/dataset/MedVR-3K/grpo.jsonl}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

export VLM_JUDGE_BASE_MODEL="${VLM_JUDGE_BASE_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
export VLM_JUDGE_LORA="${VLM_JUDGE_LORA:-}"

REWARD_ARGS=(--reward_funcs medical_vqa_acc medical_vqa_format medical_vqa_judge
             --reward_weights 1 2 1)

NPROC_PER_NODE="$NPROC_PER_NODE" \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --adapters "$REPO_ROOT/output/new_output/sft_r2/v0-20261021-181747/checkpoint-325"\
    --external_plugins "$REPO_ROOT/scripts/plugin/medical_vqa_rewards.py" \
    "${REWARD_ARGS[@]}" \
    --use_vllm false \
    --tuner_type lora \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --torch_dtype bfloat16 \
    --dataset "$GRPO_DATASET" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.1 \
    --max_completion_length 768 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --learning_rate 3e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 20 \
    --save_steps 20 \
    --save_total_limit 5 \
    --logging_steps 20 \
    --output_dir "$GRPO_OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system "$REPO_ROOT/scripts/prompt.txt" \
    --log_completions true \
    --report_to none \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
    "$@"
