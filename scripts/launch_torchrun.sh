#!/bin/bash
# Launch script for 8x L4 GPU training with torchrun (native PyTorch DDP)
# Usage: ./scripts/launch_torchrun.sh [model_size] [--use_memory]

set -e

MODEL_SIZE="${1:-1B}"
USE_MEMORY="${2:-}"

echo "========================================"
echo "BitNet-ODP DDP Training (torchrun)"
echo "========================================"
echo "Model size: $MODEL_SIZE"
echo "GPUs: 8x NVIDIA L4"
echo "========================================"

# L4 Ada optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Change to project root
cd "$(dirname "$0")/.."

# Ensure output directory exists
mkdir -p checkpoints

# Run with torchrun
if [ "$USE_MEMORY" == "--use_memory" ]; then
    echo "Training with MIRAS memory enabled"
    torchrun \
        --nproc_per_node=8 \
        --master_port=29500 \
        training/train_ddp.py \
        --model_size "$MODEL_SIZE" \
        --use_memory \
        --batch_size 16 \
        --gradient_accumulation 4 \
        --max_seq_len 2048 \
        --learning_rate 1e-4 \
        --warmup_steps 4000 \
        --max_steps 200000 \
        --log_every 50 \
        --eval_every 500 \
        --save_every 2500
else
    echo "Training without MIRAS memory"
    torchrun \
        --nproc_per_node=8 \
        --master_port=29500 \
        training/train_ddp.py \
        --model_size "$MODEL_SIZE" \
        --batch_size 16 \
        --gradient_accumulation 4 \
        --max_seq_len 2048 \
        --learning_rate 1e-4 \
        --warmup_steps 4000 \
        --max_steps 200000 \
        --log_every 50 \
        --eval_every 500 \
        --save_every 2500
fi

echo "Training complete!"
