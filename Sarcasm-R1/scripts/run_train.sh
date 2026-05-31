#!/bin/bash
# Sarcasm-R1 — 训练启动入口
#
# 使用方法:
#   bash scripts/run_train.sh                              # 直接运行 (本地 GPU)
#   bash scripts/run_train.sh --sbatch                     # sbatch 提交到集群
#   bash scripts/run_train.sh --sbatch --gres=gpu:2        # sbatch + 指定 GPU 数
#
# 本地运行前确保:
#   1. pip install -r requirements.txt
#   2. python data/download_data.py --output_dir data/raw
#   3. bash scripts/setup.sh

set -e

# 检查是否用 sbatch 提交
if [ "$1" = "--sbatch" ]; then
    shift
    echo "Submitting to SLURM cluster..."
    # 提取 sbatch 参数
    SBATCH_ARGS=()
    while [[ $# -gt 0 ]]; do
        SBATCH_ARGS+=("$1")
        shift
    done

    if [ ${#SBATCH_ARGS[@]} -gt 0 ]; then
        sbatch "${SBATCH_ARGS[@]}" scripts/sbatch_train.sh
    else
        sbatch scripts/sbatch_train.sh
    fi
    exit 0
fi

# ---- 本地 GPU 直接运行 ----

MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
DATA_PATH="${2:-data/processed/train_combined.csv}"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)

echo "=== Sarcasm-R1 GRPO Training (Local) ==="
echo "Model: $MODEL_PATH"
echo "Data:  $DATA_PATH"
echo "GPUs:  $NUM_GPUS"
echo ""

cd "$(dirname "$0")/../code/grpo"

export SARCASM_DATA_PATH="$(cd ../.. && pwd)/$DATA_PATH"

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch \
        --config_file=deepspeed_zero2.yaml \
        --num_processes "$NUM_GPUS" \
        train.py \
        --config config.yaml \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir ../../output/sarcasm-r1
else
    python train.py \
        --config config.yaml \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir ../../output/sarcasm-r1
fi

echo ""
echo "Training complete. Model: output/sarcasm-r1"
