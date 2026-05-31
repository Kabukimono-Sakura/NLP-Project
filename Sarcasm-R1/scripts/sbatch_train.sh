#!/bin/bash
#SBATCH --job-name=sr1_train
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/sr1_train_%j.out
#SBATCH --error=logs/sr1_train_%j.err

# ============================================================
# Sarcasm-R1 — 仅 GRPO 训练 (不含数据准备和评估)
# 适合数据已准备好、反复调参的场景
#
# 使用方法:
#   sbatch scripts/sbatch_train.sh
#   sbatch scripts/sbatch_train.sh Qwen/Qwen2.5-1.5B-Instruct
# ============================================================

set -e
mkdir -p logs output

echo "=========================================="
echo " Job ID    : $SLURM_JOB_ID"
echo " Node      : $SLURMD_NODENAME"
echo " Start     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"

# 环境配置
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate base

export HF_HOME="$SLURM_SUBMIT_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
export HF_DATASETS_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
export HF_HUB_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
export WANDB_MODE=disabled
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_TORCHVISION=1

# 环境信息
echo ""
python --version
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# 模型路径
MODEL_DIR="$SLURM_SUBMIT_DIR/models"
if [ -d "$MODEL_DIR/Qwen2.5-7B-Instruct" ]; then
    MODEL_PATH="$MODEL_DIR/Qwen2.5-7B-Instruct"
elif [ -d "$MODEL_DIR/Qwen2.5-1.5B-Instruct" ]; then
    MODEL_PATH="$MODEL_DIR/Qwen2.5-1.5B-Instruct"
else
    MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
fi

# 训练数据
TRAIN_DATA="$SLURM_SUBMIT_DIR/data/processed/train_combined.csv"
if [ ! -f "$TRAIN_DATA" ]; then
    for f in "$SLURM_SUBMIT_DIR/data/processed/"train_*.csv; do
        if [ -f "$f" ]; then TRAIN_DATA="$f"; break; fi
    done
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "[ERROR] No training data found. Run data preparation first."
    exit 1
fi

OUTPUT_DIR="$SLURM_SUBMIT_DIR/output/sarcasm-r1-v3"
NUM_GPUS=$(nvidia-smi -L | wc -l)

export SARCASM_DATA_PATH="$TRAIN_DATA"

echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Data : $TRAIN_DATA"
echo "[INFO] GPUs : $NUM_GPUS"
echo ""

cd "$SLURM_SUBMIT_DIR/code/grpo"

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch \
        --config_file=deepspeed_zero2.yaml \
        --num_processes "$NUM_GPUS" \
        train.py \
        --config config.yaml \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --report_to none
else
    python train.py \
        --config config.yaml \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --use_vllm false \
        --report_to none \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_generations 2 \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 10
fi

echo ""
echo "[$(date '+%H:%M:%S')] Training done."
echo "Model saved to: $OUTPUT_DIR"
echo "  $(ls -lh $OUTPUT_DIR/ 2>/dev/null | head -5 || echo 'output dir missing')"
