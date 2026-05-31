#!/bin/bash
#SBATCH --job-name=sr1_eval
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/sr1_eval_%j.out
#SBATCH --error=logs/sr1_eval_%j.err

# ============================================================
# Sarcasm-R1 — 仅评估 (需先完成训练)
#
# 使用方法:
#   sbatch scripts/sbatch_eval.sh
# ============================================================

set -e
mkdir -p logs results

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
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_TORCHVISION=1

echo ""
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# 模型路径 — 支持传入 checkpoint 子目录
# 用法: sbatch scripts/sbatch_eval.sh                     # 评估最终模型
#       sbatch scripts/sbatch_eval.sh checkpoint-500      # 评估 checkpoint-500
CHECKPOINT="${1:-}"
OUTPUT_BASE="$SLURM_SUBMIT_DIR/output/sarcasm-r1-v3"
if [ -n "$CHECKPOINT" ]; then
    OUTPUT_DIR="$OUTPUT_BASE/$CHECKPOINT"
else
    OUTPUT_DIR="$OUTPUT_BASE"
fi
MODEL_DIR="$SLURM_SUBMIT_DIR/models"
if [ -d "$MODEL_DIR/Qwen2.5-7B-Instruct" ]; then
    BASE_MODEL="$MODEL_DIR/Qwen2.5-7B-Instruct"
elif [ -d "$MODEL_DIR/Qwen2.5-1.5B-Instruct" ]; then
    BASE_MODEL="$MODEL_DIR/Qwen2.5-1.5B-Instruct"
else
    BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "[ERROR] Trained model not found at $OUTPUT_DIR"
    echo "        Run training first: sbatch scripts/sbatch_train.sh"
    exit 1
fi

echo "[INFO] Trained model: $OUTPUT_DIR"
echo "[INFO] Base model:    $BASE_MODEL"
echo ""

# ============================================================
# 评估所有可用的测试集
# ============================================================

cd "$SLURM_SUBMIT_DIR/code/evaluation"

for TEST_DATA in "$SLURM_SUBMIT_DIR/data/processed/"*_test.csv \
                 "$SLURM_SUBMIT_DIR/data/processed/"*_validation.csv; do
    if [ ! -f "$TEST_DATA" ]; then
        continue
    fi

    DATASET_NAME=$(basename "$TEST_DATA" .csv)
    RESULT_FILE="$SLURM_SUBMIT_DIR/results/${DATASET_NAME}_predictions.jsonl"
    REPORT_FILE="$SLURM_SUBMIT_DIR/results/${DATASET_NAME}_eval.json"

    echo "--- Evaluating: $DATASET_NAME ---"

    # 推理
    python inference.py \
        --model "$OUTPUT_DIR" \
        --base_model "$BASE_MODEL" \
        --data "$TEST_DATA" \
        --output "$RESULT_FILE" \
        --batch_size 4

    # 评估
    python evaluate.py \
        --results "$RESULT_FILE" \
        --output "$REPORT_FILE" \
        --name "Sarcasm-R1"

    echo ""
done

# ============================================================
# 汇总
# ============================================================
echo "=========================================="
echo " EVALUATION DONE"
echo " End Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " Results:"
for f in "$SLURM_SUBMIT_DIR/results/"*_eval.json; do
    if [ -f "$f" ]; then
        echo "  --- $(basename $f) ---"
        python -c "
import json
with open('$f') as fp:
    m = json.load(fp)
print(f\"    Accuracy : {m['accuracy']:.4f}\")
print(f\"    Macro-F1 : {m['macro_f1']:.4f}\")
for label, cm in m.get('class_metrics', {}).items():
    print(f\"    {label}: P={cm['precision']:.4f} R={cm['recall']:.4f} F1={cm['f1']:.4f}\")
" 2>/dev/null
    fi
done
echo "=========================================="
