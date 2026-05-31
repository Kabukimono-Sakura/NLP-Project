#!/bin/bash
# Sarcasm-R1 — 评估启动入口
#
# 使用方法:
#   bash scripts/run_eval.sh                                # 本地 GPU 直接运行
#   bash scripts/run_eval.sh --sbatch                       # sbatch 提交到集群
#   bash scripts/run_eval.sh ./output/sarcasm-r1            # 指定模型路径
#   bash scripts/run_eval.sh ./output/sarcasm-r1 semeval    # 指定测试集

set -e

if [ "$1" = "--sbatch" ]; then
    shift
    echo "Submitting evaluation to SLURM cluster..."
    if [ $# -gt 0 ]; then
        sbatch "$@" scripts/sbatch_eval.sh
    else
        sbatch scripts/sbatch_eval.sh
    fi
    exit 0
fi

# ---- 本地 GPU 直接运行 ----

MODEL_PATH="${1:-./output/sarcasm-r1}"
DATASET="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Sarcasm-R1 Evaluation (Local) ==="
echo "Model: $MODEL_PATH"
echo ""

cd "$PROJECT_DIR/code/evaluation"

for TEST_DATA in "$PROJECT_DIR/data/processed/"*_test.csv \
                 "$PROJECT_DIR/data/processed/"*_validation.csv; do
    if [ ! -f "$TEST_DATA" ]; then
        continue
    fi

    DATASET_NAME=$(basename "$TEST_DATA" .csv)
    RESULT_FILE="$PROJECT_DIR/results/${DATASET_NAME}_predictions.jsonl"
    REPORT_FILE="$PROJECT_DIR/results/${DATASET_NAME}_eval.json"

    echo "--- Evaluating: $DATASET_NAME ---"

    python inference.py \
        --model "$MODEL_PATH" \
        --data "$TEST_DATA" \
        --output "$RESULT_FILE" \
        --batch_size 4

    python evaluate.py \
        --results "$RESULT_FILE" \
        --output "$REPORT_FILE" \
        --name "Sarcasm-R1"

    echo ""
done

echo "Evaluation complete. Results in: results/"
