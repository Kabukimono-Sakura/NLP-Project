#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: train Framework2/Sarcasm-R1 with the local Qwen3-14B.
# It reuses the original Framework2 GRPO implementation and writes to:
#   framework2/Sarcasm-R1/output/sarcasm-r1

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

BASE_MODEL="${BASE_MODEL:-${QWEN_MODEL:-${TEACHER_MODEL:-models/Qwen3-14B}}}"
DATA_PATH="${FRAMEWORK2_TRAIN_DATA:-data/processed/semeval_train.csv}"

if [[ -d "$BASE_MODEL" ]]; then
  BASE_MODEL="$(cd "$BASE_MODEL" && pwd)"
fi

echo "=========================================="
echo "Framework2 GRPO Training with Qwen3-14B"
echo "Model: $BASE_MODEL"
echo "Data : framework2/Sarcasm-R1/$DATA_PATH"
echo "=========================================="

cd framework2/Sarcasm-R1
bash scripts/run_train.sh "$BASE_MODEL" "$DATA_PATH"
