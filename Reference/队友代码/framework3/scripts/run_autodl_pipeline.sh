#!/usr/bin/env bash
set -euo pipefail

# AutoDL-friendly entrypoint.
#
# Typical use after uploading the project:
#   export QWEN_MODEL=/root/autodl-tmp/models/Qwen3-14B
#   export TRAIN_FRAMEWORK2=1
#   export TRAIN_FRAMEWORK3=1
#   bash framework3/scripts/run_autodl_pipeline.sh
#
# If GRPO checkpoints already exist, leave TRAIN_FRAMEWORK2/TRAIN_FRAMEWORK3 at 0
# and set FRAMEWORK2_MODEL / FRAMEWORK3_MODEL before running.

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_TORCHVISION=1

BASE_MODEL="${BASE_MODEL:-${QWEN_MODEL:-${TEACHER_MODEL:-models/Qwen3-14B}}}"
TRAIN_FRAMEWORK2="${TRAIN_FRAMEWORK2:-0}"
TRAIN_FRAMEWORK3="${TRAIN_FRAMEWORK3:-0}"
DATASET="${DATASET:-all}"

echo "=========================================="
echo "AutoDL Framework3 Pipeline"
echo "Project          : $PROJECT_DIR"
echo "Base model       : $BASE_MODEL"
echo "Dataset          : $DATASET"
echo "Train Framework2 : $TRAIN_FRAMEWORK2"
echo "Train Framework3 : $TRAIN_FRAMEWORK3"
echo "=========================================="

if [[ "$TRAIN_FRAMEWORK2" == "1" ]]; then
  BASE_MODEL="$BASE_MODEL" bash framework3/scripts/train_framework2_grpo_qwen3.sh
fi

if [[ "$TRAIN_FRAMEWORK3" == "1" ]]; then
  BASE_MODEL="$BASE_MODEL" bash framework3/scripts/train_framework3_grpo.sh
fi

DATASET="$DATASET" BASE_MODEL="$BASE_MODEL" bash framework3/scripts/run_eight_experiments.sh
