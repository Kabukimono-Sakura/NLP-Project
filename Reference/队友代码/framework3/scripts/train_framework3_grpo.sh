#!/usr/bin/env bash
set -euo pipefail

# Train the Framework3 hybrid PMP-RCE policy with GRPO.
#
# Default data:
#   framework2/Sarcasm-R1/data/processed/semeval_train.csv
#
# Default output:
#   framework3/checkpoints/framework3-hybrid-grpo-qwen3-14b

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:-${QWEN_MODEL:-${TEACHER_MODEL:-models/Qwen3-14B}}}"
DATA_PATH="${SARCASM_DATA_PATH:-framework2/Sarcasm-R1/data/processed/semeval_train.csv}"
OUTPUT_DIR="${FRAMEWORK3_GRPO_OUTPUT:-framework3/checkpoints/framework3-hybrid-grpo-qwen3-14b}"
CONFIG_FILE="${GRPO_CONFIG:-framework3/configs/framework3_grpo_config.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-framework3/configs/deepspeed_zero2.yaml}"

if [[ -d "$BASE_MODEL" ]]; then
  BASE_MODEL="$(cd "$BASE_MODEL" && pwd)"
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "[ERROR] Training data not found: $DATA_PATH"
  exit 1
fi

export SARCASM_DATA_PATH="$DATA_PATH"
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_TORCHVISION=1

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "$NUM_GPUS" || "$NUM_GPUS" == "0" ]]; then
  NUM_GPUS=1
fi

echo "=========================================="
echo "Framework3 Hybrid GRPO Training"
echo "Project : $PROJECT_DIR"
echo "Model   : $BASE_MODEL"
echo "Data    : $DATA_PATH"
echo "Output  : $OUTPUT_DIR"
echo "GPUs    : $NUM_GPUS"
echo "=========================================="

if [[ "$NUM_GPUS" -gt 1 ]]; then
  accelerate launch \
    --config_file "$DEEPSPEED_CONFIG" \
    --num_processes "$NUM_GPUS" \
    framework3/scripts/train_framework3_grpo.py \
    --config "$CONFIG_FILE" \
    --model_name_or_path "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR"
else
  "$PYTHON_BIN" framework3/scripts/train_framework3_grpo.py \
    --config "$CONFIG_FILE" \
    --model_name_or_path "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR"
fi

echo "Framework3 GRPO training complete: $OUTPUT_DIR"
