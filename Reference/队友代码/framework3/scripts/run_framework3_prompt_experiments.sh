#!/usr/bin/env bash
set -euo pipefail

# Run only the prompt-based hybrid PMP-RCE method on SemEval-200 and/or SemEval-784.
# This script uses the original base model and the Framework3 hybrid prompt.
# It does not train or load any GRPO checkpoint.

PYTHON_BIN="${PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:-${QWEN_MODEL:-${TEACHER_MODEL:-models/Qwen3-14B}}}"
DATASET="${DATASET:-all}"
PRED_DIR="${PRED_DIR:-framework3/predictions}"
RESULT_DIR="${RESULT_DIR:-framework3/results}"
FRAMEWORK3_PROMPT="${FRAMEWORK3_PROMPT:-framework3/prompts/framework3_hybrid_prompt.md}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
FORCE="${FORCE:-0}"

mkdir -p "$PRED_DIR" "$RESULT_DIR"

run_prediction() {
  local name="$1"
  local input="$2"
  local output="$3"
  local id_col="$4"
  local text_col="$5"
  local gold_col="$6"

  if [[ "$FORCE" != "1" && -s "$output" ]]; then
    echo "[SKIP] $name prediction exists: $output"
    return
  fi

  echo "[RUN] $name hybrid PMP-RCE prediction -> $output"
  echo "      model : $BASE_MODEL"
  echo "      prompt: $FRAMEWORK3_PROMPT"

  "$PYTHON_BIN" framework3/scripts/run_llm_predictions.py \
    --input "$input" \
    --output "$output" \
    --model "$BASE_MODEL" \
    --prompt-file "$FRAMEWORK3_PROMPT" \
    --id-column "$id_col" \
    --text-column "$text_col" \
    --gold-column "$gold_col" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    --trust-remote-code
}

run_eval() {
  local name="$1"
  local input="$2"
  local output="$3"

  echo "[EVAL] $name -> $output"
  "$PYTHON_BIN" framework3/scripts/evaluate_predictions.py \
    --input "$input" \
    --output "$output" \
    --name "$name"
}

run_dataset() {
  local tag="$1"
  local input="$2"
  local id_col="$3"
  local text_col="$4"
  local gold_col="$5"

  local pred="$PRED_DIR/${tag}_framework3.jsonl"
  local eval="$RESULT_DIR/${tag}_framework3_eval.json"

  echo "=========================================="
  echo "Hybrid dataset : $tag"
  echo "Input          : $input"
  echo "Model          : $BASE_MODEL"
  echo "Max new tokens : $MAX_NEW_TOKENS"
  echo "=========================================="

  run_prediction "${tag}/framework3" "$input" "$pred" "$id_col" "$text_col" "$gold_col"
  run_eval "${tag}_framework3" "$pred" "$eval"
}

case "$DATASET" in
  semeval200)
    run_dataset semeval200 framework1/semeval200.tsv sample_id text label
    ;;
  semeval784)
    run_dataset semeval784 framework2/Sarcasm-R1/data/processed/semeval_test.csv index source_text answer
    ;;
  all)
    run_dataset semeval200 framework1/semeval200.tsv sample_id text label
    run_dataset semeval784 framework2/Sarcasm-R1/data/processed/semeval_test.csv index source_text answer
    ;;
  *)
    echo "[ERROR] Unknown DATASET=$DATASET. Use semeval200, semeval784, or all."
    exit 1
    ;;
esac

echo "=========================================="
echo "Prompt-only hybrid framework3 run finished"
echo "Predictions:"
find "$PRED_DIR" -maxdepth 1 -type f -name "*_framework3.jsonl" -print | sort
echo "Results:"
find "$RESULT_DIR" -maxdepth 1 -type f -name "*_framework3_eval.json" -print | sort
echo "=========================================="
