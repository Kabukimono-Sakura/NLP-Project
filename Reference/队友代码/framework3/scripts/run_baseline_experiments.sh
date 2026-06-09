#!/usr/bin/env bash
set -euo pipefail

# Run only the no-framework baseline on SemEval-200 and/or SemEval-784.
# This script does not train or load any GRPO checkpoint.
#
# Usage:
#   QWEN_MODEL=/path/to/Qwen3-14B DATASET=all bash framework3/scripts/run_baseline_experiments.sh
#   DATASET=semeval200 bash framework3/scripts/run_baseline_experiments.sh
#   DATASET=semeval784 bash framework3/scripts/run_baseline_experiments.sh

PYTHON_BIN="${PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:-${QWEN_MODEL:-${TEACHER_MODEL:-models/Qwen3-14B}}}"
DATASET="${DATASET:-all}"
PRED_DIR="${PRED_DIR:-framework3/predictions}"
RESULT_DIR="${RESULT_DIR:-framework3/results}"
BASELINE_PROMPT="${BASELINE_PROMPT:-framework3/prompts/baseline_prompt.md}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
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

  echo "[RUN] $name baseline prediction -> $output"
  echo "      model : $BASE_MODEL"
  echo "      prompt: $BASELINE_PROMPT"

  "$PYTHON_BIN" framework3/scripts/run_llm_predictions.py \
    --input "$input" \
    --output "$output" \
    --model "$BASE_MODEL" \
    --prompt-file "$BASELINE_PROMPT" \
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

  local pred="$PRED_DIR/${tag}_baseline.jsonl"
  local eval="$RESULT_DIR/${tag}_baseline_eval.json"

  echo "=========================================="
  echo "Baseline dataset: $tag"
  echo "Input           : $input"
  echo "Model           : $BASE_MODEL"
  echo "Max new tokens  : $MAX_NEW_TOKENS"
  echo "=========================================="

  run_prediction "${tag}/baseline" "$input" "$pred" "$id_col" "$text_col" "$gold_col"
  run_eval "${tag}_baseline" "$pred" "$eval"
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
echo "Baseline run finished"
echo "Predictions:"
find "$PRED_DIR" -maxdepth 1 -type f -name "*_baseline.jsonl" -print | sort
echo "Results:"
find "$RESULT_DIR" -maxdepth 1 -type f -name "*_baseline_eval.json" -print | sort
echo "=========================================="
