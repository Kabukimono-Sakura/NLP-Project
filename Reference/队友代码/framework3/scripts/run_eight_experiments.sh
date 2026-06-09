#!/usr/bin/env bash
set -euo pipefail

# Run eight controlled experiments:
#   SemEval-200: baseline, Framework1, Framework2, Framework3
#   SemEval-784: baseline, Framework1, Framework2, Framework3
#
# Model usage:
#   baseline   = original Qwen3-14B
#   framework1 = original Qwen3-14B + PMP prompt
#   framework2 = Framework2 GRPO checkpoint + RCE prompt
#   framework3 = Framework3 hybrid GRPO checkpoint + PMP-RCE prompt

PYTHON_BIN="${PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:-${QWEN_MODEL:-${TEACHER_MODEL:-models/Qwen3-14B}}}"
FRAMEWORK1_MODEL="${FRAMEWORK1_MODEL:-$BASE_MODEL}"
FRAMEWORK2_MODEL="${FRAMEWORK2_MODEL:-${GRPO_MODEL:-framework2/Sarcasm-R1/output/sarcasm-r1}}"
FRAMEWORK3_MODEL="${FRAMEWORK3_MODEL:-framework3/checkpoints/framework3-hybrid-grpo-qwen3-14b}"
DATASET="${DATASET:-all}"
PRED_DIR="${PRED_DIR:-framework3/predictions}"
RESULT_DIR="${RESULT_DIR:-framework3/results}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
FORCE="${FORCE:-0}"
ALLOW_BASE_FALLBACK="${ALLOW_BASE_FALLBACK:-0}"

BASELINE_PROMPT="${BASELINE_PROMPT:-framework3/prompts/baseline_prompt.md}"
FRAMEWORK1_PROMPT="${FRAMEWORK1_PROMPT:-framework3/prompts/teacher_full_pmp_prompt.md}"
FRAMEWORK2_PROMPT="${FRAMEWORK2_PROMPT:-framework3/prompts/pmp_r1_system_prompt.md}"
FRAMEWORK3_PROMPT="${FRAMEWORK3_PROMPT:-framework3/prompts/framework3_hybrid_prompt.md}"

mkdir -p "$PRED_DIR" "$RESULT_DIR"

ensure_grpo_model() {
  local label="$1"
  local model_path="$2"
  local fallback="$3"

  if [[ -d "$model_path" || -f "$model_path" ]]; then
    return
  fi

  if [[ "$ALLOW_BASE_FALLBACK" == "1" ]]; then
    echo "[WARN] $label checkpoint not found: $model_path"
    echo "[WARN] Falling back to base model: $fallback"
    return
  fi

  echo "[ERROR] $label must use a GRPO checkpoint, but this path does not exist:"
  echo "        $model_path"
  echo ""
  echo "Train or set the checkpoint first, for example:"
  echo "  bash framework3/scripts/train_framework2_grpo_qwen3.sh"
  echo "  bash framework3/scripts/train_framework3_grpo.sh"
  echo ""
  echo "Or explicitly set:"
  echo "  FRAMEWORK2_MODEL=/path/to/framework2/grpo/checkpoint"
  echo "  FRAMEWORK3_MODEL=/path/to/framework3/grpo/checkpoint"
  exit 1
}

resolve_model() {
  local label="$1"
  local model_path="$2"
  local fallback="$3"
  if [[ ! -e "$model_path" && "$ALLOW_BASE_FALLBACK" == "1" && "$label" != "baseline" && "$label" != "framework1" ]]; then
    printf "%s" "$fallback"
  else
    printf "%s" "$model_path"
  fi
}

run_prediction() {
  local name="$1"
  local input="$2"
  local output="$3"
  local model="$4"
  local prompt_file="$5"
  local id_col="$6"
  local text_col="$7"
  local gold_col="$8"

  if [[ "$FORCE" != "1" && -s "$output" ]]; then
    echo "[SKIP] $name prediction exists: $output"
    return
  fi

  echo "[RUN] $name prediction -> $output"
  echo "      model : $model"
  echo "      prompt: $prompt_file"
  "$PYTHON_BIN" framework3/scripts/run_llm_predictions.py \
    --input "$input" \
    --output "$output" \
    --model "$model" \
    --base-model "$BASE_MODEL" \
    --prompt-file "$prompt_file" \
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

run_method() {
  local tag="$1"
  local method="$2"
  local input="$3"
  local model="$4"
  local prompt_file="$5"
  local id_col="$6"
  local text_col="$7"
  local gold_col="$8"

  local pred="$PRED_DIR/${tag}_${method}.jsonl"
  local eval="$RESULT_DIR/${tag}_${method}_eval.json"

  run_prediction "${tag}/${method}" "$input" "$pred" "$model" "$prompt_file" "$id_col" "$text_col" "$gold_col"
  run_eval "${tag}_${method}" "$pred" "$eval"
}

run_dataset() {
  local tag="$1"
  local input="$2"
  local id_col="$3"
  local text_col="$4"
  local gold_col="$5"

  local f2_model
  local f3_model
  f2_model="$(resolve_model framework2 "$FRAMEWORK2_MODEL" "$BASE_MODEL")"
  f3_model="$(resolve_model framework3 "$FRAMEWORK3_MODEL" "$BASE_MODEL")"

  echo "=========================================="
  echo "Dataset          : $tag"
  echo "Input            : $input"
  echo "Base model       : $BASE_MODEL"
  echo "Framework2 model : $f2_model"
  echo "Framework3 model : $f3_model"
  echo "=========================================="

  run_method "$tag" baseline "$input" "$BASE_MODEL" "$BASELINE_PROMPT" "$id_col" "$text_col" "$gold_col"
  run_method "$tag" framework1 "$input" "$FRAMEWORK1_MODEL" "$FRAMEWORK1_PROMPT" "$id_col" "$text_col" "$gold_col"
  run_method "$tag" framework2 "$input" "$f2_model" "$FRAMEWORK2_PROMPT" "$id_col" "$text_col" "$gold_col"
  run_method "$tag" framework3 "$input" "$f3_model" "$FRAMEWORK3_PROMPT" "$id_col" "$text_col" "$gold_col"
}

ensure_grpo_model framework2 "$FRAMEWORK2_MODEL" "$BASE_MODEL"
ensure_grpo_model framework3 "$FRAMEWORK3_MODEL" "$BASE_MODEL"

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
echo "Eight-experiment run finished"
echo "Predictions:"
find "$PRED_DIR" -maxdepth 1 -type f -print | sort
echo "Results:"
find "$RESULT_DIR" -maxdepth 1 -type f -print | sort
echo "=========================================="
