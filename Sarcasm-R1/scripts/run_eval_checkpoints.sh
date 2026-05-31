#!/bin/bash
#SBATCH --job-name=sarcasm_eval
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
export TRANSFORMERS_NO_TF=1

OUT_DIR=$(pwd)/output/sarcasm-r1-v5
MODEL_DIR=$(pwd)/models/Qwen2.5-7B-Instruct

for CKPT in checkpoint-20 checkpoint-40 checkpoint-60 checkpoint-80 checkpoint-100; do
    if [ -d "$OUT_DIR/$CKPT" ]; then
        echo "=== Evaluating $CKPT ==="
        python code/evaluation/inference.py --model $OUT_DIR/$CKPT --base_model $MODEL_DIR --data data/processed/semeval_test.csv --output results/v5_${CKPT}_predictions.jsonl --batch_size 2
        python code/evaluation/evaluate.py --results results/v5_${CKPT}_predictions.jsonl --output results/v5_${CKPT}_eval.json
    else
        echo "=== Skipping $CKPT (not found) ==="
    fi
done

echo "=== All done ==="
