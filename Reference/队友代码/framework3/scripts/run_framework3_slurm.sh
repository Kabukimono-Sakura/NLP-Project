#!/usr/bin/env bash
#SBATCH --job-name=framework3_8
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=framework3/logs/framework3_%j.out
#SBATCH --error=framework3/logs/framework3_%j.err

set -euo pipefail

# SLURM entrypoint for the eight controlled Framework3 experiments.
#
# Usage examples:
#   sbatch framework3/scripts/run_framework3_slurm.sh
#   DATASET=semeval200 sbatch framework3/scripts/run_framework3_slurm.sh
#   DATASET=semeval784 sbatch framework3/scripts/run_framework3_slurm.sh
#   DATASET=all       sbatch framework3/scripts/run_framework3_slurm.sh
#
# Required input:
#   QWEN_MODEL or TEACHER_MODEL: path or HF repo for Qwen3-14B
#
# Example:
#   export QWEN_MODEL=models/Qwen3-14B
#   DATASET=all sbatch framework3/scripts/run_framework3_slurm.sh

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

mkdir -p framework3/logs framework3/predictions framework3/results

PYTHON_BIN="${PYTHON:-python3}"
DATASET="${DATASET:-all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_TORCHVISION=1

echo "=========================================="
echo "Framework3 eight-experiment SLURM job"
echo "Job ID       : ${SLURM_JOB_ID:-local}"
echo "Node         : ${SLURMD_NODENAME:-local}"
echo "Project dir  : $PROJECT_DIR"
echo "Dataset      : $DATASET"
echo "Qwen model   : ${QWEN_MODEL:-${TEACHER_MODEL:-UNSET}}"
echo "Max tokens   : $MAX_NEW_TOKENS"
echo "Start        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

if [[ -z "${QWEN_MODEL:-${TEACHER_MODEL:-}}" ]]; then
  echo "[ERROR] Set QWEN_MODEL or TEACHER_MODEL."
  exit 1
fi

echo "[INFO] Python environment"
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
try:
    import torch
    import transformers
    import accelerate
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("accelerate:", accelerate.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("environment check failed:", repr(exc))
    raise
PY

bash framework3/scripts/run_eight_experiments.sh

echo "=========================================="
echo "Framework3 finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results:"
find framework3/results -maxdepth 2 -type f -print | sort
echo "=========================================="
