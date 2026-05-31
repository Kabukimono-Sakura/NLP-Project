#!/bin/bash
# ============================================================
# Sarcasm-R1 — 环境搭建脚本
#
# 使用方法:
#   bash scripts/setup.sh                    # 完整搭建 (本地)
#   bash scripts/setup.sh --cluster          # 集群环境 (仅装依赖)
#   bash scripts/setup.sh --cluster --data   # 集群 + 数据准备
# ============================================================

set -e

ON_CLUSTER=false
DO_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster) ON_CLUSTER=true; shift ;;
        --data)    DO_DATA=true; shift ;;
        *) echo "Unknown option: $1"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Sarcasm-R1 Environment Setup ==="
echo "Project: $PROJECT_DIR"
echo "Cluster: $ON_CLUSTER"
echo ""

# ---- 1. 安装依赖 ----
echo "[1/3] Installing Python dependencies..."
if [ "$ON_CLUSTER" = true ]; then
    source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
    conda activate base
fi

pip install -r requirements.txt -q
echo "  Done."
echo ""

# ---- 2. 数据 ----
if [ "$ON_CLUSTER" = true ]; then
    # 集群模式: 使用离线缓存
    export HF_HOME="$PROJECT_DIR/hf_cache"
    export TRANSFORMERS_CACHE="$PROJECT_DIR/hf_cache"
    export HF_DATASETS_CACHE="$PROJECT_DIR/hf_cache"
    export HF_HUB_CACHE="$PROJECT_DIR/hf_cache"
fi

if [ "$DO_DATA" = true ] || [ "$ON_CLUSTER" = false ]; then
    echo "[2/3] Downloading and processing datasets..."
    mkdir -p data/raw data/processed

    # 下载数据
    cd "$PROJECT_DIR/data"
    python download_data.py --output_dir ./raw || {
        echo "[WARN] Download failed. For clusters, pre-download locally and upload."
    }

    # 处理数据
    cd "$PROJECT_DIR/code/data_processing"
    python process_semeval.py --input_dir "$PROJECT_DIR/data/raw/SemEval2018-Task3" \
                              --output_dir "$PROJECT_DIR/data/processed" 2>/dev/null || true
    python process_mustard.py --input_dir "$PROJECT_DIR/data/raw/MUStARD" \
                              --output_dir "$PROJECT_DIR/data/processed" 2>/dev/null || true
    python format_grpo_data.py --input_dir "$PROJECT_DIR/data/processed" \
                               --output_dir "$PROJECT_DIR/data/processed" --combine
    echo "  Done."
else
    echo "[2/3] Skipping data (use --data flag on cluster, or run locally)."
fi

echo ""

# ---- 3. 环境信息 ----
echo "[3/3] Environment check..."
python --version
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "CUDA    : $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "TRL     : $(python -c 'import trl; print(trl.__version__)' 2>/dev/null || echo 'not found')"
echo ""

echo "=== Setup Complete ==="
echo ""
if [ "$ON_CLUSTER" = true ]; then
    echo "Submit training job:"
    echo "  sbatch scripts/run_all.sh"
else
    echo "Start training:"
    echo "  bash scripts/run_train.sh"
    echo ""
    echo "Or prepare for cluster upload:"
    echo "  bash scripts/prepare_offline.sh"
fi
