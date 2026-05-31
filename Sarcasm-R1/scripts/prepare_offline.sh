#!/bin/bash
# ============================================================
# Sarcasm-R1 — 本地离线准备脚本
# 在有网络的本地机器上运行，预下载模型和数据，
# 然后将整个目录上传到集群
#
# 使用方法:
#   bash scripts/prepare_offline.sh                    # 下载全部 (7B)
#   bash scripts/prepare_offline.sh --model-size 1.5b  # 使用 1.5B 模型
#   bash scripts/prepare_offline.sh --skip-model       # 跳过模型下载
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODEL_SIZE="7b"
SKIP_MODEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-size) MODEL_SIZE="$2"; shift 2 ;;
        --skip-model) SKIP_MODEL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " Sarcasm-R1 Offline Preparation"
echo " Project dir: $PROJECT_DIR"
echo " Model size:  $MODEL_SIZE"
echo "============================================================"
echo ""

# ---- 1. 下载模型 ----
if [ "$SKIP_MODEL" = false ]; then
    MODEL_NAME="Qwen/Qwen2.5-${MODEL_SIZE^^}-Instruct"
    # Normalize model name (7b -> 7B, 1.5b -> 1.5B)
    MODEL_NAME="Qwen/Qwen2.5-$(echo $MODEL_SIZE | sed 's/b$/B/' | sed 's/\(^[0-9]*\)\([Bb]\)/\1\U\2/')-Instruct"

    echo "[1/3] Downloading model: $MODEL_NAME"
    echo "  Saving to: models/$(basename $MODEL_NAME)"

    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
model_name = '$MODEL_NAME'
save_dir = 'models/' + model_name.split('/')[-1]
os.makedirs(save_dir, exist_ok=True)
print(f'  Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)
print(f'  Downloading model (this may take a while)...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto',
    trust_remote_code=True,
)
model.save_pretrained(save_dir)
print(f'  Model saved to {save_dir}')
print(f'  Size: {sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(save_dir) for f in fns) / 1e9:.2f} GB')
"
    echo ""
else
    echo "[1/3] Skipping model download (--skip-model)"
    echo ""
fi

# ---- 2. 下载数据 ----
echo "[2/3] Downloading datasets..."
cd "$PROJECT_DIR/data"
python download_data.py --output_dir ./raw
echo ""

# ---- 3. HuggingFace 缓存 ----
echo "[3/3] Setting up HF cache for offline use..."
mkdir -p "$PROJECT_DIR/hf_cache"

echo ""
echo "============================================================"
echo " Preparation Complete!"
echo ""
echo " Upload to cluster:"
echo "   rsync -avz --progress $PROJECT_DIR/ <cluster>:$PROJECT_DIR/"
echo "   # or"
echo "   scp -r $PROJECT_DIR <cluster>:~/"
echo ""
echo " Then submit on cluster:"
echo "   sbatch scripts/run_all.sh"
echo "============================================================"

cd "$PROJECT_DIR"
echo ""
echo " Directory sizes:"
du -sh models/ data/ hf_cache/ 2>/dev/null || true
