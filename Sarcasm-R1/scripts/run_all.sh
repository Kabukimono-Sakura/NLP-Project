#!/bin/bash
#SBATCH --job-name=sarcasm_r1
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sarcasm_r1_%j.out
#SBATCH --error=logs/sarcasm_r1_%j.err

# ============================================================
# Sarcasm-R1 — GRPO 讽刺检测训练完整流程
# ============================================================
#
# 集群目录结构 (上传到集群后的预期布局):
#   Sarcasm-R1/
#   ├── scripts/run_all.sh        ← 本脚本 (sbatch 提交入口)
#   ├── code/
#   │   ├── grpo/                 ← GRPO 训练代码
#   │   ├── data_processing/      ← 数据处理代码
#   │   ├── evaluation/           ← 评估代码
#   │   └── reward_model/         ← 奖励模型 (可选)
#   ├── data/
#   │   ├── raw/                  ← 原始数据 (需预下载)
#   │   └── processed/            ← 处理后的训练数据
#   ├── models/                   ← 预下载的模型权重
#   │   └── Qwen2.5-7B-Instruct/  (或 Qwen2.5-1.5B-Instruct/)
#   ├── hf_cache/                 ← HuggingFace 缓存 (离线用)
#   ├── output/                   ← 训练输出
#   └── logs/                     ← 日志
#
# 使用方法:
#   sbatch scripts/run_all.sh
#
# 多 GPU 训练 (需修改 #SBATCH --gres=gpu:N):
#   sbatch scripts/run_all.sh
# ============================================================

set -e
mkdir -p logs output results

# ---- 作业基本信息 ----
echo "=========================================="
echo " Job ID    : $SLURM_JOB_ID"
echo " Node      : $SLURMD_NODENAME"
echo " Start     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"
echo "[INFO] Submit directory: $(pwd)"

# ============================================================
# 环境配置
# ============================================================

# 跳过 TensorFlow（我们只用 PyTorch，避免 Keras 版本冲突）
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_TORCHVISION=1

# 激活 conda 环境
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate base
echo "[INFO] Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"

# HuggingFace 离线缓存 — 指向上传的 hf_cache 目录
export HF_HOME="$SLURM_SUBMIT_DIR/hf_cache"
export TRANSFORMERS_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
export HF_DATASETS_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
export HF_HUB_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
# 如果集群有镜像
# export HF_ENDPOINT=https://hf-mirror.com
echo "[INFO] HF cache: $HF_HOME"

# 关闭 wandb 避免集群网络问题 (按需开启)
export WANDB_MODE=disabled
# export WANDB_PROJECT=sarcasm-r1

# 环境信息
echo ""
echo "---- Environment Info ----"
python --version
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "GPUs    : $(nvidia-smi -L | wc -l)"
echo "TRL     : $(python -c 'import trl; print(trl.__version__)' 2>/dev/null || echo 'not installed')"
echo "--------------------------"
echo ""

# ============================================================
# 配置参数 (按需修改)
# ============================================================

# 模型路径: 优先使用本地预下载的模型，回退到 HF 自动下载
MODEL_DIR="$SLURM_SUBMIT_DIR/models"
if [ -d "$MODEL_DIR/Qwen2.5-7B-Instruct" ]; then
    MODEL_PATH="$MODEL_DIR/Qwen2.5-7B-Instruct"
    echo "[INFO] Using local model: $MODEL_PATH"
elif [ -d "$MODEL_DIR/Qwen2.5-1.5B-Instruct" ]; then
    MODEL_PATH="$MODEL_DIR/Qwen2.5-1.5B-Instruct"
    echo "[INFO] Using local model: $MODEL_PATH"
else
    MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
    echo "[INFO] Using HuggingFace model: $MODEL_PATH (需联网下载)"
fi

DATA_DIR="$SLURM_SUBMIT_DIR/data"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/output/sarcasm-r1-v5"
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "[INFO] Model  : $MODEL_PATH"
echo "[INFO] Data   : $DATA_DIR"
echo "[INFO] Output : $OUTPUT_DIR"
echo "[INFO] GPUs   : $NUM_GPUS"
echo ""

# ---- 检查必要文件 ----
echo "[INFO] Checking required files..."
for f in code/grpo/train.py code/grpo/rewards.py code/grpo/prompts.py \
         code/grpo/config.yaml code/grpo/deepspeed_zero2.yaml \
         code/evaluation/inference.py code/evaluation/evaluate.py \
         data/download_data.py; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
    fi
done
echo ""

# ============================================================
# STAGE 1: 数据准备
# ============================================================
echo "##########################################################"
echo "# STAGE 1: Data Preparation      [$(date '+%H:%M:%S')]"
echo "##########################################################"
echo ""

cd "$SLURM_SUBMIT_DIR/data"

# 1a. 下载数据 (如果还没有)
if [ ! -d "raw/SemEval2018-Task3" ] && [ ! -d "raw/MUStARD" ]; then
    echo "[INFO] Downloading datasets..."
    python download_data.py --output_dir ./raw || {
        echo "[WARN] Download failed (集群可能无外网)."
        echo "       请在本地先运行: python data/download_data.py --output_dir ./raw"
        echo "       然后将 raw/ 目录上传到集群."
    }
else
    echo "[INFO] Raw data already exists, skipping download."
fi

# 1b. 处理数据
echo "[INFO] Processing datasets..."
cd "$SLURM_SUBMIT_DIR/code/data_processing"

if [ -d "$DATA_DIR/raw/SemEval2018-Task3" ]; then
    echo "[INFO] Processing SemEval 2018..."
    python process_semeval.py \
        --input_dir "$DATA_DIR/raw/SemEval2018-Task3" \
        --output_dir "$DATA_DIR/processed"
else
    echo "[WARN] SemEval raw data not found, skipping."
fi

if [ -d "$DATA_DIR/raw/MUStARD" ]; then
    echo "[INFO] Processing MUStARD..."
    python process_mustard.py \
        --input_dir "$DATA_DIR/raw/MUStARD" \
        --output_dir "$DATA_DIR/processed"
else
    echo "[WARN] MUStARD raw data not found, skipping."
fi

# 1c. 合并为 GRPO 训练格式
echo "[INFO] Formatting GRPO training data..."
python format_grpo_data.py \
    --input_dir "$DATA_DIR/processed" \
    --output_dir "$DATA_DIR/processed" \
    --combine

# 检查训练数据
TRAIN_DATA="$DATA_DIR/processed/train_combined.csv"
if [ ! -f "$TRAIN_DATA" ]; then
    # 回退: 尝试单独的数据集
    for f in "$DATA_DIR/processed/"train_*.csv; do
        if [ -f "$f" ]; then
            TRAIN_DATA="$f"
            break
        fi
    done
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "[ERROR] No training data found! Exiting."
    exit 1
fi

TRAIN_SAMPLES=$(python -c "import pandas as pd; print(len(pd.read_csv('$TRAIN_DATA')))")
echo ""
echo "[$(date '+%H:%M:%S')] Stage 1 (data prep) done."
echo "  Training data: $TRAIN_DATA ($TRAIN_SAMPLES samples)"
echo ""

# ============================================================
# STAGE 2: GRPO 训练
# ============================================================
echo "##########################################################"
echo "# STAGE 2: GRPO Training          [$(date '+%H:%M:%S')]"
echo "##########################################################"
echo ""

cd "$SLURM_SUBMIT_DIR/code/grpo"

# 设置数据路径环境变量
export SARCASM_DATA_PATH="$TRAIN_DATA"

if [ "$NUM_GPUS" -gt 1 ]; then
    # 多 GPU: 使用 DeepSpeed ZeRO-2
    echo "[INFO] Multi-GPU training with $NUM_GPUS GPUs + DeepSpeed ZeRO-2"
    accelerate launch \
        --config_file=deepspeed_zero2.yaml \
        --num_processes "$NUM_GPUS" \
        train.py \
        --config config.yaml \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --report_to none
else
    # 单 GPU: 所有参数由 config.yaml 控制，避免命令行覆盖
    echo "[INFO] Single-GPU training (config.yaml controlled)"
    python train.py \
        --config config.yaml \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR"
fi

echo ""
echo "[$(date '+%H:%M:%S')] Stage 2 (GRPO training) done."
echo "  Model saved: $(ls -d $OUTPUT_DIR 2>/dev/null || echo 'MISSING')"
echo ""

# ============================================================
# STAGE 3: 评估
# ============================================================
echo "##########################################################"
echo "# STAGE 3: Evaluation             [$(date '+%H:%M:%S')]"
echo "##########################################################"
echo ""

cd "$SLURM_SUBMIT_DIR/code/evaluation"

# 查找测试数据
for TEST_DATA in "$DATA_DIR/processed/semeval_test.csv" \
                 "$DATA_DIR/processed/semeval_validation.csv" \
                 "$DATA_DIR/processed/mustard_test.csv"; do
    if [ -f "$TEST_DATA" ]; then
        DATASET_NAME=$(basename "$TEST_DATA" .csv)
        RESULT_FILE="$SLURM_SUBMIT_DIR/results/${DATASET_NAME}_predictions.jsonl"

        echo "[INFO] Evaluating on: $TEST_DATA"

        # 推理
        python inference.py \
            --model "$OUTPUT_DIR" \
            --base_model "$MODEL_PATH" \
            --data "$TEST_DATA" \
            --output "$RESULT_FILE" \
            --batch_size 4

        # 评估
        python evaluate.py \
            --results "$RESULT_FILE" \
            --output "$SLURM_SUBMIT_DIR/results/${DATASET_NAME}_eval.json" \
            --name "Sarcasm-R1"

        echo ""
    fi
done

echo "[$(date '+%H:%M:%S')] Stage 3 (evaluation) done."

# ============================================================
# 汇总
# ============================================================
cd "$SLURM_SUBMIT_DIR"
echo ""
echo "=========================================="
echo " ALL DONE"
echo " End Time : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " Output files:"
ls -lh output/sarcasm-r1-v5/ 2>/dev/null || echo "  (model output not found)"
echo ""
echo " Evaluation results:"
for f in results/*_eval.json; do
    if [ -f "$f" ]; then
        echo "  --- $(basename $f) ---"
        python -c "
import json
with open('$f') as fp:
    m = json.load(fp)
print(f\"    Accuracy : {m['accuracy']:.4f}\")
print(f\"    Macro-F1 : {m['macro_f1']:.4f}\")
for label, cm in m.get('class_metrics', {}).items():
    print(f\"    {label}: P={cm['precision']:.4f} R={cm['recall']:.4f} F1={cm['f1']:.4f}\")
" 2>/dev/null || echo "  (parse error)"
    fi
done
echo "=========================================="
