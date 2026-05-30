# Sarcasm-R1 实验总结报告

> **项目**: Bridging Pragmatics and Focused Reasoning: A Hybrid Approach for Sarcasm Detection on SLMs
> **架构二**: 基于 GRPO 强化学习的多维度推理讽刺检测
> **作者**: Member B (CS310, SP 2026)

---

## 1. 项目概述

Sarcasm-R1 借鉴 DeepSeek-R1 的强化学习范式，使用 GRPO (Group Relative Policy Optimization) 训练 Qwen2.5-1.5B-Instruct 模型，使其具备结构化的三维度讽刺推理能力：

| 维度 | 目标 |
|------|------|
| **Rhetoric (修辞)** | 识别反语、夸张、含蓄、字面与隐含意思对比 |
| **Context (背景)** | 评估情境上下文与背景知识的匹配/冲突 |
| **Emotion (情绪)** | 检测字面情感与隐含情感之间的反差 |

### 技术栈

- **基础模型**: Qwen2.5-1.5B-Instruct (1.54B 参数)
- **训练框架**: TRL GRPOTrainer + PyTorch + DeepSpeed
- **训练数据**: SemEval 2018 Task 3 (tweet_eval/irony) + MUStARD，共 5,291 条
- **评估数据**: SemEval 2018 Test (784 条) + Validation (955 条)
- **集群环境**: SLURM, NVIDIA L40 GPU (46GB), Python 3.9

---

## 2. 训练配置

| 参数 | 值 |
|------|-----|
| 训练轮次 | 2 epochs (5,290 steps) |
| 每卡 batch size | 1 |
| 梯度累积步数 | 4 |
| 每样本生成数 | 2 |
| 学习率 | 3e-6, cosine schedule |
| 精度 | BF16 |
| 最大 prompt 长度 | 512 |
| 最大 completion 长度 | 512 |
| vLLM | 关闭 |
| 训练时长 | **18 小时 44 分钟** |

### 奖励函数

| 奖励函数 | 作用 | 分值范围 |
|----------|------|----------|
| `format_reasoning_reward` | 推理格式质量（长度、结构、{answer}格式） | -5 ~ +10 |
| `accuracy_reward_log` | 最终答案准确性 | +10 / -5 |

---

## 3. 训练动态分析

![Training Curves](Sarcasm-R1/results/training_curves.png)

### 关键观察

**奖励曲线**:
- 总奖励从 ~8.4 上升到 ~12.5 并稳定
- `format_reasoning_reward` 快速达到满分 10.0（约第 30 步后）
- `accuracy_reward_log` 始终较低（0~2.5 范围波动）

**策略熵 (Entropy)**:
- 从 1.216 下降到 ~0.5
- 表明模型输出趋于单一模式，丧失了多样性

**损失 (Loss)**:
- 训练中后期 loss 稳定在 0.0
- 结合低 entropy，说明模型策略已坍缩

### 训练过程诊断

| 阶段 | Steps | 总奖励 | 格式奖励 | 准确性奖励 | 熵 |
|------|-------|--------|----------|-----------|-----|
| 早期 | 0 | 8.4 | 7.0 | 1.4 | 1.216 |
| 中期 | 50 | 12.5 | 10.0 | 2.5 | 0.742 |
| 中后期 | 200 | 11.8 | 10.0 | 1.8 | 0.434 |
| 后期 | 528 | 12.5 | 10.0 | 2.5 | 0.498 |

模型快速学会了输出格式良好的推理链（format reward 满分），但未能学到有效的讽刺/非讽刺区分能力。

---

## 4. 评估结果

### 4.1 核心指标

| 数据集 | Accuracy | Macro-F1 | Sarcastic F1 | Non-sarcastic F1 |
|--------|----------|----------|-------------|------------------|
| SemEval Test | **0.3967** | **0.2840** | 0.5680 | 0.0000 |
| SemEval Validation | **0.4764** | **0.3227** | 0.6454 | 0.0000 |

### 4.2 混淆矩阵

![Confusion Matrices](Sarcasm-R1/results/confusion_matrices.png)

**SemEval Test (n=784)**:

| | 预测: Sarcastic | 预测: Non-sarcastic |
|---|---|---|
| **实际: Sarcastic (311)** | 311 (TP) | 0 (FN) |
| **实际: Non-sarcastic (473)** | 473 (FP) | 0 (TN) |

### 4.3 预测分布

![Prediction Distribution](Sarcasm-R1/results/prediction_distribution.png)

模型将 **100% 的样本** 预测为 "sarcastic"（784/784），完全丧失了区分两类的能力。

### 4.4 Per-class Metrics

![Class Metrics](Sarcasm-R1/results/class_metrics.png)

- Sarcastic 类: Precision=0.40, Recall=1.00 (全部预测为 sarcastic)
- Non-sarcastic 类: 所有指标为 0

---

## 5. 问题诊断与分析

### 5.1 模型坍缩 (Mode Collapse)

这是 GRPO 训练中典型的**奖励欺骗 (Reward Hacking)** 现象：

1. **格式奖励占主导**: `format_reasoning_reward` 满分为 10.0，而 `accuracy_reward` 的期望值仅 ~1.4（因为正负样本各半，随机答对的期望较低）
2. **生成多样性不足**: 每样本仅生成 2 个候选（`num_generations=2`），GRPO 难以产生足够对比来学习区分
3. **策略坍缩到安全解**: 输出 "sarcastic" 能稳定获得高格式奖励 + 部分准确性奖励，模型收敛到这个局部最优

### 5.2 其他影响因素

| 因素 | 影响 |
|------|------|
| 单卡限制 | batch_size=1, num_generations=2，GRPO group 太小 |
| Python 3.9 | 环境兼容性问题导致多次中断 |
| vLLM 关闭 | 生成速度慢，训练时间 18.7 小时 |
| 奖励函数权重 | format(10.0) 远大于 accuracy 期望(~1.4)，格式奖励压倒了准确性信号 |

---

## 6. 改进建议

### 短期改进（可立即尝试）

1. **调整奖励权重**: 使用 `combined_reward` 替代当前的分离奖励，增大 accuracy 权重
   ```yaml
   # accuracy: 1.0 → 2.0, format: 1.0 → 0.1
   ```
2. **增大 num_generations**: 从 2 提高到 4~8（需要更多 GPU 显存或使用 1.5B 模型）
3. **增加 accuracy_reward 惩罚**: 对错误预测加大惩罚（-5 → -10）
4. **在奖励中添加类别平衡惩罚**: 如果连续预测同一类别，降低奖励

### 中期改进（需要额外资源）

1. **使用 SFT 预训练**: 先用推理链数据做 SFT，再做 GRPO，类似 DeepSeek-R1 的训练流程
2. **使用更大模型**: Qwen2.5-7B-Instruct 可能更好地学到区分能力
3. **多 GPU 训练**: 启用 DeepSpeed ZeRO-2，增大 batch_size 和 num_generations
4. **使用 vLLM 加速**: 大幅减少生成时间，允许更多 epochs

### 长期方向

1. **在线 RL (Online RL)**: 使用 PPO 替代 GRPO，可能更好地处理稀疏奖励
2. **DPO 直接偏好优化**: 用推理链质量构建偏好对，训练更稳定
3. **多任务训练**: 同时训练讽刺检测和相关任务（情感分析、语用推理）

---

## 7. 项目文件结构

```
NLP-Project/
├── README.md                          # 本文件（总结报告）
├── Sarcasm-R1/
│   ├── plan.md                        # 设计规划文档
│   ├── step.md                        # 实现步骤与进度
│   ├── requirements.txt               # Python 依赖
│   │
│   ├── data/
│   │   ├── download_data.py           # 数据下载脚本
│   │   ├── raw/                       # 原始数据
│   │   │   ├── SemEval2018-Task3/     # SemEval 2018 Task 3
│   │   │   └── MUStARD/              # MUStARD 对话数据集
│   │   └── processed/                 # 处理后的训练数据
│   │       ├── semeval_train.csv      # 2,862 samples
│   │       ├── semeval_test.csv       # 784 samples
│   │       ├── semeval_validation.csv # 955 samples
│   │       ├── mustard_train.csv      # 690 samples
│   │       └── train_combined.csv     # 5,291 samples (GRPO 训练用)
│   │
│   ├── models/
│   │   └── Qwen2.5-1.5B-Instruct/    # 预下载的基础模型
│   │
│   ├── code/
│   │   ├── data_processing/
│   │   │   ├── process_semeval.py     # SemEval 数据处理
│   │   │   ├── process_mustard.py     # MUStARD 数据处理
│   │   │   ├── format_grpo_data.py    # GRPO 格式化
│   │   │   └── generate_reasoning.py  # 推理链生成
│   │   ├── grpo/
│   │   │   ├── train.py              # GRPO 训练主脚本
│   │   │   ├── config.yaml           # 训练配置
│   │   │   ├── rewards.py            # 奖励函数
│   │   │   ├── prompts.py            # Prompt 模板
│   │   │   └── deepspeed_zero2.yaml  # DeepSpeed 配置
│   │   ├── reward_model/
│   │   │   └── generative_verifier.py # 生成式验证器
│   │   └── evaluation/
│   │       ├── inference.py          # 推理脚本
│   │       └── evaluate.py           # 评估脚本
│   │
│   ├── scripts/
│   │   ├── run_all.sh               # 完整流程 sbatch 脚本
│   │   ├── sbatch_train.sh          # 仅训练
│   │   ├── sbatch_eval.sh           # 仅评估
│   │   ├── run_train.sh             # 训练入口
│   │   ├── run_eval.sh              # 评估入口
│   │   ├── setup.sh                 # 环境搭建
│   │   └── prepare_offline.py       # 离线预下载 (Windows 兼容)
│   │
│   ├── results/                      # 评估结果与可视化
│   │   ├── semeval_test_eval.json
│   │   ├── semeval_test_predictions.jsonl
│   │   ├── semeval_validation_eval.json
│   │   ├── semeval_validation_predictions.jsonl
│   │   ├── training_metrics.json
│   │   ├── plot_results.py           # 可视化生成脚本
│   │   ├── training_curves.png
│   │   ├── confusion_matrices.png
│   │   ├── prediction_distribution.png
│   │   └── class_metrics.png
│   │
│   ├── output/
│   │   └── sarcasm-r1/              # 训练好的模型权重 (集群上)
│   │
│   └── logs/                         # 训练/评估日志
│       ├── sarcasm_r1_86629.out      # 训练日志 (6.9MB)
│       └── sr1_eval_86710.out        # 评估日志
```

---

## 8. 队友复用指南

### 8.1 使用训练好的模型

训练好的模型在集群的 `output/sarcasm-r1/` 目录下。直接加载推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/output/sarcasm-r1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# 推理
messages = [
    {"role": "system", "content": "你是讽刺检测专家..."},
    {"role": "user", "content": "请判断以下文本是否含有讽刺: \"Oh great, another meeting...\""},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(output[0]))
```

### 8.2 使用处理好的数据

处理好的数据在 `data/processed/` 下，格式为 CSV：

| 文件 | 用途 | 样本数 |
|------|------|--------|
| `semeval_train.csv` | 训练 | 2,862 |
| `semeval_test.csv` | 测试 | 784 |
| `semeval_validation.csv` | 验证 | 955 |
| `mustard_train.csv` | 训练 | 690 |
| `train_combined.csv` | GRPO 训练 (合并) | 5,291 |

每列含义：`question` (输入 prompt), `answer` (sarcastic/non_sarcastic), `source_text`, `source_label`

### 8.3 重新训练（修改超参数）

```bash
# 1. 上传到集群
rsync -avz Sarcasm-R1/ <user>@<host>:~/Sarcasm-R1/

# 2. 修改配置
vim code/grpo/config.yaml   # 修改 num_generations, learning_rate 等

# 3. 提交训练
sbatch scripts/sbatch_train.sh

# 4. 单独评估
sbatch scripts/sbatch_eval.sh
```

### 8.4 集群环境备忘

集群 Python 3.9，已安装关键依赖。首次使用需注意：
- 设置 `export TRANSFORMERS_NO_TF=1` 避免 TensorFlow 冲突
- 上传前先在本地运行 `python scripts/prepare_offline.py` 下载模型和数据
- 所有 `.py` 文件已加 `from __future__ import annotations` 兼容 Python 3.9

---

## 9. 参考文献

1. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
2. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
3. Lee et al., "Pragmatic Metacognitive Prompting Improves LLM Performance on Sarcasm Detection" (EMNLP 2025)
4. Zhang et al., "Generative Verifiers: Reward Modeling as Next-Token Prediction" (ICLR 2025)
5. Van Hee et al., "SemEval-2018 Task 3: Irony Detection in English Tweets"
6. Castro et al., "Towards Multimodal Sarcasm Detection" (MUStARD)
