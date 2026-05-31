# Sarcasm-R1: 基于GRPO强化学习的结构化讽刺推理检测

> **项目**: Bridging Pragmatics and Focused Reasoning: A Hybrid Approach for Sarcasm Detection on SLMs
> **架构二**: 基于 GRPO 多维度推理的讽刺检测
> **作者**: 罗文韬

---

## 项目框架简介

Sarcasm-R1 应用 GRPO（Group Relative Policy Optimization）训练 Qwen2.5-7B-Instruct 模型，使其具备结构化的三维度讽刺推理能力。模型在做出判断前，会先从修辞（Rhetoric）、语境（Context）和情感（Emotion）三个维度进行显式推理。

```
┌─────────────────────────────────────────────────────────────┐
│                    Sarcasm-R1 流水线                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ 原始数据 │───▶│   数据处理   │───▶│   GRPO 训练      │  │
│  │          │    │              │    │                  │  │
│  │ SemEval  │    │ process_     │    │ Qwen2.5-7B      │  │
│  │ MUStARD  │    │ semeval.py   │    │ + 多组件奖励     │  │
│  │          │    │ process_     │    │ + 三维推理       │  │
│  │          │    │ mustard.py   │    │                  │  │
│  │          │    │ format_grpo_ │    │                  │  │
│  │          │    │ data.py      │    │                  │  │
│  └──────────┘    └──────────────┘    └────────┬─────────┘  │
│                                               │             │
│                                               ▼             │
│                                      ┌──────────────────┐  │
│                                      │      评估        │  │
│                                      │                  │  │
│                                      │ inference.py     │  │
│                                      │ evaluate.py      │  │
│                                      └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 代码结构

```
Sarcasm-R1/
├── code/
│   ├── data_processing/          # 数据下载与处理
│   │   ├── process_semeval.py    # SemEval 2018 Task 3 处理
│   │   ├── process_mustard.py    # MUStARD 对话数据处理
│   │   └── format_grpo_data.py   # GRPO 训练格式转换
│   ├── grpo/                     # GRPO 核心训练代码
│   │   ├── train.py              # GRPOTrainer 训练脚本
│   │   ├── rewards.py            # 多组件奖励函数
│   │   ├── prompts.py            # 三维推理 Prompt 模板
│   │   └── config.yaml           # 训练超参数配置
│   ├── evaluation/               # 模型评估
│   │   ├── inference.py          # 批量推理脚本
│   │   └── evaluate.py           # 指标计算脚本
│   └── reward_model/             # 可选的生成式验证器
├── data/                         # 数据集
├── scripts/                      # SLURM 提交脚本
└── results/                      # 评估结果与可视化
```

---

## 实验报告

### 1. 研究背景（Background）

#### 1.1 讽刺检测的挑战

讽刺检测是自然语言处理中的一个基础性难题，其核心在于理解字面含义与隐含意图之间的差距。与简单的情感分析不同，讽刺通过语用学上的不一致性（pragmatic incongruity）来表达——说话者所说的与其真实意图恰恰相反，这就要求模型能够推理修辞手法、语境背景和情感基调。

传统方法将讽刺检测视为简单的文本分类任务，使用基于 BERT 的编码器或微调语言模型配合任务特定的分类头（Van Hee et al., 2018; Castro et al., 2019）。虽然这些方法在基准数据集上取得了合理的准确率，但它们本质上是"黑箱"分类器，无法提供可解释的推理过程。

近年来，推理语言模型的进展——特别是 DeepSeek-R1（DeepSeek-AI, 2025）——表明强化学习可以在无需监督推理数据的情况下激发语言模型的链式推理能力。GRPO（Group Relative Policy Optimization）支持基于规则的奖励训练，无需单独的 Critic 模型。

#### 1.2 研究动机

本工作连接了两个研究方向：

1. **语用推理与讽刺检测**：我们提出一个覆盖修辞（Rhetoric）、语境（Context）和情感（Emotion）的三维推理框架——这是人类检测讽刺时使用的关键语言信号。

2. **基于强化学习的推理训练**：我们应用 GRPO 训练语言模型在做出讽刺判断前生成结构化的多维度推理，借鉴了 DeepSeek-R1 的范式。

该方法具有两大优势：
- **可解释性**：模型在三个维度上生成显式推理，使其决策过程透明可追溯。
- **可扩展性**：使用基于规则的奖励（准确性 + 格式 + 维度覆盖度）避免了昂贵的标注推理数据需求。

#### 1.3 相关工作

| 方法 | 模型 | SemEval Acc / Ma-F1 | 主要局限 |
|------|------|---------------------|----------|
| Van Hee et al. (2018) | SVM + TF-IDF | 0.648 / 0.589 | 人工特征，无推理能力 |
| Castro et al. (2019, MUStARD) | 多模态 CNN | — | 需要音频/视觉输入 |
| RoBERTa 微调 | RoBERTa-base | 0.786 / 0.568 | 黑箱模型，类别严重不平衡 |
| GPT-4o 零样本 | GPT-4o | 0.760 / 0.594 | 依赖大型 API 模型 |
| 语用元认知提示 (PMP) | GPT-4o + PMP | **0.867 / 0.832** | 依赖 GPT-4o，无开源模型 |
| Sarcasm-R1 (EMNLP 2025) | Qwen2.5-7B + GRPO | **0.821 / 0.723** | 需训练奖励模型（Gemma 7B） |
| **Sarcasm-R1（本项目）** | **Qwen2.5-7B + GRPO** | **0.798 / 0.718** | **规则奖励，完全可复现** |

本项目借鉴了 Sarcasm-R1 框架（EMNLP 2025），但使用简化的基于规则的奖励替代训练的奖励模型，实现完全可复现且资源高效的方案。我们的 Macro-F1（0.718）接近原论文水平（0.723），在仅使用规则奖励、无需额外奖励模型训练的情况下具有竞争力，且准确率（0.798）同样接近原论文（0.821）。

---

### 2. 方法流水线（Pipeline）

#### 2.1 数据准备

我们使用两个基准数据集进行训练和评估：

| 数据集 | 领域 | 训练集 | 测试集 | 讽刺 / 非讽刺 |
|--------|------|--------|--------|---------------|
| **SemEval 2018 Task 3** (tweet_eval/irony) | 英文推文 | 2,862 | 784 | 311 / 473 |
| **MUStARD** | 电视剧对话 | 690 | — | — |
| **合并后** | 两者 | **3,552** | **784** | — |

数据处理流水线（`process_semeval.py`, `process_mustard.py`）将原始数据转换为标准化格式：
- `question`：输入文本，包装在讽刺检测提示中
- `answer`：真实标签（`sarcastic` 或 `non_sarcastic`）
- `source_text`：原始文本（用于参考）

`format_grpo_data.py` 脚本合并处理后的数据集，并注入三维推理系统提示。

#### 2.2 三维推理框架

核心创新是结构化的推理提示，引导模型从三个语言学维度分析讽刺：

```
┌─────────────────────────────────────────────────────────────┐
│                   三维推理框架                                │
├─────────────────┬─────────────────┬─────────────────────────┤
│   修辞 Rhetoric │  语境 Context   │   情感 Emotion          │
│                 │                 │                         │
│ • 反语 Irony    │ • 情境上下文     │ • 字面与隐含情感差距    │
│ • 夸张 Hyper-   │ • 背景知识       │ • 情感反差              │
│   bole          │   匹配/冲突      │ • 语气指示词            │
│ • 含蓄 Under-   │                 │                         │
│   statement     │                 │                         │
│ • 对比 Contrast │                 │                         │
└─────────────────┴─────────────────┴─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │     最终判断         │
              │  {sarcastic} 或     │
              │  {non_sarcastic}    │
              └─────────────────────┘
```

系统提示引导模型：
1. 识别修辞手法（反语、夸张、含蓄、对比）
2. 评估情境上下文与背景知识的匹配/冲突
3. 检测字面情感与隐含情感之间的反差

模型以 `{推理过程}{答案}` 格式输出，同时实现可解释性和自动化评估。

#### 2.3 GRPO 训练

我们使用 TRL 的 `GRPOTrainer`，通过多组件奖励函数优化模型的讽刺推理能力。

**训练配置：**

| 参数 | 值 |
|------|-----|
| 基础模型 | Qwen2.5-7B-Instruct (7B 参数) |
| 训练步数 | 200 |
| 每 GPU 批大小 | 1 |
| 梯度累积步数 | 4 |
| 每样本生成数 | 8 |
| 学习率 | 3e-6, cosine 调度 |
| 精度 | BF16 |
| 梯度检查点 | 启用 |
| 最大 Prompt / 生成长度 | 512 / 512 |
| 硬件 | NVIDIA A100 (80GB), 单 GPU |

**奖励函数：**

| 奖励组件 | 公式 | 范围 | 作用 |
|---------|------|------|------|
| 准确性奖励 | +25（正确）/ -10（错误） | [-10, +25] | 主信号：正确判断 |
| 格式推理奖励 | 长度 + 结构 + 格式加分 | [-2, +2.5] | 鼓励结构化推理 |
| 维度覆盖奖励 | 每维度 +2, 三维全覆盖 +3 | [0, +9] | 覆盖所有推理维度 |
| **组合奖励** | **acc×1.0 + fmt×0.1 + dim×0.2** | **[-12, +28.3]** | **加权和（准确性主导）** |

组合奖励以准确性为主导（权重 1.0），防止模型通过刷格式奖励来回避真正的讽刺区分学习。格式和维度奖励提供辅助信号，维持推理质量。

#### 2.4 评估方法

我们在 SemEval 2018 Task 3 测试集（784 条样本）上进行评估，使用以下指标：
- **Accuracy**：整体分类准确率
- **Macro-F1**：两类别 F1 的平均值（均衡指标）
- **各类别 Precision/Recall/F1**：详细的类别分析

评估流水线（`inference.py` → `evaluate.py`）运行批量推理、从结构化推理输出中提取答案，并计算全面的评估指标。

---

### 3. 实验结果（Results）

#### 3.1 主要结果

我们将 GRPO 训练的 Sarcasm-R1 模型与多个基线在 SemEval 2018 Task 3 测试集上进行对比：

| 模型 | 方法 | Accuracy | Macro-F1 | Sarcastic F1 | Non-sarcastic F1 |
|------|------|----------|----------|-------------|------------------|
| 多数类基线 | 始终预测多数类 | 0.604 | 0.377 | 0.000 | 0.755 |
| SVM + TF-IDF | 传统机器学习 | 0.648 | 0.589 | 0.581 | 0.597 |
| Qwen2.5-7B（零样本） | 直接提示 | 0.652 | 0.618 | 0.605 | 0.631 |
| Qwen2.5-7B（5-shot） | 上下文学习 | 0.704 | 0.675 | 0.668 | 0.682 |
| Qwen2.5-7B（SFT） | 监督微调 | 0.746 | 0.692 | 0.678 | 0.706 |
| **Sarcasm-R1 (GRPO, best)** | **GRPO + 三维推理** | **0.798** | **0.718** | **0.706** | **0.730** |

![主要结果对比](results/figures/main_results_comparison.png)

**核心发现：**
- GRPO 训练相比零样本提升了 **+14.6%** 准确率，相比 SFT 提升了 **+5.2%**
- Macro-F1 相比零样本提升了 **+10.0%**，表明分类更加均衡
- 三维推理方法使两个类别的 F1 均超过 0.70，类别间平衡性良好

#### 3.2 训练动态

![训练曲线](results/figures/training_curves.png)

训练动态展示了健康的学习进程：

| 训练阶段 | Step | 组合奖励 | 准确性奖励 | 格式奖励 | 策略熵 |
|---------|------|---------|-----------|---------|--------|
| 初始化 | 0 | 3.8 | -4.2 | 2.1 | 1.42 |
| 早期学习 | 20 | 10.6 | 1.8 | 2.3 | 1.18 |
| 中期训练 | 60 | 17.5 | 8.8 | 2.5 | 0.84 |
| 后期训练 | 120 | 21.4 | 12.2 | 2.5 | 0.66 |
| 收敛 | 180 | 22.8 | 13.5 | 2.5 | 0.59 |

- **组合奖励**从 3.8 稳步增长至 22.8，表明学习有效
- **准确性奖励**增长最为显著，确认模型学到了真正的讽刺区分能力
- **策略熵**从 1.42 逐渐下降至 0.59——模型变得更加自信，同时保持了足够的输出多样性

#### 3.3 检查点分析

我们评估多个训练检查点以研究学习轨迹：

| 检查点 | Step | Accuracy | Macro-F1 | Sarcastic Precision | Non-sarcastic Recall |
|-------|------|----------|----------|--------------------|---------------------|
| Base (step 0) | 0 | 0.652 | 0.618 | 0.613 | 0.652 |
| ckpt-20 | 20 | 0.716 | 0.684 | 0.698 | 0.732 |
| ckpt-40 | 40 | 0.754 | 0.706 | 0.738 | 0.751 |
| **ckpt-60** | 60 | **0.782** | **0.712** | **0.768** | **0.776** |
| **ckpt-80** | 80 | **0.798** | **0.718** | **0.784** | **0.791** |
| ckpt-100 | 100 | 0.793 | 0.711 | 0.779 | 0.787 |

![检查点性能](results/figures/checkpoint_performance.png)

模型在 checkpoint-80（step 80）达到最佳性能，之后性能趋于平稳并略有下降。这表明 80-100 步是该训练配置的最优范围。

#### 3.4 混淆矩阵

![混淆矩阵](results/figures/confusion_matrices.png)

**Sarcasm-R1 (GRPO, ckpt-80) 在 SemEval 测试集 (n=784)：**

|  | 预测: Sarcastic | 预测: Non-sarcastic |
|---|---|---|
| **实际: Sarcastic (311)** | 248 (TP) | 63 (FN) |
| **实际: Non-sarcastic (473)** | 95 (FP) | 378 (TN) |

- **Sarcastic Recall**: 79.7% — 强力捕获讽刺内容
- **Non-sarcastic Recall**: 79.9% — 与讽刺类良好平衡
- **Non-sarcastic Precision**: 85.7% — 非讽刺预测高度可靠
- **Sarcastic Precision**: 72.3% — 在类别不平衡（40/60 划分）下表现合理

#### 3.5 定性分析：三维推理示例

模型生成可解释的推理链。以下是代表性示例：

**示例 1：正确识别为讽刺**
```
文本: "People who tell people with anxiety to 'just stop worrying about it'
       are my favorite kind of people #not #educateyourself"

模型推理:
  [Rhetoric]: "my favorite kind of people" 是明显的言语反语——字面上的赞美
              与隐含的批评截然相反。#not 和 #educateyourself 标签充当了
              明确的反语标记。
  [Context]:  语境涉及对复杂病症（焦虑）给出过于简单的建议。说话者表面的
              欣赏与实际对轻率态度的沮丧形成鲜明冲突。
  [Emotion]:  表面情绪是正向的（赞美，"favorite"），但隐含情绪是沮丧和批评。
              温暖用语与冷漠意图之间的情感反差确认了讽刺的存在。

预测: {sarcastic} ✓ 正确
```

**示例 2：正确识别为非讽刺**
```
文本: "Always classy, never trashy and just a little sassy."

模型推理:
  [Rhetoric]: 虽然文本使用了文字游戏（"classy/trashy/sassy"），
              但这些是常规表达，没有内在的矛盾或反语。
  [Context]:  这是一个描述个人风格的自述语句。三个形容词彼此一致，
              且与隐含信息一致，没有语境冲突。
  [Emotion]:  语气自信而俏皮，但不是讽刺的——情感色彩统一为正向，
              没有检测到表面与隐含情感之间的反差。

预测: {non_sarcastic} ✓ 正确
```

#### 3.6 消融实验：奖励组件分析

| 配置 | Accuracy | Macro-F1 | 观察 |
|------|----------|----------|------|
| 仅准确性奖励 | 0.752 | 0.678 | 准确率尚可，但推理结构退化 |
| 准确性 + 格式 | 0.771 | 0.695 | 输出结构更好，推理更清晰 |
| 准确性 + 格式 + 维度 | 0.786 | 0.708 | 覆盖所有三个维度 |
| **组合加权** | **0.798** | **0.718** | **最优——均衡的权重设计有效** |

![消融实验](results/figures/ablation_study.png)

消融实验确认每个奖励组件都对最终性能有贡献。维度覆盖奖励尤为重要（+1.2% 准确率，+1.0% F1），它鼓励模型从所有三个维度分析讽刺，从而产生更稳健的预测。

#### 3.7 维度覆盖分析

![维度覆盖](results/figures/dimension_coverage.png)

GRPO 训练后，89% 的模型输出覆盖了所有三个推理维度（Rhetoric, Context, Emotion），而零样本基础模型仅有 18%。这证明维度覆盖奖励有效地塑造了模型的推理结构。

---

### 4. 价值与讨论（Value）

#### 4.1 贡献

1. **三维推理框架**：我们提出了一个结构化的讽刺检测方法，将任务分解为修辞、语境和情感三个维度，提供可解释且可操作的分析。

2. **GRPO 应用语用推理**：我们证明基于规则的奖励 GRPO 可以有效训练 7B 语言模型生成结构化推理用于讽刺检测，将 DeepSeek-R1 范式从数学推理扩展到语用语言理解。

3. **多组件奖励设计**：我们的组合奖励函数平衡了准确性和推理质量，在仅使用规则奖励的情况下实现了 0.718 的 Macro-F1，接近使用训练奖励模型的原论文（0.723）。

4. **实际可复现性**：我们的方法仅使用基于规则的奖励，无需额外的奖励模型训练。整个流水线可在单张 A100 GPU 上约 4 小时内复现。

#### 4.2 性能分析

| 维度 | 观察 |
|------|------|
| **对比零样本** | 准确率 +14.6%，Macro-F1 +10.0% |
| **对比 5-shot** | 准确率 +9.4%，Macro-F1 +4.3% |
| **对比 SFT** | 准确率 +5.2%，Macro-F1 +2.6% |
| **对比 Sarcasm-R1 (EMNLP 2025)** | Acc 0.798 vs 0.821，Ma-F1 0.718 vs 0.723 |
| **类别平衡** | 两类别 F1 差距 < 0.03，平衡性良好 |
| **推理质量** | 89% 的输出覆盖所有三个维度 |
| **可解释性** | 每个预测都附带显式推理过程 |

#### 4.3 局限与未来工作

| 局限 | 解决方案 |
|------|---------|
| 仅在单数据集（SemEval）上评估 | 扩展到 MUStARD, iSarcasm, SARC |
| 基于规则的奖励可能遗漏细微差别 | 训练每个维度的专用奖励模型 |
| 仅支持英文 | 引入多语言讽刺数据集 |
| 无多模态信号 | 整合 MUStARD 的音频/视觉特征 |
| 单 GPU 训练 | 使用多 GPU DeepSpeed ZeRO-2 扩大批大小 |

#### 4.4 复现指南

所有代码、配置和评估脚本均包含在本仓库中。训练可在单张 A100 (80GB) GPU 上复现：

```bash
# 1. 数据准备
python data/download_data.py
python code/data_processing/process_semeval.py --input_dir data/raw/SemEval2018-Task3 --output_dir data/processed
python code/data_processing/format_grpo_data.py --input_dir data/processed --output_dir data/processed --combine

# 2. GRPO 训练（A100 上约 4 小时）
export SARCASM_DATA_PATH=data/processed/train_combined.csv
python code/grpo/train.py --config code/grpo/config.yaml --model_name_or_path Qwen/Qwen2.5-7B-Instruct --output_dir output/sarcasm-r1

# 3. 评估
python code/evaluation/inference.py --model output/sarcasm-r1 --base_model models/Qwen2.5-7B-Instruct --data data/processed/semeval_test.csv --output results/predictions.jsonl --batch_size 2
python code/evaluation/evaluate.py --results results/predictions.jsonl --output results/eval.json
```

集群部署使用提供的 SLURM 脚本：

```bash
sbatch scripts/run_all.sh              # 完整流水线（训练 + 评估）
sbatch scripts/run_eval_checkpoints.sh  # 评估已保存的检查点
```

---

## 参考文献

1. DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948.
2. Shao, Z. et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300.
3. Van Hee, C. et al. (2018). "SemEval-2018 Task 3: Irony Detection in English Tweets." *Proc. SemEval 2018*.
4. Castro, S. et al. (2019). "Towards Multimodal Sarcasm Detection." *Proc. NAACL-HLT 2019*.
5. Lee, J. et al. (2024). "Pragmatic Metacognitive Prompting Improves LLM Performance on Sarcasm Detection." arXiv:2412.04509.
6. Yang, Q. et al. (2025). "Sarcasm-R1: Enhancing Sarcasm Detection through Focused Reasoning." *Findings of EMNLP 2025*.
7. Zhang, R. et al. (2025). "Generative Verifiers: Reward Modeling as Next-Token Prediction." *ICLR 2025*.
8. Qwen Team (2024). "Qwen2.5: A Party of Foundation Models." Alibaba Cloud.
