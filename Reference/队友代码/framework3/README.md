# Framework3: Hybrid PMP-RCE Sarcasm Detection with GRPO

This directory contains the final hybrid sarcasm detection framework for the CS310 project.

Framework3 is built from two earlier ideas:

```text
Framework1: Pragmatic Metacognitive Prompting (PMP)
Framework2: Rhetoric / Context / Emotion (RCE) structured reasoning + GRPO training
Framework3: Hybrid PMP-RCE reasoning + GRPO policy optimization
```

The main goal is not to replace the base model, but to improve how the same LLM reasons about sarcasm and then use GRPO to reinforce the desired reasoning behavior.

All controlled experiments use:

```text
Qwen3-14B
```

---

## Method Overview

| Method | Model | Reasoning Design |
|---|---|---|
| Baseline | Qwen3-14B | Direct sarcasm classification |
| Framework1 | Qwen3-14B | PMP pragmatic reasoning |
| Framework2 | Qwen3-14B or GRPO-tuned Qwen3-14B | RCE structured reasoning |
| Framework3 | GRPO-tuned Qwen3-14B | Hybrid PMP + RCE reasoning optimized by hybrid rewards |

The four methods are evaluated on two test sets:

```text
SemEval-200
SemEval-784
```

This gives eight evaluation groups:

```text
2 datasets x 4 methods = 8 experiments
```

---

## Core Idea

Sarcasm usually depends on a gap between literal wording and intended meaning. A good detector should not only look for sentiment words. It should ask:

- what is literally said;
- what is implied beyond the literal wording;
- what the speaker intends;
- whether there is pretense or pragmatic incongruity;
- whether rhetoric, context, or emotion supports a sarcastic reading.

Framework1 and Framework2 cover different parts of this reasoning.

### Framework1: PMP

Framework1 uses Pragmatic Metacognitive Prompting. The improved prompt follows two phases:

```text
Phase 1: Preliminary pragmatic analysis
  - implicature
  - presupposition
  - speaker intent
  - polarity
  - pretense
  - literal-vs-implied meaning gap

Phase 2: Reflection and calibration
  - evidence for sarcasm
  - evidence against sarcasm
  - avoid over-predicting sarcasm from negative emotion alone
```

Prompt file:

```text
framework3/prompts/teacher_full_pmp_prompt.md
```

### Framework2: RCE + GRPO

Framework2 uses structured sarcasm reasoning inspired by Sarcasm-R1. It decomposes evidence into:

| Dimension | Meaning |
|---|---|
| Rhetoric | irony, exaggeration, understatement, contrast, mock praise, hashtags |
| Context | whether the literal statement fits common sense or the implied situation |
| Emotion | mismatch between surface emotion and implied emotion |

Prompt file:

```text
framework3/prompts/pmp_r1_system_prompt.md
```

GRPO training code:

```text
framework2/Sarcasm-R1/code/grpo/train.py
framework2/Sarcasm-R1/code/grpo/rewards.py
```

Convenience script for Qwen3-14B:

```text
framework3/scripts/train_framework2_grpo_qwen3.sh
```

### Framework3: Hybrid PMP-RCE + GRPO

Framework3 combines the two reasoning procedures at the method level and then applies GRPO training to optimize the policy toward this hybrid reasoning behavior:

```text
PMP pragmatic reasoning
        +
RCE structured reasoning
        =
Hybrid PMP-RCE sarcasm detection + GRPO optimization
```

It is not a majority vote or weighted ensemble. The model performs both reasoning processes inside one decision procedure, and GRPO rewards outputs that are correct, well formatted, and cover both PMP and RCE evidence.

Hybrid prompt file:

```text
framework3/prompts/framework3_hybrid_prompt.md
```

Hybrid GRPO training code:

```text
framework3/scripts/train_framework3_grpo.py
framework3/configs/framework3_grpo_config.yaml
```

The hybrid GRPO reward combines:

```text
classification correctness
+ output format stability
+ PMP coverage
+ RCE coverage
```

---

## Directory Structure

```text
framework3/
├── README.md
├── requirements-inference.txt
├── requirements-training.txt
├── configs/
│   ├── deepspeed_zero2.yaml
│   └── framework3_grpo_config.yaml
├── prompts/
│   ├── baseline_prompt.md
│   ├── framework3_hybrid_prompt.md
│   ├── pmp_r1_system_prompt.md
│   └── teacher_full_pmp_prompt.md
├── scripts/
│   ├── evaluate_predictions.py
│   ├── run_llm_predictions.py
│   ├── run_baseline_experiments.sh
│   ├── run_framework1_experiments.sh
│   ├── run_framework3_prompt_experiments.sh
│   ├── run_eight_experiments.sh
│   ├── run_autodl_pipeline.sh
│   ├── train_framework2_grpo_qwen3.sh
│   ├── train_framework3_grpo.py
│   └── train_framework3_grpo.sh
└── results/
    ├── project_report.tex
    ├── project_presentation.tex
    └── final_experiment_tables.tex
```

Generated outputs are saved under directories such as:

```text
framework3/predictions/
framework3/results/
framework3/logs/
framework3/results_baseline/
framework3/results_pmp_v2/
```

---

## Data

The project uses the existing SemEval data files:

| Dataset | File |
|---|---|
| SemEval-200 | `framework1/semeval200.tsv` |
| SemEval-784 | `framework2/Sarcasm-R1/data/processed/semeval_test.csv` |
| GRPO training data | `framework2/Sarcasm-R1/data/processed/semeval_train.csv` |

---

## Installation

For inference only:

```bash
python3 -m pip install -U -r framework3/requirements-inference.txt
```

For GRPO training:

```bash
python3 -m pip install -U -r framework3/requirements-training.txt
```

If the network is slow:

```bash
python3 -m pip install -U -r framework3/requirements-training.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
```

The model path should be exported before running experiments:

```bash
export QWEN_MODEL=/path/to/Qwen3-14B
```

Example:

```bash
export QWEN_MODEL=/root/autodl-tmp/project/models/Qwen3-14B
```

---

## Fast Prompt-Only Runs

These scripts do not train GRPO models. They are useful for quick ablations and debugging, while the full Framework3 method includes GRPO training.

### Baseline

```bash
cd /root/autodl-tmp/project

export QWEN_MODEL=/root/autodl-tmp/project/models/Qwen3-14B
export DATASET=all
export MAX_NEW_TOKENS=32
export FORCE=1

bash framework3/scripts/run_baseline_experiments.sh
```

Outputs:

```text
framework3/predictions/<dataset>_baseline.jsonl
framework3/results/<dataset>_baseline_eval.json
```

### Framework1: PMP

```bash
cd /root/autodl-tmp/project

export QWEN_MODEL=/root/autodl-tmp/project/models/Qwen3-14B
export DATASET=all
export MAX_NEW_TOKENS=160
export FORCE=1

bash framework3/scripts/run_framework1_experiments.sh
```

Outputs:

```text
framework3/predictions/<dataset>_framework1.jsonl
framework3/results/<dataset>_framework1_eval.json
```

### Framework3 Ablation: Prompt-Only Hybrid PMP-RCE

```bash
cd /root/autodl-tmp/project

export QWEN_MODEL=/root/autodl-tmp/project/models/Qwen3-14B
export DATASET=all
export MAX_NEW_TOKENS=192
export FORCE=1

bash framework3/scripts/run_framework3_prompt_experiments.sh
```

Outputs:

```text
framework3/predictions/<dataset>_framework3.jsonl
framework3/results/<dataset>_framework3_eval.json
```

---

## GRPO Training Runs

GRPO is used when we want Framework2 or Framework3 to learn the desired reasoning format instead of only following prompts.

### Train Framework2 GRPO

```bash
cd /root/autodl-tmp/project

export QWEN_MODEL=/root/autodl-tmp/project/models/Qwen3-14B
bash framework3/scripts/train_framework2_grpo_qwen3.sh
```

Default checkpoint:

```text
framework2/Sarcasm-R1/output/sarcasm-r1
```

### Train Framework3 Hybrid GRPO

```bash
cd /root/autodl-tmp/project

export QWEN_MODEL=/root/autodl-tmp/project/models/Qwen3-14B
bash framework3/scripts/train_framework3_grpo.sh
```

Default checkpoint:

```text
framework3/checkpoints/framework3-hybrid-grpo-qwen3-14b
```

### Full GRPO Pipeline

The full pipeline trains Framework2, trains Framework3, and then runs all eight experiments:

```bash
cd /root/autodl-tmp/project

export QWEN_MODEL=/root/autodl-tmp/project/models/Qwen3-14B
export TRAIN_FRAMEWORK2=1
export TRAIN_FRAMEWORK3=1
export DATASET=all
export FORCE=1

bash framework3/scripts/run_autodl_pipeline.sh
```

If checkpoints already exist, skip training:

```bash
export TRAIN_FRAMEWORK2=0
export TRAIN_FRAMEWORK3=0
export FRAMEWORK2_MODEL=framework2/Sarcasm-R1/output/sarcasm-r1
export FRAMEWORK3_MODEL=framework3/checkpoints/framework3-hybrid-grpo-qwen3-14b

bash framework3/scripts/run_eight_experiments.sh
```

---

## Final Report Results

The final report currently uses:

| Method | Result Source |
|---|---|
| Baseline | `framework3/results_baseline/` |
| Framework1 | `framework3/results_pmp_v2/` |
| Framework2 | `framework3/results/` |
| Framework3 | `framework3/results/` |

Main results:

| Dataset | Method | Accuracy | Macro-F1 | Sarc. F1 | Non-Sarc. F1 |
|---|---:|---:|---:|---:|---:|
| SemEval-200 | Baseline | 0.6100 | 0.5954 | 0.6723 | 0.5185 |
| SemEval-200 | Framework1 (PMP) | 0.7400 | 0.7391 | 0.7547 | 0.7234 |
| SemEval-200 | Framework2 (RCE) | 0.6900 | 0.6862 | 0.7207 | 0.6517 |
| SemEval-200 | Framework3 (PMP+RCE) | **0.7500** | **0.7496** | **0.7596** | **0.7396** |
| SemEval-784 | Baseline | 0.5791 | 0.5598 | 0.6519 | 0.4677 |
| SemEval-784 | Framework1 (PMP) | 0.6901 | 0.6869 | **0.7184** | 0.6553 |
| SemEval-784 | Framework2 (RCE) | 0.6531 | 0.6464 | 0.6951 | 0.5976 |
| SemEval-784 | Framework3 (PMP+RCE) | **0.6952** | **0.6932** | 0.7178 | **0.6685** |

The LaTeX report and final tables are saved in:

```text
framework3/results/project_report.tex
framework3/results/final_experiment_tables.tex
```

---

## Output Format

Prediction files are JSONL. Each row contains:

```json
{
  "id": "0",
  "text": "...",
  "pred": "sarcastic",
  "gold": "non_sarcastic",
  "confidence": 0.9,
  "raw_output": "..."
}
```

Evaluation files contain:

```json
{
  "name": "semeval200_framework3",
  "total": 200,
  "accuracy": 0.75,
  "macro_f1": 0.7496,
  "per_label": {},
  "confusion_matrix": {}
}
```

---

## Summary

Framework3 is a method-level hybrid sarcasm detector with GRPO optimization:

```text
PMP pragmatic reasoning
  + RCE structured reasoning
  + GRPO reward optimization
  = Framework3 hybrid sarcasm detection
```

Framework3 is designed to improve over direct classification, PMP-only prompting, and RCE-only structured reasoning. Its key advantage is that the hybrid reward can reinforce correct predictions while encouraging both pragmatic reasoning coverage and structured RCE evidence coverage.
