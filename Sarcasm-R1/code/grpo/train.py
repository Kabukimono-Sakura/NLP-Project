#!/usr/bin/env python
"""Sarcasm-R1 GRPO Training Script.

Trains a small language model (Qwen2.5-1.5B/7B) to perform structured
three-dimensional sarcasm reasoning using GRPO (Group Relative Policy Optimization).

The model learns to reason through Rhetoric, Context, and Emotion dimensions
before making a sarcasm judgment, guided by multi-component reward functions.

Usage:
  accelerate launch --config_file=deepspeed_zero2.yaml train.py --config config.yaml

References:
  - GRPO: Group Relative Policy Optimization (Shao et al., 2024)
  - DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL
  - Generative Verifiers (Zhang et al., ICLR 2025)
"""

from __future__ import annotations
import os
import sys

import json
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from rewards import (
    accuracy_reward,
    accuracy_reward_log,
    combined_reward,
    format_reasoning_reward,
    dimension_coverage_reward,
    parse_reasoning_content,
    get_completion_content,
)
from prompts import SYSTEM_PROMPT, build_messages


def load_data(data_path: str, split: str = "train") -> Dataset:
    """Load and format training data for GRPO.

    Expects a CSV with columns: 'question', 'answer'
    Optionally: 'prompt' (system prompt override), 'source_text', 'source_label'

    The dataset is converted to the format expected by GRPOTrainer:
      - prompt: list of message dicts [{"role": ..., "content": ...}, ...]
      - answer: ground truth label
    """
    if not data_path:
        raise ValueError("data_path must be specified in config or as argument")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Raw data: {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")

    records = []
    for _, row in df.iterrows():
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        custom_system = str(row.get("prompt", ""))

        if not question.strip() or not answer.strip():
            continue

        system_content = custom_system if custom_system and custom_system != "nan" else SYSTEM_PROMPT

        records.append({
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": question},
            ],
            "answer": answer,
        })

    dataset = Dataset.from_list(records)
    print(f"  Formatted: {len(dataset)} samples")
    if records:
        label_counts = {}
        for r in records:
            label_counts[r["answer"]] = label_counts.get(r["answer"], 0) + 1
        print(f"  Distribution: {label_counts}")

    return dataset


def main(training_args: GRPOConfig, model_args: ModelConfig):
    """Main training entry point."""
    # Determine data path from config or environment
    data_path = os.environ.get("SARCASM_DATA_PATH", getattr(training_args, "data_path", ""))
    if not data_path:
        # Check common locations
        for path in [
            "../../data/processed/train_combined.csv",
            "../../data/processed/semeval_train.csv",
            "train_data.csv",
        ]:
            if os.path.exists(path):
                data_path = path
                break

    if not data_path:
        print("ERROR: No training data found. Set SARCASM_DATA_PATH or place data in data/processed/")
        print("  Run: python code/data_processing/format_grpo_data.py --combine")
        sys.exit(1)

    data = load_data(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Reward functions for GRPO
    # Use combined_reward as the primary signal (accuracy-dominant)
    # accuracy_reward_log is for logging only (not used for training gradient)
    reward_funcs = [
        combined_reward,
        accuracy_reward_log,
    ]

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=data,
    )

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: {model_args.model_name_or_path}")
    print(f"  Data: {len(data)} samples")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Generations per prompt: {training_args.num_generations}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Output: {training_args.output_dir}")
    print(f"  Reward functions: {[f.__name__ for f in reward_funcs]}")
    print(f"{'='*60}\n")

    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"\nModel saved to: {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((GRPOConfig, ModelConfig))
    training_args, model_args = parser.parse_args_and_config()
    main(training_args, model_args)
