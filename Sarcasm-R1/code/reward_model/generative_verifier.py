#!/usr/bin/env python
"""Generative Verifier for Sarcasm-R1.

Implements a generative verification approach (Zhang et al., ICLR 2025)
where a verifier model judges the quality of reasoning chains generated
during GRPO training.

The verifier is trained to predict whether a given reasoning chain correctly
leads to the right sarcasm judgment, using next-token prediction.

Usage:
  # Train the verifier
  python generative_verifier.py --mode train --data ../../data/processed/semeval_reasoning.jsonl --model Qwen/Qwen2.5-1.5B-Instruct

  # Use as reward signal
  python generative_verifier.py --mode verify --data ../../data/processed/semeval_reasoning.jsonl --model ./output/verifier

References:
  Zhang et al., "Generative Verifiers: Reward Modeling as Next-Token Prediction", ICLR 2025
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)


class SarcasmVerifierDataset(Dataset):
    """Dataset for training the generative verifier.

    Each example is formatted as:
      [reasoning chain] -> "The reasoning is [correct/incorrect]."

    The verifier learns to predict "correct" or "incorrect" as the next token.
    """

    VERIFY_TEMPLATE = """Analyze whether the following reasoning correctly identifies sarcasm.

Text: "{text}"
Ground truth: {ground_truth}

Reasoning:
{reasoning}

Predicted answer: {predicted}

Is this reasoning and prediction correct? Answer: {label}"""

    def __init__(self, data: list[dict], tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", item.get("source_text", ""))
        ground_truth = item.get("ground_truth", "")
        reasoning = item.get("reasoning", "")
        predicted = item.get("predicted_answer", "")
        is_correct = predicted == ground_truth

        prompt = self.VERIFY_TEMPLATE.format(
            text=text,
            ground_truth=ground_truth,
            reasoning=reasoning[:500],
            predicted=predicted,
            label="correct." if is_correct else "incorrect.",
        )

        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def load_reasoning_data(data_path: str, limit: int = -1) -> list[dict]:
    """Load reasoning annotations from JSONL file."""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if limit > 0 and len(data) >= limit:
                    break
    return data


def train_verifier(args):
    """Train the generative verifier model."""
    print(f"Loading data from {args.data}...")
    data = load_reasoning_data(args.data, args.limit)
    print(f"  Loaded {len(data)} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Split into train/val
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = SarcasmVerifierDataset(train_data, tokenizer, args.max_length)
    val_dataset = SarcasmVerifierDataset(val_data, tokenizer, args.max_length) if val_data else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_dataset else "no",
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        report_to="wandb" if args.wandb else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    print(f"\nTraining verifier: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val")
    trainer.train()
    trainer.save_model(args.output)
    print(f"Verifier saved to {args.output}")


def verify_reasoning(args):
    """Use the verifier to score reasoning chains."""
    print(f"Loading verifier from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    data = load_reasoning_data(args.data, args.limit)
    scores = []

    for item in data:
        text = item.get("text", "")
        ground_truth = item.get("ground_truth", "")
        reasoning = item.get("reasoning", "")
        predicted = item.get("predicted_answer", "")

        prompt = SarcasmVerifierDataset.VERIFY_TEMPLATE.format(
            text=text,
            ground_truth=ground_truth,
            reasoning=reasoning[:500],
            predicted=predicted,
            label="",  # Leave blank for prediction
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        is_correct = "correct" in response.lower() and "incorrect" not in response.lower()
        scores.append(1.0 if is_correct else 0.0)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\nVerification results: {sum(scores)}/{len(scores)} correct ({avg_score*100:.1f}%)")

    if args.output:
        results = [{"score": s} for s in scores]
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sarcasm-R1 Generative Verifier")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "verify"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", type=str, required=True, help="Reasoning JSONL file")
    parser.add_argument("--output", type=str, default="./output/verifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        train_verifier(args)
    elif args.mode == "verify":
        verify_reasoning(args)


if __name__ == "__main__":
    main()
