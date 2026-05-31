#!/usr/bin/env python
"""Process SemEval 2018 Task 3 (tweet_eval irony) data for Sarcasm-R1 GRPO training.

Converts the tweet_eval/irony dataset into a standardized CSV with:
  - question: the input text wrapped in a sarcasm detection prompt
  - answer: "sarcastic" or "non_sarcastic"

Usage:
  python process_semeval.py --input_dir ../../data/raw/SemEval2018-Task3 --output_dir ../../data/processed
"""

from __future__ import annotations
import argparse
import os

import pandas as pd

LABEL_MAP = {0: "non_sarcastic", 1: "sarcastic"}

SYSTEM_HINT = (
    "Analyze the following text and determine if it is sarcastic. "
    "Consider rhetorical devices, context, and emotional contrast."
)


def build_question(text: str) -> str:
    return (
        f"{SYSTEM_HINT}\n\n"
        f"Text: \"{text}\"\n\n"
        f"Is this text sarcastic? Respond with your reasoning, "
        f"then provide your answer as either 'sarcastic' or 'non_sarcastic'."
    )


def process_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        label = int(row.get("label", 0))
        if not text.strip():
            continue
        records.append({
            "question": build_question(text),
            "answer": LABEL_MAP[label],
            "source_text": text,
            "source_label": label,
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Process SemEval 2018 Task 3 data")
    parser.add_argument("--input_dir", type=str,
                        default="../../data/raw/SemEval2018-Task3")
    parser.add_argument("--output_dir", type=str,
                        default="../../data/processed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "test", "validation"]:
        csv_path = os.path.join(args.input_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {split}: {csv_path} not found")
            continue

        df = process_split(csv_path)
        out_path = os.path.join(args.output_dir, f"semeval_{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"Processed {split}: {len(df)} samples -> {out_path}")
        if "answer" in df.columns:
            print(f"  Distribution: {df['answer'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
