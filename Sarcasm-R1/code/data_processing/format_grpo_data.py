#!/usr/bin/env python
"""Format processed datasets into GRPO training format.

Combines processed CSVs and outputs the final training file with:
  - question: prompt text (system + user message content)
  - answer: expected answer ("sarcastic" or "non_sarcastic")

The GRPO trainer expects a dataset with 'prompt' and 'answer' fields,
where 'prompt' is a list of message dicts.

Usage:
  python format_grpo_data.py --input_dir ../../data/processed --output_dir ../../data/processed
  python format_grpo_data.py --input_dir ../../data/processed --output_dir ../../data/processed --combine
"""

from __future__ import annotations
import argparse
import os

import pandas as pd

SYSTEM_PROMPT = (
    "You are an expert at detecting sarcasm in text. "
    "Analyze the given text by examining three key dimensions:\n"
    "1. Rhetoric: Look for rhetorical devices like irony, hyperbole, or understatement.\n"
    "2. Context: Consider the situation, background knowledge, and conversational context.\n"
    "3. Emotion: Identify emotional contrasts between literal meaning and implied sentiment.\n\n"
    "Provide your analysis for each dimension, then conclude with your judgment.\n"
    "Format your response as:\n"
    "{Rhetoric analysis}\n"
    "{Context analysis}\n"
    "{Emotion analysis}\n"
    "{sarcastic or non_sarcastic}"
)


def format_for_grpo(df: pd.DataFrame) -> pd.DataFrame:
    """Convert processed CSV to GRPO training format.

    Each row becomes:
      - prompt: [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": <question without system hint>}]
      - answer: "sarcastic" or "non_sarcastic"
    """
    records = []
    for _, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]

        # Strip the system hint from the question since it's now in system prompt
        user_content = question
        # The question already contains a hint line; keep the text part
        # We'll use the question as-is for the user message

        records.append({
            "prompt": SYSTEM_PROMPT,
            "question": user_content,
            "answer": answer,
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Format data for GRPO training")
    parser.add_argument("--input_dir", type=str, default="../../data/processed")
    parser.add_argument("--output_dir", type=str, default="../../data/processed")
    parser.add_argument("--combine", action="store_true",
                        help="Combine all datasets into one training file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all processed CSVs
    all_dfs = []
    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith(".csv"):
            continue
        if fname.startswith("train_") or fname.startswith("grpo_"):
            continue
        fpath = os.path.join(args.input_dir, fname)
        df = pd.read_csv(fpath)
        if "question" in df.columns and "answer" in df.columns:
            all_dfs.append(df)
            print(f"  Loaded {fname}: {len(df)} samples")

    if not all_dfs:
        print("No processed CSV files found. Run process_semeval.py and process_mustard.py first.")
        return

    if args.combine:
        combined = pd.concat(all_dfs, ignore_index=True)
        grpo_df = format_for_grpo(combined)
        out_path = os.path.join(args.output_dir, "train_combined.csv")
        grpo_df.to_csv(out_path, index=False)
        print(f"\nCombined GRPO data: {len(grpo_df)} samples -> {out_path}")
        print(f"  Distribution: {grpo_df['answer'].value_counts().to_dict()}")
    else:
        for i, df in enumerate(all_dfs):
            grpo_df = format_for_grpo(df)
            source = os.listdir(args.input_dir)
            source = [f for f in source if f.endswith(".csv") and not f.startswith(("train_", "grpo_"))]
            out_name = f"train_{source[i]}" if i < len(source) else f"train_{i}.csv"
            out_path = os.path.join(args.output_dir, out_name)
            grpo_df.to_csv(out_path, index=False)
            print(f"  GRPO data: {len(grpo_df)} samples -> {out_path}")


if __name__ == "__main__":
    main()
