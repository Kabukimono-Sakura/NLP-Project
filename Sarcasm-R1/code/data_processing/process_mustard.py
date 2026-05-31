#!/usr/bin/env python
"""Process MUStARD dataset for Sarcasm-R1 GRPO training.

Converts the MUStARD dataset (multimodal sarcasm detection from TV shows)
into a standardized CSV with:
  - question: the dialogue context + target utterance with sarcasm detection prompt
  - answer: "sarcastic" or "non_sarcastic"

The MUStARD dataset contains JSON files with speaker utterances and context.
Each entry has:
  - utterance: the target statement to analyze
  - context: preceding dialogue (list of {speaker, utterance} dicts)
  - sarcasm: boolean label

Usage:
  python process_mustard.py --input_dir ../../data/raw/MUStARD --output_dir ../../data/processed
"""

from __future__ import annotations
import argparse
import json
import os
from typing import List, Optional

import pandas as pd

SYSTEM_HINT = (
    "You will be given dialogue from a TV show or movie. "
    "Analyze the target statement and determine if it is sarcastic, "
    "considering the conversation context, rhetorical devices, and emotional contrast."
)


def build_question(utterance: str, context: Optional[List] = None,
                   context_speakers: Optional[List] = None) -> str:
    context_str = ""
    if context:
        lines = []
        for i, c in enumerate(context):
            speaker = (context_speakers[i] if context_speakers and i < len(context_speakers)
                       else "Speaker")
            if isinstance(c, dict):
                lines.append(f"  {c.get('speaker', speaker)}: {c.get('utterance', '')}")
            else:
                lines.append(f"  {speaker}: {c}")
        context_str = "Conversation context:\n" + "\n".join(lines) + "\n\n"

    return (
        f"{SYSTEM_HINT}\n\n"
        f"{context_str}"
        f"Target statement: \"{utterance}\"\n\n"
        f"Is this statement sarcastic? Respond with your reasoning, "
        f"then provide your answer as either 'sarcastic' or 'non_sarcastic'."
    )


def process_json(json_path: str) -> pd.DataFrame:
    """Process MUStARD JSON format."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.values() if isinstance(data, dict) else data

    records = []
    for entry in entries:
        utterance = entry.get("utterance", "")
        context = entry.get("context", [])
        context_speakers = entry.get("context_speakers", None)
        sarcasm = entry.get("sarcasm", False)

        if not utterance.strip():
            continue

        records.append({
            "question": build_question(utterance, context, context_speakers),
            "answer": "sarcastic" if sarcasm else "non_sarcastic",
            "source_utterance": utterance,
            "source_context": json.dumps(context, ensure_ascii=False),
            "source_label": int(sarcasm),
        })
    return pd.DataFrame(records)


def process_csv(csv_path: str) -> pd.DataFrame:
    """Process MUStARD CSV format (HuggingFace export)."""
    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        # HuggingFace format may have different column names
        utterance = str(row.get("utterance", row.get("text", "")))
        label_col = row.get("label", row.get("sarcasm", row.get("is_sarcastic", 0)))

        if isinstance(label_col, str):
            label = label_col.lower() in ("true", "1", "sarcastic", "yes")
        else:
            label = bool(label_col)

        if not utterance.strip():
            continue

        context = row.get("context", None)
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except (json.JSONDecodeError, TypeError):
                context = None

        records.append({
            "question": build_question(utterance, context if isinstance(context, list) else None),
            "answer": "sarcastic" if label else "non_sarcastic",
            "source_utterance": utterance,
            "source_context": str(context) if context else "",
            "source_label": int(label),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Process MUStARD data")
    parser.add_argument("--input_dir", type=str,
                        default="../../data/raw/MUStARD")
    parser.add_argument("--output_dir", type=str,
                        default="../../data/processed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Try JSON format first, then CSV
    json_path = os.path.join(args.input_dir, "sarcasm_data.json")
    if os.path.exists(json_path):
        df = process_json(json_path)
        out_path = os.path.join(args.output_dir, "mustard_train.csv")
        df.to_csv(out_path, index=False)
        print(f"Processed from JSON: {len(df)} samples -> {out_path}")
    else:
        for split in ["train", "test", "validation"]:
            csv_path = os.path.join(args.input_dir, f"{split}.csv")
            if not os.path.exists(csv_path):
                print(f"Skipping {split}: {csv_path} not found")
                continue

            df = process_csv(csv_path)
            out_path = os.path.join(args.output_dir, f"mustard_{split}.csv")
            df.to_csv(out_path, index=False)
            print(f"Processed {split}: {len(df)} samples -> {out_path}")
            if "answer" in df.columns:
                print(f"  Distribution: {df['answer'].value_counts().to_dict()}")

    print("MUStARD processing complete.")


if __name__ == "__main__":
    main()
