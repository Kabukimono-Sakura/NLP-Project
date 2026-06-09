#!/usr/bin/env python3
"""Evaluate a binary sarcasm prediction JSONL file.

Expected JSONL fields:
  {"id": "...", "gold": "sarcastic", "pred": "non_sarcastic"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


LABELS = ("sarcastic", "non_sarcastic")


def normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_").strip("{}[]().,:;\"'")
    if text in {"1", "true", "yes", "sarcastic", "ironic", "irony"}:
        return "sarcastic"
    if text in {"0", "false", "no", "non_sarcastic", "not_sarcastic", "nonsarcastic", "literal"}:
        return "non_sarcastic"
    raise ValueError(f"Unknown label: {value!r}")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "gold" not in row:
                raise ValueError(f"{path}:{line_no} missing gold")
            if "pred" not in row:
                raise ValueError(f"{path}:{line_no} missing pred")
            row["gold"] = normalize_label(row["gold"])
            row["pred"] = normalize_label(row["pred"])
            rows.append(row)
    return rows


def classification_report(golds: Iterable[str], preds: Iterable[str]) -> dict:
    gold_list = list(golds)
    pred_list = list(preds)
    total = len(gold_list)
    correct = sum(g == p for g, p in zip(gold_list, pred_list))

    per_label = {}
    f1_values = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold_list, pred_list))
        fp = sum(g != label and p == label for g, p in zip(gold_list, pred_list))
        fn = sum(g == label and p != label for g, p in zip(gold_list, pred_list))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(g == label for g in gold_list),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    tp = sum(g == "sarcastic" and p == "sarcastic" for g, p in zip(gold_list, pred_list))
    fp = sum(g == "non_sarcastic" and p == "sarcastic" for g, p in zip(gold_list, pred_list))
    fn = sum(g == "sarcastic" and p == "non_sarcastic" for g, p in zip(gold_list, pred_list))
    tn = sum(g == "non_sarcastic" and p == "non_sarcastic" for g, p in zip(gold_list, pred_list))

    return {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_label": per_label,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", default="prediction")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    result = {
        "name": args.name,
        "input": str(args.input),
        **classification_report([row["gold"] for row in rows], [row["pred"] for row in rows]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

