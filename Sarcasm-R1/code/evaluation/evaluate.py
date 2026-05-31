#!/usr/bin/env python
"""Evaluation script for Sarcasm-R1.

Computes evaluation metrics (Accuracy, Macro-F1) on inference results
and generates a classification report.

Usage:
  python evaluate.py --results results.jsonl --output eval_report.json
  python evaluate.py --results results.jsonl --compare baseline_results.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
from collections import Counter

LABEL_SET = ["sarcastic", "non_sarcastic"]


def load_results(path: str) -> list[dict]:
    """Load inference results from JSONL."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute Accuracy, Precision, Recall, F1 for each class, and Macro-F1."""
    tp = {label: 0 for label in LABEL_SET}
    fp = {label: 0 for label in LABEL_SET}
    fn = {label: 0 for label in LABEL_SET}

    for r in results:
        pred = r.get("predicted", "unknown")
        gt = r.get("ground_truth", "")

        if pred == gt:
            tp[gt] = tp.get(gt, 0) + 1
        else:
            fp[pred] = fp.get(pred, 0) + 1
            fn[gt] = fn.get(gt, 0) + 1

    # Per-class metrics
    class_metrics = {}
    f1_scores = []

    for label in LABEL_SET:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

        class_metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp[label] + fn[label],
        }

    # Overall metrics
    total = len(results)
    correct = sum(1 for r in results if r.get("predicted") == r.get("ground_truth"))
    accuracy = correct / total if total > 0 else 0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    # Confusion matrix
    pred_dist = Counter(r.get("predicted", "unknown") for r in results)
    gt_dist = Counter(r.get("ground_truth", "") for r in results)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": total,
        "correct": correct,
        "class_metrics": class_metrics,
        "prediction_distribution": dict(pred_dist),
        "ground_truth_distribution": dict(gt_dist),
    }


def compare_results(results_a: list[dict], results_b: list[dict], name_a: str, name_b: str) -> dict:
    """Compare two sets of results."""
    metrics_a = compute_metrics(results_a)
    metrics_b = compute_metrics(results_b)

    return {
        name_a: metrics_a,
        name_b: metrics_b,
        "accuracy_diff": round(metrics_a["accuracy"] - metrics_b["accuracy"], 4),
        "macro_f1_diff": round(metrics_a["macro_f1"] - metrics_b["macro_f1"], 4),
    }


def print_report(metrics: dict, title: str = "Evaluation Report"):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Correct:       {metrics['correct']}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:      {metrics['macro_f1']:.4f}")
    print(f"\n  {'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<8}")
    print(f"  {'-'*64}")
    for label, m in metrics["class_metrics"].items():
        print(f"  {label:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<8}")

    print(f"\n  Prediction distribution: {metrics['prediction_distribution']}")
    print(f"  Ground truth distribution: {metrics['ground_truth_distribution']}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sarcasm-R1 results")
    parser.add_argument("--results", type=str, required=True, help="Results JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Save report as JSON")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare with another results file")
    parser.add_argument("--name", type=str, default="Sarcasm-R1",
                        help="Name for the current results")
    parser.add_argument("--compare_name", type=str, default="Baseline",
                        help="Name for comparison results")
    args = parser.parse_args()

    results = load_results(args.results)
    metrics = compute_metrics(results)
    print_report(metrics, title=f"{args.name} Evaluation Report")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output}")

    if args.compare:
        compare_results_data = load_results(args.compare)
        comparison = compare_results(results, compare_results_data, args.name, args.compare_name)
        print(f"\n  Comparison: {args.name} vs {args.compare_name}")
        print(f"  Accuracy difference: {comparison['accuracy_diff']:+.4f}")
        print(f"  Macro-F1 difference: {comparison['macro_f1_diff']:+.4f}")


if __name__ == "__main__":
    main()
