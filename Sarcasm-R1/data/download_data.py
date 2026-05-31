#!/usr/bin/env python
"""Download datasets for Sarcasm-R1.

Downloads:
  - SemEval 2018 Task 3 (via HuggingFace tweet_eval/irony)
  - MUStARD (via HuggingFace, fallback to Google Drive instructions)

Usage:
  python download_data.py --output_dir ./raw
  python download_data.py --output_dir ./raw --dataset semeval
  python download_data.py --output_dir ./raw --dataset mustard
"""
from __future__ import annotations
import argparse
import os

import pandas as pd


def download_semeval(output_dir: str):
    """Download SemEval 2018 Task 3 via HuggingFace tweet_eval/irony."""
    from datasets import load_dataset

    print("[SemEval] Downloading tweet_eval/irony from HuggingFace...")
    ds = load_dataset("tweet_eval", "irony")

    semeval_dir = os.path.join(output_dir, "SemEval2018-Task3")
    os.makedirs(semeval_dir, exist_ok=True)

    for split in ds:
        df = ds[split].to_pandas()
        path = os.path.join(semeval_dir, f"{split}.csv")
        df.to_csv(path, index=False)
        print(f"  {split}: {len(df)} samples -> {path}")

    print("[SemEval] Done.\n")
    return semeval_dir


def download_mustard(output_dir: str):
    """Download MUStARD dataset from HuggingFace."""
    print("[MUStARD] Downloading MUStARD dataset from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("skhurana/MUStARD")
        mustard_dir = os.path.join(output_dir, "MUStARD")
        os.makedirs(mustard_dir, exist_ok=True)
        for split in ds:
            path = os.path.join(mustard_dir, f"{split}.csv")
            ds[split].to_csv(path)
        print("[MUStARD] Done.\n")
        return mustard_dir
    except Exception as e:
        print(f"[MUStARD] HuggingFace download failed: {e}")
        print("[MUStARD] Please download manually from:")
        print("  https://github.com/soujanyaporia/MUStARD")
        print("  Place sarcasm_data.json in: data/raw/MUStARD/")
        print()
        return None


def verify_data(output_dir: str):
    """Verify downloaded data."""
    print("=== Verification ===")
    semeval_path = os.path.join(output_dir, "SemEval2018-Task3", "train.csv")
    if os.path.exists(semeval_path):
        df = pd.read_csv(semeval_path)
        print(f"SemEval train: {len(df)} samples, columns: {list(df.columns)}")
    else:
        print("SemEval: train.csv not found.")

    mustard_path = os.path.join(output_dir, "MUStARD", "train.csv")
    if os.path.exists(mustard_path):
        df = pd.read_csv(mustard_path)
        print(f"MUStARD train: {len(df)} samples, columns: {list(df.columns)}")
    else:
        print("MUStARD: train.csv not found (may need manual download).")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download Sarcasm-R1 datasets")
    parser.add_argument("--output_dir", type=str, default="./raw",
                        help="Output directory for raw data")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "semeval", "mustard"],
                        help="Which dataset to download")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset in ("all", "semeval"):
        download_semeval(args.output_dir)

    if args.dataset in ("all", "mustard"):
        download_mustard(args.output_dir)

    verify_data(args.output_dir)
    print("Download complete.")


if __name__ == "__main__":
    main()
