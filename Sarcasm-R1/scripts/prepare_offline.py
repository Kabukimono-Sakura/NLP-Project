#!/usr/bin/env python
"""Sarcasm-R1 — 本地离线准备脚本

在有网络的本地机器上运行，预下载模型和数据，然后上传到集群。

Usage:
  python scripts/prepare_offline.py                     # 下载全部 (7B)
  python scripts/prepare_offline.py --model-size 1.5b   # 使用 1.5B 模型
  python scripts/prepare_offline.py --skip-model        # 跳过模型下载
  python scripts/prepare_offline.py --skip-data         # 跳过数据下载
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys


def get_project_dir():
    """Get project root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def download_model(model_size: str, project_dir: str):
    """Download model from HuggingFace to local directory."""
    size_map = {"7b": "7B", "1.5b": "1.5B"}
    normalized = size_map.get(model_size.lower(), model_size.upper())
    model_name = f"Qwen/Qwen2.5-{normalized}-Instruct"
    save_dir = os.path.join(project_dir, "models", model_name.split("/")[-1])

    os.makedirs(save_dir, exist_ok=True)

    print(f"  Model: {model_name}")
    print(f"  Save to: {save_dir}")
    print()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)
    print("  Tokenizer done.")

    print("  Downloading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.save_pretrained(save_dir)

    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(save_dir)
        for f in fns
    )
    print(f"  Model saved: {total_size / 1e9:.2f} GB")
    print()


def download_data(project_dir: str):
    """Download datasets."""
    data_dir = os.path.join(project_dir, "data")
    script = os.path.join(data_dir, "download_data.py")

    if not os.path.exists(script):
        print(f"  [ERROR] {script} not found")
        return

    raw_dir = os.path.join(data_dir, "raw")
    print(f"  Running: python download_data.py --output_dir ./raw")
    print()

    subprocess.run(
        [sys.executable, script, "--output_dir", raw_dir],
        cwd=data_dir,
        check=False,
    )
    print()


def main():
    parser = argparse.ArgumentParser(description="Sarcasm-R1 Offline Preparation")
    parser.add_argument("--model-size", type=str, default="7b",
                        choices=["7b", "1.5b"],
                        help="Model size to download (default: 7b)")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip model download")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data download")
    args = parser.parse_args()

    project_dir = get_project_dir()
    os.makedirs(os.path.join(project_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "hf_cache"), exist_ok=True)

    print("=" * 60)
    print(" Sarcasm-R1 Offline Preparation")
    print(f" Project dir: {project_dir}")
    print(f" Model size:  {args.model_size}")
    print("=" * 60)
    print()

    # 1. 下载模型
    if args.skip_model:
        print("[1/3] Skipping model download (--skip-model)")
    else:
        print("[1/3] Downloading model...")
        download_model(args.model_size, project_dir)

    # 2. 下载数据
    if args.skip_data:
        print("[2/3] Skipping data download (--skip-data)")
    else:
        print("[2/3] Downloading datasets...")
        download_data(project_dir)

    # 3. HF缓存目录
    print("[3/3] Setting up HF cache directory...")
    hf_cache = os.path.join(project_dir, "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    print(f"  Created: {hf_cache}")

    # 汇总
    print()
    print("=" * 60)
    print(" Preparation Complete!")
    print()
    print(" Next steps:")
    print("   1. Upload to cluster:")
    print(f"      rsync -avz {project_dir}/ <cluster>:~/Sarcasm-R1/")
    print("   2. Submit on cluster:")
    print("      sbatch scripts/run_all.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
