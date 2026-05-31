#!/usr/bin/env python
"""Inference script for Sarcasm-R1 trained models.

Runs sarcasm detection on test data using a trained GRPO model.

Usage:
  python inference.py --model ./output/sarcasm-r1 --data ../../data/processed/semeval_test.csv --output results.jsonl
  python inference.py --model ./output/sarcasm-r1 --data ../../data/processed/mustard_test.csv --output results.jsonl --batch_size 8
"""

from __future__ import annotations
import argparse
import json
import os
import re

# Patch old huggingface_hub: skip repo_id validation for local directories
try:
    import huggingface_hub.utils._validators as _hf_val
    _orig_validate = _hf_val.validate_repo_id
    def _patched_validate(repo_id, **kwargs):
        if os.path.isdir(str(repo_id)):
            return
        return _orig_validate(repo_id, **kwargs)
    _hf_val.validate_repo_id = _patched_validate
except Exception:
    pass

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add parent directory for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grpo"))
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, USER_PROMPT_MUSTARD_TEMPLATE
from rewards import extract_answer


def load_model(model_path: str, base_model: str = None, device: str = "auto"):
    """Load a trained model (full or LoRA adapter)."""
    # Load tokenizer from base_model first (checkpoints may lack tokenizer files)
    tokenizer_path = base_model if base_model else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if base_model and os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # Load LoRA adapter
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )

    model.eval()
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list[str]:
    """Run batched inference."""
    all_outputs = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only new tokens
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            new_tokens = output[input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_outputs.append(text)

        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"  Processed {i + batch_size}/{len(prompts)} samples")

    return all_outputs


def main():
    parser = argparse.ArgumentParser(description="Sarcasm-R1 Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model (for LoRA)")
    parser.add_argument("--data", type=str, required=True, help="Test data CSV")
    parser.add_argument("--output", type=str, default="results.jsonl", help="Output JSONL")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, args.base_model)

    print(f"Loading test data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df)} test samples")

    # Build prompts
    full_prompts = []
    for _, row in df.iterrows():
        question = str(row.get("question", ""))
        system = str(row.get("prompt", SYSTEM_PROMPT))

        messages = [
            {"role": "system", "content": system if system != "nan" else SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_prompts.append(prompt_text)

    print(f"Running inference...")
    responses = run_inference(
        model, tokenizer, full_prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Extract answers and save
    results = []
    for i, (response, (_, row)) in enumerate(zip(responses, df.iterrows())):
        predicted = extract_answer(response)
        ground_truth = str(row.get("answer", ""))

        results.append({
            "index": i,
            "response": response,
            "predicted": predicted or "unknown",
            "ground_truth": ground_truth,
            "correct": predicted == ground_truth if predicted else False,
            "source_text": str(row.get("source_text", row.get("source_utterance", ""))),
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\nResults: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
