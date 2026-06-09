#!/usr/bin/env python3
"""Run a local/HuggingFace causal LLM on a sarcasm test set and write JSONL predictions.

Input can be CSV, TSV, or JSONL. Output rows are compatible with
evaluate_hybrid_predictions.py.

Example:
  PYTHONDONTWRITEBYTECODE=1 python3 framework3/scripts/run_llm_predictions.py \
    --input data/semeval200.csv \
    --output predictions/semeval200_cot.jsonl \
    --model Qwen/Qwen3-14B \
    --prompt-file framework3/prompts/teacher_cot_prompt.md \
    --id-column id \
    --text-column text \
    --gold-column gold \
    --default-confidence 1.0 \
    --trust-remote-code
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


LABELS = ("sarcastic", "non_sarcastic")
LABEL_PATTERN = r"non[\s_-]?sarcastic|not[\s_-]?sarcastic|nonsarcastic|sarcastic"


def normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = text.strip("{}[]().,:;\"'")

    sarcastic_values = {
        "1",
        "true",
        "yes",
        "sarcastic",
        "ironic",
        "irony",
        "positive",
    }
    non_sarcastic_values = {
        "0",
        "false",
        "no",
        "non_sarcastic",
        "not_sarcastic",
        "nonsarcastic",
        "literal",
        "negative",
    }

    if text in sarcastic_values:
        return "sarcastic"
    if text in non_sarcastic_values:
        return "non_sarcastic"
    raise ValueError(f"Unknown label: {value!r}")


def extract_prediction(output: str) -> tuple[str, float]:
    text = output.strip()

    final_match = re.search(
        rf"\[\s*final\s*\]\s*\{{?\s*({LABEL_PATTERN})\s*\}}?",
        text,
        flags=re.IGNORECASE,
    )
    if final_match:
        label = normalize_label(final_match.group(1))
        return label, 0.90

    explicit_match = re.search(
        rf"(?:final\s*(?:answer|label|prediction)?|answer|label|prediction)\s*[:：]\s*\{{?\s*({LABEL_PATTERN})\s*\}}?",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_match:
        return normalize_label(explicit_match.group(1)), 0.85

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1].strip("{}[]().,:;\"'")
        if re.fullmatch(LABEL_PATTERN, last_line, flags=re.IGNORECASE):
            return normalize_label(last_line), 0.85

    braced = re.findall(rf"\{{\s*({LABEL_PATTERN})\s*\}}", text, flags=re.IGNORECASE)
    if len(braced) == 1:
        return normalize_label(braced[0]), 0.80

    labels = re.findall(LABEL_PATTERN, text, flags=re.IGNORECASE)
    if labels:
        return normalize_label(labels[-1]), 0.65

    # Ambiguous outputs are treated conservatively as low-confidence non-sarcastic.
    return "non_sarcastic", 0.50


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_delimited(path: Path, delimiter: str) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_input(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".tsv":
        return read_delimited(path, "\t")
    if suffix == ".csv":
        return read_delimited(path, ",")
    raise ValueError(f"Unsupported input format: {path}")


def build_prompt(template: str, text: str) -> str:
    if "{{text}}" in template:
        return template.replace("{{text}}", text)
    return template.rstrip() + "\n\nText:\n" + text + "\n"


def iter_selected(rows: List[dict], limit: Optional[int]) -> Iterable[dict]:
    if limit is None:
        return rows
    return rows[:limit]


def is_peft_adapter(model_path: str) -> bool:
    path = Path(model_path)
    return path.exists() and (path / "adapter_config.json").exists()


def adapter_base_model_path(adapter_path: str, explicit_base: Optional[str]) -> str:
    if explicit_base:
        return explicit_base

    config_path = Path(adapter_path) / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"{config_path} does not contain base_model_name_or_path. "
            "Pass --base-model explicitly."
        )
    return str(base)


def load_model(args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install transformers and torch in the environment "
            "where you run inference."
        ) from exc

    tokenizer_source = args.model
    model_source = args.model
    adapter_source = None

    if is_peft_adapter(args.model):
        adapter_source = args.model
        model_source = adapter_base_model_path(args.model, args.base_model)
        tokenizer_candidate = Path(args.model)
        tokenizer_source = args.model if (tokenizer_candidate / "tokenizer_config.json").exists() else model_source

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
    )

    dtype = None
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

    if adapter_source is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit(
                "This model path is a PEFT/LoRA adapter. Install peft or pass a full merged model."
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_source)

    model.eval()
    return tokenizer, model


def render_chat_or_plain(
    tokenizer,
    prompt: str,
    use_chat_template: bool,
    enable_thinking: bool,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def generate_one(tokenizer, model, prompt: str, args: argparse.Namespace) -> str:
    import torch

    rendered = render_chat_or_plain(
        tokenizer,
        prompt,
        not args.no_chat_template,
        args.enable_thinking,
    )
    inputs = tokenizer(rendered, return_tensors="pt")

    target_device = None
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for device in model.hf_device_map.values():
            if str(device) not in {"cpu", "disk", "meta"}:
                target_device = torch.device(device)
                break
    if target_device is None:
        try:
            device = getattr(model, "device", None) or next(model.parameters()).device
            if str(device) != "meta":
                target_device = device
        except StopIteration:
            target_device = None

    if target_device is not None:
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-model",
        help="Base model path for PEFT/LoRA adapter checkpoints. If omitted, adapter_config.json is used.",
    )
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--gold-column", default="gold")
    parser.add_argument("--omit-gold", action="store_true")
    parser.add_argument("--default-confidence", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode if the tokenizer chat template supports it.",
    )
    args = parser.parse_args()

    rows = read_input(args.input)
    template = args.prompt_file.read_text(encoding="utf-8")
    tokenizer, model = load_model(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(iter_selected(rows, args.limit), start=1):
            sample_id = str(row.get(args.id_column, index - 1))
            text = str(row[args.text_column])
            prompt = build_prompt(template, text)
            raw_output = generate_one(tokenizer, model, prompt, args)
            pred, extracted_confidence = extract_prediction(raw_output)
            confidence = args.default_confidence
            if confidence is None:
                confidence = extracted_confidence

            out: Dict[str, object] = {
                "id": sample_id,
                "text": text,
                "pred": pred,
                "confidence": confidence,
                "raw_output": raw_output,
            }
            if not args.omit_gold and args.gold_column in row and row[args.gold_column] != "":
                out["gold"] = normalize_label(row[args.gold_column])

            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(rows)}] id={sample_id} pred={pred} confidence={confidence:.2f}", flush=True)


if __name__ == "__main__":
    main()
