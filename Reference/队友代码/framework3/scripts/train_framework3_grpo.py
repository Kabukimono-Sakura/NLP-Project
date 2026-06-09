#!/usr/bin/env python3
"""Train Framework3 with GRPO on the existing sarcasm data.

Framework3 combines the PMP dimensions from Framework1 with the structured
Rhetoric / Context / Emotion dimensions from Framework2. The reward keeps
accuracy as the main signal and adds light rewards for covering both reasoning
families before producing the final sarcasm label.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

try:
    from trl import get_peft_config
except ImportError:
    def get_peft_config(model_args):
        return None


SYSTEM_PROMPT = """You are an expert sarcasm detection model.

Use a hybrid Framework3 reasoning process that combines:
1. Pragmatic Metacognitive Prompting: literal meaning, speaker intention,
   contextual fit, pragmatic incongruity, and affective contrast.
2. Structured Rhetoric / Context / Emotion reasoning: rhetorical signals,
   context mismatch, and emotional mismatch.

For each input, reason through the useful evidence and then finish with exactly
one final label in this format:
[Final] {sarcastic}
or
[Final] {non_sarcastic}
"""

USER_PROMPT_TEMPLATE = """Determine whether the following text is sarcastic.

Text: "{text}"
"""

LABEL_PATTERN = r"non[\s_-]?sarcastic|not[\s_-]?sarcastic|nonsarcastic|sarcastic"


def normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_").strip("{}[]().,:;\"'")
    if text in {"1", "true", "yes", "sarcastic", "ironic", "irony"}:
        return "sarcastic"
    if text in {"0", "false", "no", "non_sarcastic", "not_sarcastic", "nonsarcastic", "literal"}:
        return "non_sarcastic"
    raise ValueError(f"Unknown label: {value!r}")


def completion_content(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(completion)


def extract_answer(text: str) -> Optional[str]:
    final_match = re.search(
        rf"\[\s*final\s*\]\s*\{{?\s*({LABEL_PATTERN})\s*\}}?",
        text,
        flags=re.IGNORECASE,
    )
    if final_match:
        return normalize_label(final_match.group(1))

    brace_matches = re.findall(rf"\{{\s*({LABEL_PATTERN})\s*\}}", text, flags=re.IGNORECASE)
    if brace_matches:
        return normalize_label(brace_matches[-1])

    label_matches = re.findall(LABEL_PATTERN, text[-300:], flags=re.IGNORECASE)
    if label_matches:
        return normalize_label(label_matches[-1])

    return None


def strip_answer(text: str) -> str:
    return re.sub(
        rf"\[\s*final\s*\]\s*\{{?\s*({LABEL_PATTERN})\s*\}}?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def count_patterns(text: str, patterns: list[str]) -> int:
    lower = text.lower()
    return sum(1 for pattern in patterns if re.search(pattern, lower, flags=re.IGNORECASE))


def accuracy_reward(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_answer(completion_content(completion))
        rewards.append(20.0 if pred == normalize_label(gold) else -8.0)
    return rewards


def format_reward(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        text = completion_content(completion)
        reasoning = strip_answer(text)
        score = 0.0

        if extract_answer(text) is not None:
            score += 2.0
        else:
            score -= 3.0

        if len(reasoning) >= 40:
            score += 0.5
        if len(reasoning) >= 100:
            score += 0.5
        if re.search(r"\[\s*final\s*\]", text, flags=re.IGNORECASE):
            score += 0.5

        rewards.append(score)
    return rewards


def pmp_coverage_reward(prompts, completions, answer, **kwargs) -> list[float]:
    patterns = [
        r"\bliteral\b|\bexplicit\b|\bsurface\b",
        r"\bintention\b|\bspeaker\b|\bimplied\b|\bmeans\b",
        r"\bpragmatic\b|\bincongruity\b|\bmismatch\b|\bcontrast\b",
        r"\baffective\b|\battitude\b|\bcritici[sz]e\b|\bmock\b",
    ]
    rewards = []
    for completion in completions:
        covered = count_patterns(completion_content(completion), patterns)
        score = float(covered)
        if covered >= 3:
            score += 1.0
        rewards.append(score)
    return rewards


def rce_coverage_reward(prompts, completions, answer, **kwargs) -> list[float]:
    patterns = [
        r"\brhetoric|\birony|\bhyperbole|\bunderstatement|\bexaggeration|\bmock praise",
        r"\bcontext|\bsituation|\bbackground|\bcommon sense",
        r"\bemotion|\bemotional|\bsentiment|\bfeeling|\bfrustration",
    ]
    rewards = []
    for completion in completions:
        covered = count_patterns(completion_content(completion), patterns)
        score = 1.5 * covered
        if covered == 3:
            score += 1.5
        rewards.append(score)
    return rewards


def hybrid_reward(prompts, completions, answer, **kwargs) -> list[float]:
    acc = accuracy_reward(prompts, completions, answer)
    fmt = format_reward(prompts, completions, answer)
    pmp = pmp_coverage_reward(prompts, completions, answer)
    rce = rce_coverage_reward(prompts, completions, answer)
    return [
        a + 0.15 * f + 0.20 * p + 0.20 * r
        for a, f, p, r in zip(acc, fmt, pmp, rce)
    ]


def reward_log(prompts, completions, answer, **kwargs) -> list[float]:
    if completions:
        text = completion_content(completions[0])
        print(
            "\n" + "-" * 50
            + "\nExample Framework3 GRPO response:\n"
            + text[:700]
            + "\nGold: "
            + str(answer[0])
            + "\nPred: "
            + str(extract_answer(text))
            + "\n"
            + "-" * 50,
            flush=True,
        )
    return [0.0] * len(answer)


def load_training_data(data_path: Path) -> Dataset:
    df = pd.read_csv(data_path)
    records = []

    for _, row in df.iterrows():
        if "source_text" in row and pd.notna(row["source_text"]) and str(row["source_text"]).strip():
            text = str(row["source_text"])
        elif "text" in row and pd.notna(row["text"]) and str(row["text"]).strip():
            text = str(row["text"])
        else:
            text = str(row.get("question", ""))

        raw_answer = row.get("answer", row.get("label", row.get("source_label", "")))
        if not text.strip() or str(raw_answer).strip() == "":
            continue

        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
                ],
                "answer": normalize_label(raw_answer),
            }
        )

    if not records:
        raise ValueError(f"No usable training rows found in {data_path}")

    label_counts: dict[str, int] = {}
    for record in records:
        label_counts[record["answer"]] = label_counts.get(record["answer"], 0) + 1

    print(f"Loaded {len(records)} Framework3 GRPO training samples from {data_path}")
    print(f"Label distribution: {label_counts}")
    return Dataset.from_list(records)


def main(training_args: GRPOConfig, model_args: ModelConfig) -> None:
    data_path = Path(os.environ.get("SARCASM_DATA_PATH", "framework2/Sarcasm-R1/data/processed/semeval_train.csv"))
    if not data_path.exists():
        print(f"ERROR: training data not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    train_dataset = load_training_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=getattr(model_args, "trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=[hybrid_reward, reward_log],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
    )

    print("=" * 60)
    print("Framework3 Hybrid GRPO Training")
    print(f"Model : {model_args.model_name_or_path}")
    print(f"Data  : {data_path}")
    print(f"Output: {training_args.output_dir}")
    print("=" * 60)

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Framework3 GRPO model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((GRPOConfig, ModelConfig))
    train_args, model_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(train_args, model_args)
