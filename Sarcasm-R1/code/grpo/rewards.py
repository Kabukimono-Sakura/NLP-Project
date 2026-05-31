"""Reward functions for Sarcasm-R1 GRPO training.

Reward functions evaluate model outputs during GRPO training:
  - accuracy_reward: Correctness of the final sarcasm judgment
  - format_reasoning_reward: Quality and structure of the reasoning
  - dimension_coverage_reward: Coverage of the three analysis dimensions
  - combined_reward: Weighted combination of all rewards
"""

from __future__ import annotations
import re
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract the predicted answer from model output.

    Looks for {sarcastic} or {non_sarcastic} in the response.
    Falls back to scanning for the last occurrence of the keywords.
    """
    # Primary: look for {answer} pattern
    brace_matches = re.findall(r"\{(\s*(?:non_)?sarcastic\s*)\}", text)
    if brace_matches:
        return brace_matches[-1].strip()

    # Fallback: look for the answer word at the end of the text
    last_200 = text[-200:] if len(text) > 200 else text
    if re.search(r"\bnon.sarcastic\b", last_200, re.IGNORECASE):
        return "non_sarcastic"
    if re.search(r"\bsarcastic\b", last_200, re.IGNORECASE):
        return "sarcastic"

    return None


def parse_reasoning_content(text: str) -> dict:
    """Parse model output into reasoning content and answer."""
    answer = extract_answer(text)

    # Split text into reasoning sections
    reasoning = text
    if answer:
        # Remove the answer part to get pure reasoning
        answer_pattern = r"\{(\s*" + re.escape(answer) + r"\s*)\}"
        reasoning = re.sub(answer_pattern, "", text).strip()

    return {
        "thinking_content": reasoning,
        "response": answer or "",
    }


def get_completion_content(completion: list[dict]) -> str:
    """Extract text content from a completion."""
    return completion[0]["content"]


def parse_responses(completions: list[list[dict]]) -> list[dict]:
    """Parse all completions into reasoning + answer dicts."""
    return [parse_reasoning_content(get_completion_content(c)) for c in completions]


# ---------------------------------------------------------------------------
# Reward Functions (signature: prompts, completions, answer, **kwargs -> list[float])
# All reward functions follow the TRL GRPOTrainer convention.
# ---------------------------------------------------------------------------

def accuracy_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward correct sarcasm judgment. +25 for correct, -10 for wrong."""
    parsed = parse_responses(completions)
    return [25.0 if r["response"] == a else -10.0 for r, a in zip(parsed, answer)]


def accuracy_reward_log(prompts, completions, answer, **kwargs):
    """Log reward breakdown. Returns zeros so it doesn't affect training."""
    import json
    rewards = {
        "accuracy": accuracy_reward(prompts, completions, answer),
        "format_reasoning": format_reasoning_reward(prompts, completions, answer),
        "dimension_coverage": dimension_coverage_reward(prompts, completions, answer),
        "combined": combined_reward(prompts, completions, answer),
    }

    example_response = get_completion_content(completions[0])
    example_parsed = parse_reasoning_content(example_response)
    example_prompt = prompts[0][-1]["content"] if prompts else ""

    print(
        f"{'-' * 50}\n"
        f"Example prompt:\n{example_prompt}\n"
        f"{'-' * 10}\n"
        f"Example response:\n{example_response[:300]}...\n"
        f"{'-' * 10}\n"
        f"Example answer: {answer[0]}\n"
        f"Predicted: {example_parsed['response']}\n"
        f"Correct: {example_parsed['response'] == answer[0]}\n"
        f"{'-' * 10}\n"
        f"Rewards:\n{json.dumps(rewards, indent=2)}"
    )

    # Return zeros — this function is only for logging, not training signal
    return [0.0] * len(answer)


def format_reasoning_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward structured reasoning. Reduced scale to avoid dominating accuracy."""
    parsed = parse_responses(completions)
    rewards = []

    for r, a in zip(parsed, answer):
        score = 0.0
        thinking = r["thinking_content"]

        # Reward having substantial reasoning (reduced scale)
        if len(thinking) >= 20:
            score += 0.5
        if len(thinking) >= 60:
            score += 0.5
        if len(thinking) >= 120:
            score += 0.5

        # Penalty for no reasoning
        if thinking.strip() == "":
            score -= 2.0
        else:
            score += 0.5

        # Reward for using the answer format {answer}
        full_text = get_completion_content(completions[parsed.index(r)])
        if re.search(r"\{(non_)?sarcastic\}", full_text):
            score += 1.0

        rewards.append(score)

    return rewards


def dimension_coverage_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward coverage of the three analysis dimensions: Rhetoric, Context, Emotion."""
    rewards = []
    dimensions = [
        (r"\b(rhetoric|rhetorical)\b", "Rhetoric"),
        (r"\b(context|situational|background)\b", "Context"),
        (r"\b(emotion|emotional|sentiment|feeling)\b", "Emotion"),
    ]

    for completion in completions:
        text = get_completion_content(completion).lower()
        score = 0.0

        for pattern, name in dimensions:
            if re.search(pattern, text):
                score += 2.0

        # Bonus for covering all three dimensions
        covered = sum(1 for pattern, _ in dimensions if re.search(pattern, text))
        if covered == 3:
            score += 3.0

        rewards.append(score)

    return rewards


def combined_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Weighted combination: accuracy dominates.

    Weights:
      - accuracy: 1.0 (primary signal)
      - format_reasoning: 0.1
      - dimension_coverage: 0.2
    """
    acc = accuracy_reward(prompts, completions, answer)
    fmt = format_reasoning_reward(prompts, completions, answer)
    dim = dimension_coverage_reward(prompts, completions, answer)

    return [
        a + 0.1 * f + 0.2 * d
        for a, f, d in zip(acc, fmt, dim)
    ]


# Default reward functions used in GRPO training
REWARD_FUNCS = [
    combined_reward,
    accuracy_reward_log,
]
