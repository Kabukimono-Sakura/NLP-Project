#!/usr/bin/env python
"""Generate structured reasoning annotations for sarcasm detection training data.

Uses an LLM (via vLLM or API) to generate three-dimensional reasoning chains
(Rhetoric, Context, Emotion) for each training example. These annotations
can be used to create high-quality SFT data before GRPO training, or as
reference reasoning for reward model training.

Usage:
  python generate_reasoning.py --input ../../data/processed/semeval_train.csv --output ../../data/processed/semeval_reasoning.jsonl
  python generate_reasoning.py --input ../../data/processed/semeval_train.csv --output ../../data/processed/semeval_reasoning.jsonl --model Qwen/Qwen2.5-7B-Instruct --use_vllm
  python generate_reasoning.py --input ../../data/processed/semeval_train.csv --output ../../data/processed/semeval_reasoning.jsonl --use_api --api_base http://localhost:8000/v1
"""

from __future__ import annotations
import argparse
import json
import os
import re

REASONING_PROMPT_TEMPLATE = """You are analyzing whether the following text is sarcastic.

Text: "{text}"

Analyze this text across three dimensions:

1. Rhetoric Analysis: What rhetorical devices are used? Look for irony, hyperbole, understatement, or contrast between literal and implied meaning.

2. Context Analysis: What is the situational context? What background knowledge is needed? Is there a mismatch between the statement and the situation?

3. Emotion Analysis: What emotional contrast exists? Is there a gap between the literal sentiment and the implied emotional state?

After your analysis, conclude with your judgment: {{sarcastic}} or {{non_sarcastic}}"""

REASONING_PROMPT_MUSTARD = """You are analyzing whether the following statement from a TV show/movie dialogue is sarcastic.

Conversation context:
{context}

Target statement: "{utterance}"

Analyze this statement across three dimensions:

1. Rhetoric Analysis: What rhetorical devices are used? Look for irony, hyperbole, understatement, or contrast between literal and implied meaning.

2. Context Analysis: What is the situational context? What background knowledge is needed? Is there a mismatch between the statement and the situation?

3. Emotion Analysis: What emotional contrast exists? Is there a gap between the literal sentiment and the implied emotional state?

After your analysis, conclude with your judgment: {{sarcastic}} or {{non_sarcastic}}"""


def build_reasoning_prompt(text: str, context: str = "", is_mustard: bool = False) -> str:
    if is_mustard and context:
        return REASONING_PROMPT_MUSTARD.format(context=context, utterance=text)
    return REASONING_PROMPT_TEMPLATE.format(text=text)


def parse_reasoning_response(response: str) -> dict:
    """Parse the LLM response into structured reasoning + answer."""
    # Try to extract the final judgment
    answer = None
    if re.search(r"\{sarcastic\}", response):
        answer = "sarcastic"
    elif re.search(r"\{non_sarcastic\}", response):
        answer = "non_sarcastic"

    # Fallback: look for the answer in the text
    if answer is None:
        if re.search(r"\b(sarcastic)\b", response, re.IGNORECASE):
            # Check if it says "non_sarcastic" or "not sarcastic" first
            if re.search(r"\b(not sarcastic|non.sarcastic)\b", response, re.IGNORECASE):
                answer = "non_sarcastic"
            else:
                answer = "sarcastic"
        else:
            answer = "non_sarcastic"

    return {
        "full_response": response,
        "predicted_answer": answer,
    }


def generate_with_vllm(data: list[dict], model_name: str, batch_size: int = 8) -> list[dict]:
    """Generate reasoning using vLLM for fast inference."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.7, max_tokens=512, top_p=0.9)

    prompts = [item["prompt_text"] for item in data]
    outputs = llm.generate(prompts, sampling)

    results = []
    for item, output in zip(data, outputs):
        response = output.outputs[0].text
        parsed = parse_reasoning_response(response)
        results.append({
            **item,
            "reasoning": parsed["full_response"],
            "predicted_answer": parsed["predicted_answer"],
        })
    return results


def generate_with_api(data: list[dict], model_name: str, api_base: str, batch_size: int = 4) -> list[dict]:
    """Generate reasoning using OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key="dummy")
    results = []

    for item in data:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": item["prompt_text"]}],
                temperature=0.7,
                max_tokens=512,
            )
            response = resp.choices[0].message.content
            parsed = parse_reasoning_response(response)
            results.append({
                **item,
                "reasoning": parsed["full_response"],
                "predicted_answer": parsed["predicted_answer"],
            })
        except Exception as e:
            print(f"  Error generating for item: {e}")
            results.append({**item, "reasoning": "", "predicted_answer": "error"})
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning annotations")
    parser.add_argument("--input", type=str, required=True, help="Input processed CSV")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for generation")
    parser.add_argument("--use_api", action="store_true", help="Use OpenAI-compatible API")
    parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=-1, help="Max number of samples to process")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.input)
    if args.limit > 0:
        df = df.head(args.limit)

    # Build prompts
    data = []
    for _, row in df.iterrows():
        text = row.get("source_text", row.get("source_utterance", row.get("question", "")))
        context = row.get("source_context", "")
        is_mustard = "mustard" in args.input.lower()

        prompt_text = build_reasoning_prompt(text, context, is_mustard)
        data.append({
            "prompt_text": prompt_text,
            "text": text,
            "ground_truth": row["answer"],
        })

    print(f"Generating reasoning for {len(data)} samples...")

    if args.use_vllm:
        results = generate_with_vllm(data, args.model, args.batch_size)
    elif args.use_api:
        results = generate_with_api(data, args.model, args.api_base, args.batch_size)
    else:
        print("Please specify --use_vllm or --use_api")
        return

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Print stats
    correct = sum(1 for r in results if r["predicted_answer"] == r["ground_truth"])
    print(f"\nResults saved to {args.output}")
    print(f"Agreement with ground truth: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
