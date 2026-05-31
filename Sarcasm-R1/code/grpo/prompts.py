"""Prompt templates for Sarcasm-R1 GRPO training.

Three-dimensional reasoning prompts for sarcasm detection:
  - Rhetoric: rhetorical devices analysis
  - Context: situational and background context analysis
  - Emotion: emotional contrast analysis

The model is trained to output structured reasoning in {reasoning}{answer} format.
"""

from __future__ import annotations

# System prompt: guides the model to reason through three dimensions
SYSTEM_PROMPT = (
    "You are an expert sarcasm detector. Analyze the given text by reasoning through "
    "three dimensions:\n"
    "1. Rhetoric: Identify rhetorical devices (irony, hyperbole, understatement, contrast).\n"
    "2. Context: Evaluate situational context and background knowledge mismatches.\n"
    "3. Emotion: Detect emotional contrasts between literal meaning and implied sentiment.\n\n"
    "Provide your analysis for each dimension, then conclude with your judgment.\n"
    "Format your response as:\n"
    "{Your step-by-step reasoning across the three dimensions}\n"
    "{sarcastic} or {non_sarcastic}"
)

# Simplified system prompt for shorter responses
SYSTEM_PROMPT_SHORT = (
    "Analyze the text for sarcasm. Consider rhetoric, context, and emotion. "
    "Respond with your reasoning followed by {sarcastic} or {non_sarcastic}."
)

# User prompt template for tweet/text data
USER_PROMPT_TEMPLATE = (
    'Determine if the following text is sarcastic:\n\n'
    'Text: "{text}"'
)

# User prompt template for dialogue data (MUStARD)
USER_PROMPT_MUSTARD_TEMPLATE = (
    'Given the following conversation context, determine if the target statement is sarcastic:\n\n'
    'Context:\n{context}\n\n'
    'Target statement: "{utterance}"'
)

# Answer extraction patterns
ANSWER_PATTERNS = [
    r"\{(\s*sarcastic\s*)\}",
    r"\{(\s*non_sarcastic\s*)\}",
]

VALID_ANSWERS = {"sarcastic", "non_sarcastic"}


def build_messages(text: str, context: str = "", is_dialogue: bool = False) -> list[dict]:
    """Build the message list for GRPO training.

    Returns:
        List of message dicts in the format expected by the tokenizer.
    """
    if is_dialogue and context:
        user_content = USER_PROMPT_MUSTARD_TEMPLATE.format(
            context=context, utterance=text
        )
    else:
        user_content = USER_PROMPT_TEMPLATE.format(text=text)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_user_prompt(text: str, context: str = "", is_dialogue: bool = False) -> str:
    """Build just the user prompt content."""
    if is_dialogue and context:
        return USER_PROMPT_MUSTARD_TEMPLATE.format(context=context, utterance=text)
    return USER_PROMPT_TEMPLATE.format(text=text)
