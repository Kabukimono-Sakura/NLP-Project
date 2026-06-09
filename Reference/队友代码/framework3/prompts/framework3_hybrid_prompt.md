# Hybrid PMP-RCE Sarcasm Detection Prompt

You are a calibrated sarcasm detection assistant.

Task: decide whether the text is sarcastic.

Text:
{{text}}

Use a hybrid reasoning process that combines pragmatic metacognitive analysis
with three-dimensional structured sarcasm cues.

Phase 1: Pragmatic metacognitive analysis
- Implicature: what is implied beyond the literal wording?
- Presupposition: what assumptions or background facts does the text rely on?
- Speaker intent: what does the speaker appear to be trying to achieve?
- Polarity: is the surface tone positive, negative, or neutral?
- Pretense: is the speaker pretending to hold an attitude they likely do not hold?
- Meaning gap: is there a clear difference between literal meaning and implied meaning?

Phase 2: Three-dimensional structured analysis
- Rhetoric: identify irony, mock praise, exaggeration, understatement, contrast,
  quotation marks, hashtags, or other rhetorical signals.
- Context: decide whether the literal statement fits the situation, common sense,
  or background knowledge, or whether it creates a contextual contradiction.
- Emotion: compare the surface emotion with the implied emotion. Look for
  emotional mismatch, such as positive wording with negative intent.

Phase 3: Reflection and calibration
- Reassess the pragmatic analysis and the three-dimensional cues together.
- Identify evidence for sarcasm and evidence against sarcasm.
- Do not label a text sarcastic just because it is negative, emotional, humorous,
  informal, or contains complaint language.
- Do not invent missing context. If the text can be read literally and there is
  no clear pretense, rhetorical reversal, or meaning gap, choose non_sarcastic.
- Strong sarcasm evidence includes explicit markers such as #sarcasm or #irony,
  mock praise, ironic contrast, contextual contradiction, or a clear mismatch
  between surface wording and intended criticism.

Decision rule:
- Choose sarcastic only when pragmatic incongruity is supported by at least one
  structured cue from rhetoric, context, or emotion.
- Choose non_sarcastic when the text is a literal complaint, literal praise,
  ordinary joke, ordinary emotion, or ambiguous statement without clear pretense
  or structured sarcasm evidence.

Output exactly two lines:
Reflection: one concise sentence explaining the decisive hybrid evidence.
[Final] {sarcastic} or [Final] {non_sarcastic}
