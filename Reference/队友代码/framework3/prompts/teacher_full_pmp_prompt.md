# Pragmatic Metacognitive Prompting Sarcasm Detection

You are a calibrated sarcasm detection assistant.

Task: decide whether the text is sarcastic.

Text:
{{text}}

Use Pragmatic Metacognitive Prompting in two short phases.

Phase 1: Preliminary pragmatic analysis
- Implicature: what is implied beyond the literal wording?
- Presupposition: what assumptions or background facts does the text rely on?
- Speaker intent: what does the speaker appear to be trying to achieve?
- Polarity: is the surface tone positive, negative, or neutral?
- Pretense: is the speaker pretending to hold an attitude they likely do not hold?
- Meaning gap: is there a clear difference between literal meaning and implied meaning?

Phase 2: Reflection and calibration
- Reassess the preliminary analysis.
- Identify evidence for sarcasm and evidence against sarcasm.
- Do not label a text sarcastic just because it is negative, emotional, humorous, informal, or contains complaint language.
- Do not invent missing context. If the text can be read literally and there is no clear pretense or meaning gap, choose non_sarcastic.
- Strong sarcasm evidence includes explicit markers such as #sarcasm or #irony, mock praise, ironic contrast, or a clear contradiction between surface wording and intended criticism.

Decision rule:
- Choose sarcastic only when there is clear pragmatic incongruity: the speaker's implied attitude conflicts with the literal wording or uses pretense/mock attitude.
- Choose non_sarcastic when the text is a literal complaint, literal praise, ordinary joke, ordinary emotion, or ambiguous statement without clear pretense.

Output exactly two lines:
Reflection: one concise sentence explaining the decisive pragmatic evidence.
[Final] {sarcastic} or [Final] {non_sarcastic}
