# Segmentation repair on the installed 6.6.2 baseline

Based on commit `2313fd12e1bd9b72f6db9aca17cde4359102341e`.

With content cleanup disabled, rule splitting balances contiguous whole sentences
instead of filling the final bubble with every remaining sentence. Markdown
blocks and creative excerpts remain atomic and separate from surrounding prose.
The bubble limit is soft when structural barriers require additional bubbles.
Merging preserves terminal punctuation rather than replacing it with commas.

Regex splitting uses complete match spans, retains unmatched text and supports
capturing groups without discarding content. Blank lines create candidates;
terminal punctuation clusters and closing quotes stay together.

Existing explicit LLM boundaries, functional-command exclusions, result ownership,
media ordering, history IDs, official TTS handling and long-text threshold remain
in effect. Explicit content-cleanup mode retains its legacy merge semantics.
No configuration, persona, memory or knowledge-base changes are required.

This does not rewrite an indivisible long sentence, force the model to obey a
requested number of steps, split inside a table/code block, or override the
configured long-text bypass threshold. Local tests do not prove QQ delivery.

Regression coverage includes balanced chat, table/code isolation, blank-line
explanations, punctuation clusters, capturing groups and unmatched regex tails,
alongside the existing marker, ownership, TTS, component and proactive tests.
