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

Existing explicit LLM boundaries for ordinary prose, functional-command exclusions, result ownership,
media ordering, history IDs, official TTS handling and long-text threshold remain
in effect. Explicit content-cleanup mode retains its legacy merge semantics.
No configuration, persona, memory or knowledge-base changes are required.

This does not rewrite an indivisible long sentence, force the model to obey a
requested number of steps, split inside a table/code block, or override the
configured long-text bypass threshold. Local tests do not prove QQ delivery.

Regression coverage includes balanced chat, table/code isolation, blank-line
explanations, punctuation clusters, capturing groups and unmatched regex tails,
alongside the existing marker, ownership, TTS, component and proactive tests.

## Second acceptance round

Explicit sequential steps now precede sentence balancing: complete steps remain
whole, even when this exceeds the ordinary bubble budget or a model boundary
appears inside a step. Fenced code and quoted mentions are excluded from step
detection. Clear single-message/verbatim requests bypass splitting and optional
text transformations for that turn; delimited verbatim payloads are restored
before delivery, while reserved model boundaries are removed for normal
single-message replies.

Each inbound request also gets a temporary current-task reminder after auxiliary
history/memory content. Same-user merged followups are retained; multiuser and
external proactive tasks are not re-anchored to one old inbound message.

For explicit requests to produce a Python example, a nonempty final response
without Python code can be regenerated once, using the selected provider/model,
the original system prompt and history, and no tools. Valid code, tool turns,
refusals and clarification requests are excluded. Failed repair returns an
honest failure message rather than the known off-topic answer. Normal successful
responses incur no extra model call; this recovery has a 45-second timeout.
Request/response diagnostic logs contain fingerprints, not full prompt content.

The observed wrong-topic response existed before segmentation. Its unique
original cause cannot be established from the supplied logs; the task reminder,
artifact check and bounded recovery are defensive fixes, not proof of that cause.
Persona/style prompts, settings and memory scopes are unchanged.

## Third acceptance round

Quoted payloads following an explicit `directly reply:`/`only reply:` directive
also use the literal single-message contract. A real QQ-tools `[MSG_ID:...]`
suffix is ignored only for constraint detection and only when it matches this
event's actual message ID; the event/tool metadata and literal IDs stay intact.

An independent blank-line paragraph following the last step now gets its own
bubble. Markdown protection is applied first so internal code/list/table blank
lines do not become accidental split points. Regression coverage includes the
runtime metadata suffix, direct-answer false positives, CRLF paragraphs and a
final numbered step containing code followed by a closing question.
