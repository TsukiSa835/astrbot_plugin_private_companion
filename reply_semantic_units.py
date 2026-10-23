"""Detect explicit sequential steps before sentence balancing."""
from __future__ import annotations

import re

from .markdown_segment_guard import protect_markdown_blocks


_CN = {value: index for index, value in enumerate(
    ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十"), 1
)}
_STEPS = re.compile(
    r"(?:^|(?<=[\n。！？!?]))[ \t]*(?P<label>第(?P<cn>[一二三四五六七八九十]|\d{1,2})(?:步|点|项)?[，、：:]|步骤[ \t]*(?P<step>\d{1,2}|[一二三四五六七八九十])[，、：:])",
    re.M,
)
_NUMBERED = re.compile(r"^(?P<indent> {0,3})(?P<number>\d{1,2})[.)、][ \t]+", re.M)


def explicit_step_units(text: str) -> list[str] | None:
    """Require an unambiguous 1,2,... sequence; preserve each step's body.

    Ignore fenced code and quoted dialogue, including examples which merely
    mention step labels. A hard boundary may exceed the ordinary bubble budget.
    """
    masked = list(text)
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        match = re.match(r" {0,3}(`{3,}|~{3,})", line)
        if fence:
            masked[offset:offset + len(line)] = ["\n" if c == "\n" else " " for c in line]
            if match and match.group()[0:].strip()[0] == fence[0] and len(match.group().strip()) >= len(fence):
                fence = None
        elif match:
            fence = match.group().strip()
            masked[offset:offset + len(line)] = ["\n" if c == "\n" else " " for c in line]
        offset += len(line)
    # Protect complete quoted spans; unmatched quotes do not hide the rest.
    safe = "".join(masked)
    for pattern in (r'"[^"\n]*"', r"“[^”]*”", r"「[^」]*」", r"『[^』]*』", r"`[^`\n]*`"):
        for match in re.finditer(pattern, safe):
            masked[match.start():match.end()] = " " * (match.end() - match.start())
    safe = "".join(masked)
    matches = list(_STEPS.finditer(safe))
    numbered = False
    if not matches:
        matches = list(_NUMBERED.finditer(safe))
        numbered = True
        if matches:
            indent = min(len(m.group("indent")) for m in matches)
            matches = [m for m in matches if len(m.group("indent")) == indent]
    if len(matches) < 2:
        return None
    numbers = []
    for match in matches:
        value = match.group("number") if numbered else (match.group("cn") or match.group("step"))
        numbers.append(int(value) if value.isdigit() else _CN[value])
    if numbers != list(range(1, len(numbers) + 1)):
        return None
    starts = [m.start() for m in matches]
    units = [text[:starts[0]].strip()] if text[:starts[0]].strip() else []
    units.extend(text[start:end].strip() for start, end in zip(starts, starts[1:] + [len(text)]))
    # The final step ends at an independent paragraph, rather than owning the
    # entire remainder of the reply. Protect nested Markdown first so blank
    # lines inside code, tables or a list continuation remain untouched.
    last = protect_markdown_blocks(units[-1])
    tail_parts = re.split(r"\r?\n[ \t]*\r?\n+", last.protected_text)
    units[-1:] = [last.restore(part).strip() for part in tail_parts if part.strip()]
    return [unit for unit in units if unit]
