"""Contiguous, punctuation-preserving chat grouping; structural blocks are barriers."""
from __future__ import annotations

import re
from collections.abc import Callable

from .markdown_segment_guard import MARKDOWN_BLOCK_TOKEN_PATTERN


def balanced_segments(
    raw: list[str], *, limit: int, minimum: int,
    length: Callable[[str], int], atomic: Callable[[str], bool],
) -> list[str]:
    """Balance whole sentences without spending the chat budget on Markdown.

    A protected block (or a creative excerpt) occupies its own bubble. The
    configured limit is a soft limit across prose; structural barriers can
    require additional bubbles rather than swallowing the following prose.
    """
    pieces = [part.strip() for item in raw
              for part in re.split(f"({MARKDOWN_BLOCK_TOKEN_PATTERN})", item)
              if part.strip()]

    def join(items: list[str]) -> str:
        result = ""
        for item in items:
            # Never replace an author's terminal punctuation with a comma.
            separator = " " if result and result[-1:].isascii() and (result[-1:].isalnum() or result[-1:] in '.!?') and item[:1].isascii() and item[:1].isalnum() else ""
            result += separator + item
        return result

    def group(items: list[str], budget: int) -> list[str]:
        if not items:
            return []
        compact = []
        pending = ""
        for item in items:
            item = join([pending, item]) if pending else item
            pending = ""
            body = re.sub(r"[。！？!?…~～,.，、\s]+$", "", item)
            if length(item) < minimum or body in {"嗯", "唔", "啊", "呀", "诶", "欸", "哈哈", "哈", "还有"} or re.fullmatch(r"[.。！？!?…～~]+", item):
                pending = item
            else:
                compact.append(item)
        if pending:
            if compact:
                compact[-1] = join([compact[-1], pending])
            else:
                compact.append(pending)
        items = compact
        sizes = [length(item) for item in items]
        count = min(len(items), max(1, budget), max(1, sum(sizes) // max(1, minimum)))
        prefix = [0]
        for size in sizes:
            prefix.append(prefix[-1] + size)
        target = prefix[-1] / count
        # O(limit * sentence_count^2), bounded by the existing 1024-char gate.
        costs = {(0, 0): (0.0, [])}
        for k in range(1, count + 1):
            for end in range(k, len(items) + 1):
                options = []
                for start in range(k - 1, end):
                    previous = costs.get((k - 1, start))
                    if previous is None:
                        continue
                    size = prefix[end] - prefix[start]
                    penalty = (size - target) ** 2 + max(0, minimum - size) ** 2 * 4
                    options.append((previous[0] + penalty, previous[1] + [end]))
                costs[k, end] = min(options, key=lambda option: option[0])
        ends = costs[count, len(items)][1]
        result, start = [], 0
        for end in ends:
            result.append(join(items[start:end]))
            start = end
        return result

    runs: list[list[str] | str] = []
    pending: list[str] = []
    for piece in pieces:
        if re.fullmatch(MARKDOWN_BLOCK_TOKEN_PATTERN, piece) or atomic(piece):
            if pending:
                runs.append(pending)
                pending = []
            runs.append(piece)
        else:
            pending.append(piece)
    if pending:
        runs.append(pending)
    prose_runs = [run for run in runs if isinstance(run, list)]
    remaining = max(limit - sum(isinstance(run, str) for run in runs), len(prose_runs))
    result = []
    for run in runs:
        if isinstance(run, str):
            result.append(run)
            continue
        others = len(prose_runs) - 1
        budget = max(1, round(remaining * sum(length(x) for x in run) / max(1, sum(length(x) for r in prose_runs for x in r))))
        budget = min(budget, remaining - others)
        grouped = group(run, budget)
        result.extend(grouped)
        remaining -= len(grouped)
        prose_runs.pop(0)
    return result
