"""Turn-local output constraints. Never infer them from recalled memory/history."""
from __future__ import annotations

import ast
import asyncio
from copy import deepcopy
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

from astrbot.core.agent.message import TextPart

from .logging_util import get_module_logger
from .segmented_message import LLM_SEGMENT_MARKER, sanitize_llm_segment_control_tokens

logger = get_module_logger(__name__)


@dataclass(frozen=True)
class ReplyContract:
    source: str
    single_message: bool = False
    verbatim: str | None = None
    python_example: bool = False


def current_user_text(event: Any) -> str:
    if event is None or getattr(event, "private_companion_proactive_framework", False) or getattr(event, "_private_companion_external_proactive_source", ""):
        return ""
    captured = getattr(event, "_private_companion_reply_contract", None)
    if isinstance(captured, ReplyContract):
        return captured.source
    # message_str belongs to this event, unlike recalled text in req.prompt.
    text = str(getattr(event, "message_str", "") or "")
    message = getattr(event, "message_obj", None)
    message_id = getattr(message, "message_id", None)
    raw = getattr(message, "raw_message", None)
    if message_id is None and isinstance(raw, dict):
        message_id = raw.get("message_id")
    if message_id is not None and str(message_id):
        # QQ tools append this suffix to event.message_str before LLM hooks.
        # Match only this event's real ID; preserve quoted/literal IDs and leave
        # the event/provider tool metadata itself unchanged.
        text = re.sub(r"[ \t]+\[MSG_ID:" + re.escape(str(message_id)) + r"\][ \t]*$", "", text)
    return text


def reply_contract(text: str) -> ReplyContract:
    source = str(text or "")
    # A quoted occurrence of these words is not an instruction.
    directive = re.match(
        r"^\s*(?:请|麻烦|请你|你)?\s*(?:只需|只要|只)?\s*"
        r"(?:原样(?:回复|复述|输出|发送)|逐字(?:回复|复述|输出|发送)|"
        r"(?:用|以)?(?:一条|单条)(?:消息)?(?:回复|发送|输出)|不要(?:切分|分段))",
        source,
    )
    single = bool(directive) or bool(re.search(
        r"(?:[，。；;\n]|^)(?:请)?(?:不要分段|不要切分|只发一条|单条发送)[。！!\s]*$", source
    ))
    exact = None
    pairs = {"“": "”", '"': '"', "「": "」", "『": "』"}
    if directive and re.search(r"原样|逐字", directive.group()):
        remaining = source[directive.end():].strip()
        remaining = re.sub(r"^[：:]\s*", "", remaining)
        if remaining and remaining[0] in pairs and remaining.endswith(pairs[remaining[0]]):
            exact = remaining[1:-1]
    # "Directly reply: <quoted payload>" specifies the response itself. Do not
    # confuse it with a generic request to answer a question directly.
    literal_directive = re.match(
        r"^\s*(?:请你|请|麻烦|你)?\s*(?:直接|只|仅)(?:回复|回答|输出|发送)\s*[：:]\s*",
        source,
    )
    if literal_directive:
        remaining = source[literal_directive.end():].strip()
        if remaining and remaining[0] in pairs and remaining.endswith(pairs[remaining[0]]):
            exact = remaining[1:-1]
            single = True
    python_example = bool(
        re.search(r"(?<![A-Za-z0-9_])python(?![A-Za-z0-9_])", source, re.I)
        and re.search(r"(?:写|编写|给(?:我)?|提供|生成).{0,100}(?:示例|代码|程序)|(?:示例|代码|程序).{0,40}(?:写|编写|生成)", source, re.S)
        and not single
        and not re.search(r"(?:不要|无需|不用).{0,8}(?:代码|程序|示例)", source)
        and not (re.search(r"解释|讲解|分析", source) and not re.search(r"写|编写|生成|示例", source))
    )
    return ReplyContract(source, single, exact, python_example)


def has_python_code(text: str) -> bool:
    # Presence validation, not a demand that intentionally incomplete/error
    # examples must compile. Do not silently replace a supplied code example.
    if re.search(r"(?ms)^\s*(`{3,}|~{3,})(?:python|py)[ \t]*\n\S?.+?^\s*\1\s*$", text):
        return True
    candidates = re.findall(r"(?ms)^\s*(`{3,}|~{3,})(?:python|py)?[^\n]*\n(.*?)^\s*\1\s*$", text)
    bodies = [body for _, body in candidates] or [text]
    for body in bodies:
        try:
            tree = ast.parse(body.strip())
        except (SyntaxError, ValueError, RecursionError):
            continue
        # Plain quoted prose or an empty fence is not a supplied program.
        if any(not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Constant) for node in tree.body):
            return True
    return False


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def single_message_text(text: str, event: Any) -> str:
    contract = reply_contract(current_user_text(event))
    if contract.verbatim == text:
        return text
    if LLM_SEGMENT_MARKER in text or "{{split_marker}}" in text:
        return sanitize_llm_segment_control_tokens(text)
    return text


def capture_contract(event: Any) -> None:
    if not isinstance(getattr(event, "_private_companion_reply_contract", None), ReplyContract):
        setattr(event, "_private_companion_reply_contract", reply_contract(current_user_text(event)))


def finalize_request(event: Any, req: Any) -> None:
    """Reassert the current task after historical/auxiliary user content.

    Keep all configuration, history, tools and memory in place. Only this
    temporary part is replaced on repeated hooks; never share state by session.
    """
    capture_contract(event)
    if getattr(event, "_private_companion_multiuser_turn", False):
        return
    merged = getattr(event, "_private_companion_merged_turn_source", None)
    if isinstance(merged, str) and merged.strip():
        event._private_companion_reply_contract = reply_contract(merged)
    contract = event._private_companion_reply_contract
    if not contract.source.strip() or contract.source.lstrip().startswith(("/", "／")):
        return
    parts = list(getattr(req, "extra_user_content_parts", None) or [])
    parts = [p for p in parts if not getattr(p, "_private_companion_current_task", False)]
    part = TextPart(text=(
        "【本轮当前任务】\n以下引用的是本次用户消息；历史对话和召回记忆仅作背景，"
        "不要把已经回答的旧任务当作本轮任务。请直接完成此次请求。\n"
        + contract.source
    ))
    marker = getattr(part, "mark_as_temp", None)
    if callable(marker):
        part = marker()
    setattr(part, "_private_companion_current_task", True)
    parts.append(part)
    req.extra_user_content_parts = parts
    prompt = str(getattr(req, "prompt", "") or "")
    # A diagnostic fingerprint, never raw prompts or memories in the log.
    trace = f"{id(event):x}:{fingerprint(contract.source)}"
    if contract.python_example:
        event._private_companion_code_request = {
            "system_prompt": str(getattr(req, "system_prompt", "") or ""),
            "contexts": deepcopy(getattr(req, "contexts", None) or []),
            "model": getattr(req, "model", None),
            "trace": trace,
        }
    logger.info("[本轮任务] request trace=%s source_in_prompt=%s python_contract=%s", trace, contract.source.strip() in prompt, contract.python_example)


async def enforce_response(owner: Any, event: Any, resp: Any) -> None:
    contract = getattr(event, "_private_companion_reply_contract", None)
    if not isinstance(contract, ReplyContract):
        return
    text = str(getattr(resp, "completion_text", "") or "")
    if not text.strip():
        return
    if getattr(resp, "tools_call_name", None) or getattr(resp, "role", "assistant") == "tool":
        return
    if getattr(event, "private_companion_debounce_pending_merged", False):
        return
    if contract.verbatim is not None:
        resp.completion_text = contract.verbatim
        resp.result_chain = None
        return
    snapshot = getattr(event, "_private_companion_code_request", None)
    if not contract.python_example or not text.strip() or has_python_code(text) or not isinstance(snapshot, dict):
        return
    # Do not turn a legitimate refusal or clarification into a forced retry.
    if re.search(r"无法(?:帮助|提供|编写)|不能(?:帮助|提供|编写)|需要您.{0,20}(?:提供|说明|确认)|请(?:您)?先.{0,20}(?:说明|提供)", text):
        return
    if getattr(event, "_private_companion_code_repair_attempted", False):
        return
    event._private_companion_code_repair_attempted = True
    logger.warning("[本轮任务] missing_python_code trace=%s response_sha=%s", snapshot["trace"], fingerprint(text))
    getter = getattr(event, "get_extra", None)
    provider_id = getter("selected_provider") if callable(getter) else None
    try:
        if not provider_id:
            provider_id = await owner.context.get_current_chat_provider_id(str(getattr(event, "unified_msg_origin", "") or ""))
        kwargs = {
            "chat_provider_id": provider_id,
            "prompt": contract.source + "\n\n请完成以上当前请求，直接提供可运行的 Python 示例代码。不要继续回答历史中的其他任务。",
            "system_prompt": snapshot["system_prompt"],
            "contexts": snapshot["contexts"],
            # Bounded text-only repair: never repeat tools or external actions.
            "tools": None,
        }
        if snapshot.get("model"):
            kwargs["model"] = snapshot["model"]
        started = time.monotonic()
        repaired = await asyncio.wait_for(owner.context.llm_generate(**kwargs), timeout=45)
        candidate = str(getattr(repaired, "completion_text", "") or "")
        recorder = getattr(owner, "_record_external_llm_usage", None)
        if callable(recorder):
            try:
                recorder(
                    provider_id=provider_id, task="reply_code_repair",
                    prompt=kwargs["prompt"], completion=candidate,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    success=has_python_code(candidate), resp=repaired,
                    session_id=str(getattr(event, "unified_msg_origin", "") or ""),
                )
            except Exception:
                pass  # Accounting must not discard a valid repaired response.
        if has_python_code(candidate) and not getattr(repaired, "tools_call_name", None):
            resp.completion_text = candidate
            resp.result_chain = None
            logger.info("[本轮任务] python_repair_ok trace=%s", snapshot["trace"])
            return
    except Exception as exc:
        logger.warning("[本轮任务] python_repair_failed trace=%s error_type=%s", snapshot["trace"], type(exc).__name__)
    # Never deliver the known wrong-topic response as though the task succeeded.
    resp.completion_text = "这次没能正确生成您要的 Python 示例，请再试一次。"
    resp.result_chain = None
