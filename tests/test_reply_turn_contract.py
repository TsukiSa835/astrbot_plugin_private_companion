from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot.core.agent.message import TextPart
from astrbot_plugin_private_companion.main import PrivateCompanionPlugin
from astrbot_plugin_private_companion.reply_semantic_units import explicit_step_units
from astrbot_plugin_private_companion.reply_turn_contract import (
    capture_contract, enforce_response, finalize_request, has_python_code, reply_contract,
)
from astrbot_plugin_private_companion.segmented_message import LLM_SEGMENT_MARKER
from test_markdown_safe_segmenting import _Harness


def event(text):
    return SimpleNamespace(message_str=text, unified_msg_origin="test:FriendMessage:user")


@pytest.mark.parametrize("joiner", ["", " ", "\n\n"])
def test_five_prose_steps_remain_five_whole_bubbles(joiner):
    steps = [f"第{n}，整理这一类书籍。核对这一步的两句话。" for n in "一二三四五"]
    source = joiner.join(steps)
    harness = _Harness()
    harness.segmented_proactive_max_segments = 3
    harness.segmented_proactive_min_segment_chars = 8
    assert harness._split_proactive_text(source) == steps


def test_step_boundaries_outrank_model_boundaries_inside_a_step():
    harness = _Harness()
    source = f"第一，整理书籍。\n{LLM_SEGMENT_MARKER}\n核对分类。第二，安放书籍。核对位置。"
    result = PrivateCompanionPlugin._split_llm_controlled_text_buffers_for_event(harness, event("给我两步"), [source])
    assert result == [["第一，整理书籍。\n核对分类。", "第二，安放书籍。核对位置。"]]


def test_numbered_steps_keep_nested_code_and_subitems():
    source = "1. 第一步说明。\n\n   ```python\n   print('第一，第二，')\n   ```\n\n2. 第二步说明。\n   - 子项一\n   - 子项二\n\n3. 第三步说明。"
    units = explicit_step_units(source)
    assert len(units) == 3
    assert "print('第一，第二，')" in units[0]
    assert "子项二" in units[1]


@pytest.mark.parametrize("source", [
    "这是第一本书。那是第二本书。",
    "他说：“第一，准备。第二，执行。”我听见了。",
    "```text\n第一，准备。第二，执行。\n```",
    "第一，准备。第三，执行。",
])
def test_non_step_content_is_not_reclassified(source):
    assert explicit_step_units(source) is None


def test_verbatim_bypasses_cleanup_replacement_and_model_marker_splitting():
    source = "真的吗？！原来如此......我明白了。这就是所谓的「情感」？！"
    request = event(f"请原样回复：“{source}”")
    harness = _Harness()
    harness.enable_segmented_proactive_content_cleanup = True
    harness.enable_segmented_proactive_content_replacement = True
    harness.segmented_proactive_content_replacements = ["情感 => 测试"]
    assert harness._split_proactive_text(source, event=request) == [source]
    assert PrivateCompanionPlugin._split_llm_controlled_text_buffers_for_event(harness, request, [source]) == [[source]]
    assert not reply_contract("她说“请原样回复”是什么意思？").single_message


def test_single_message_constraint_removes_model_boundaries_without_leaking_tokens():
    harness = _Harness()
    e = event("请用一条消息回复")
    text = f"第一句话。\n{LLM_SEGMENT_MARKER}\n第二句话。"
    result = PrivateCompanionPlugin._split_llm_controlled_text_buffers_for_event(harness, e, [text])
    assert len(result[0]) == 1
    assert LLM_SEGMENT_MARKER not in result[0][0]
    assert "第一句话。" in result[0][0] and "第二句话。" in result[0][0]


@pytest.mark.parametrize("text,expected", [
    ("写一段带注释，空行和中文标点字符串的Python示例", True),
    ("请给我一个 Python 示例", True),
    ("请解释 Python 代码里的缩进，不用代码示例", False),
    ("给我讲解Python代码语法", False),
    ("写一段Pythonic风格的说明", False),
])
def test_narrow_python_artifact_intent(text, expected):
    assert reply_contract(text).python_example == expected


def request():
    return SimpleNamespace(prompt="写一个Python示例", contexts=[{"role": "assistant", "content": "旧任务"}], system_prompt="角色保持独立判断", extra_user_content_parts=[TextPart(text="召回背景")], model="test-model")


def test_current_task_tail_is_temporary_idempotent_and_event_local():
    a, b = event("写一段Python示例"), event("只聊阅读")
    req = request()
    original_contexts = list(req.contexts)
    finalize_request(a, req)
    finalize_request(a, req)
    assert len(req.extra_user_content_parts) == 2
    assert req.extra_user_content_parts[-1]._no_save
    assert req.extra_user_content_parts[-1].text.endswith(a.message_str)
    assert req.contexts == original_contexts
    assert req.system_prompt == "角色保持独立判断"
    assert not hasattr(b, "_private_companion_code_request")


def test_merged_followup_is_part_of_current_task():
    e = event("写一段Python示例")
    capture_contract(e)
    e._private_companion_merged_turn_source = "写一段Python示例\n请使用中文注释，并保留空行"
    req = request()
    finalize_request(e, req)
    assert req.extra_user_content_parts[-1].text.endswith(e._private_companion_merged_turn_source)


def test_multiuser_turn_does_not_become_a_single_users_task():
    e = event("写一段Python示例")
    e._private_companion_multiuser_turn = True
    req = request()
    finalize_request(e, req)
    assert len(req.extra_user_content_parts) == 1
    assert not hasattr(e, "_private_companion_code_request")


def test_external_proactive_event_is_not_reanchored_to_old_inbound_text():
    e = event("写一段Python示例")
    e._private_companion_external_proactive_source = "proactive_chat"
    req = request()
    finalize_request(e, req)
    assert len(req.extra_user_content_parts) == 1
    assert not hasattr(e, "_private_companion_code_request")


@pytest.mark.asyncio
async def test_verbatim_response_replaces_generation_but_never_revives_discarded_response():
    e = event('请原样回复：“  您好。再见。  ”')
    capture_contract(e)
    resp = SimpleNamespace(completion_text="您好。", result_chain="old")
    await enforce_response(None, e, resp)
    assert resp.completion_text == "  您好。再见。  "
    assert resp.result_chain is None
    resp.completion_text = ""
    await enforce_response(None, e, resp)
    assert resp.completion_text == ""


@pytest.mark.asyncio
async def test_wrong_topic_python_response_repaired_once_without_tools():
    e = event("写一段带注释和空行的Python示例")
    finalize_request(e, request())
    code = '```python\n# 说明\nprint("您好。")\n```'
    context = SimpleNamespace(get_current_chat_provider_id=AsyncMock(return_value="selected"), llm_generate=AsyncMock(return_value=SimpleNamespace(completion_text=code)))
    owner = SimpleNamespace(context=context)
    resp = SimpleNamespace(completion_text="第一，按题材整理书籍。第二，按颜色排列。", result_chain="old")
    await enforce_response(owner, e, resp)
    assert resp.completion_text == code
    assert resp.result_chain is None
    kwargs = context.llm_generate.call_args.kwargs
    assert kwargs["tools"] is None
    assert kwargs["system_prompt"] == "角色保持独立判断"
    assert kwargs["prompt"].startswith(e.message_str)
    assert kwargs["model"] == "test-model"
    await enforce_response(owner, e, resp)
    context.llm_generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_repair_does_not_claim_old_topic_succeeded():
    e = event("写一段Python示例")
    finalize_request(e, request())
    context = SimpleNamespace(get_current_chat_provider_id=AsyncMock(return_value="selected"), llm_generate=AsyncMock(side_effect=RuntimeError("offline")))
    resp = SimpleNamespace(completion_text="第一，整理书架。", result_chain="old")
    await enforce_response(SimpleNamespace(context=context), e, resp)
    assert "没能正确生成" in resp.completion_text
    assert resp.result_chain is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text,tools", [("```python\nprint('ok')\n```", None), ("正在调用工具", ["tool"]), ("我无法提供这类代码。", None), ("请先说明您需要处理的数据格式。", None)])
async def test_valid_code_tool_turns_and_clarifications_are_not_retried(text, tools):
    e = event("写一段Python示例")
    finalize_request(e, request())
    generate = AsyncMock()
    resp = SimpleNamespace(completion_text=text, tools_call_name=tools)
    await enforce_response(SimpleNamespace(context=SimpleNamespace(llm_generate=generate)), e, resp)
    generate.assert_not_awaited()
    assert resp.completion_text == text
