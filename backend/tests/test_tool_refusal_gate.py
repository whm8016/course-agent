"""工具层拒答硬闸（任务9）：ToolResult.success=False 不再当普通结果喂模型。

根因：rag 空命中返回 success=False +「未检索到」文案，但 dispatch_tool_calls
不读 success——内容照常作为 role=tool 消息进 messages，status 也记 ok。拒答
只靠 chat.yaml 提示词软约束，学生若开了 web_search 模型可绕过课程资料作答。

修法：
1. success=False → status=error（metrics/日志可观测）；
2. rag 空命中且本轮 LLM 未调 web_search → tool content 追加 [系统指令] 硬约束，
   禁止通用知识硬答，要求显式说明「课程资料未覆盖」。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from core.agent.tool_protocol import ToolResult
from core.agentic.tool_dispatch import dispatch_tool_calls
from core.agentic.types import ToolCall


def _tc(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(
        id=f"call_{name}", name=name, arguments=args or {"query": "q"},
        arguments_str="{}",
    )


def _stream() -> MagicMock:
    s = MagicMock()
    s.emit = AsyncMock()
    return s


def _find_tool_msg(outcome, name: str) -> dict:
    return next(m for m in outcome.tool_messages if m.get("name") == name)


async def test_rag_failure_marks_error_status(monkeypatch):
    """success=False → observe_tool_call 记 error（不再是 ok）。"""
    statuses: list[str] = []

    async def _fake_execute(tool_name, **kw):
        return ToolResult(content="（知识库中未检索到与该问题相关的内容。）", success=False)

    def _fake_observe(**kw):
        statuses.append(kw["status"])

    monkeypatch.setattr("core.agent.tool_registry.execute_tool", _fake_execute)
    monkeypatch.setattr("core.agentic.tool_dispatch.observe_tool_call", _fake_observe)

    outcome = await dispatch_tool_calls(
        [_tc("rag")], course_id="c1", enabled_tools=["rag"], stream=_stream(),
    )
    assert statuses == ["error"]
    msg = _find_tool_msg(outcome, "rag")
    assert "[系统指令]" in msg["content"]


async def test_rag_empty_without_web_search_appends_hard_refusal(monkeypatch):
    """rag 空命中 + 本轮无 web_search → tool content 追加硬拒答指令。"""
    async def _fake_execute(tool_name, **kw):
        if tool_name == "rag":
            return ToolResult(content="（知识库中未检索到与该问题相关的内容。）", success=False)
        return ToolResult(content="搜索结果")

    monkeypatch.setattr("core.agent.tool_registry.execute_tool", _fake_execute)

    outcome = await dispatch_tool_calls(
        [_tc("rag")], course_id="c1", enabled_tools=["rag"], stream=_stream(),
    )
    msg = _find_tool_msg(outcome, "rag")
    assert "未检索到" in msg["content"]
    assert "禁止使用通用知识" in msg["content"]
    assert "暂无法确认" in msg["content"]


async def test_rag_empty_with_web_search_no_hard_refusal(monkeypatch):
    """同轮 LLM 调了 web_search → 不加硬拒答（联网结果可作答依据）。"""
    async def _fake_execute(tool_name, **kw):
        if tool_name == "rag":
            return ToolResult(content="（知识库中未检索到与该问题相关的内容。）", success=False)
        return ToolResult(content="搜索命中：xxx")

    monkeypatch.setattr("core.agent.tool_registry.execute_tool", _fake_execute)

    outcome = await dispatch_tool_calls(
        [_tc("rag"), _tc("web_search")],
        course_id="c1", enabled_tools=["rag", "web_search"], stream=_stream(),
    )
    rag_msg = _find_tool_msg(outcome, "rag")
    assert "[系统指令]" not in rag_msg["content"]
    ws_msg = _find_tool_msg(outcome, "web_search")
    assert ws_msg["content"] == "搜索命中：xxx"
    assert ws_msg["content"] != ""  # web_search 正常内容不受影响


async def test_rag_success_unchanged(monkeypatch):
    """success=True 的正常结果：无指令、status ok（原行为不变）。"""
    seen: list[str] = []

    async def _fake_execute(tool_name, **kw):
        return ToolResult(content="[证据1] KVL：回路电压代数和为零。", success=True)

    def _fake_observe(**kw):
        seen.append(kw["status"])

    monkeypatch.setattr("core.agent.tool_registry.execute_tool", _fake_execute)
    monkeypatch.setattr("core.agentic.tool_dispatch.observe_tool_call", _fake_observe)

    outcome = await dispatch_tool_calls(
        [_tc("rag")], course_id="c1", enabled_tools=["rag"], stream=_stream(),
    )
    msg = _find_tool_msg(outcome, "rag")
    assert "[系统指令]" not in msg["content"]
    assert seen == ["ok"]
