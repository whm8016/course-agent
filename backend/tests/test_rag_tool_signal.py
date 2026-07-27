"""RAG 工具层无命中信号回归测试（plan §1.3 / §4）。

核心断言：知识库无命中时，_execute_rag 返回 success=False + 明确提示且不带 sources，
而不是把空串/英文道歉句当证据喂给 Agent 并弹出无效来源卡片。
"""
from __future__ import annotations

import pytest

from core.agent.tool_registry import _execute_rag


class _FakeRetriever:
    """假检索器：retrieve_context / graph_augmented_retrieve 返回固定字符串。"""

    def __init__(self, context_text: str = "") -> None:
        self._ctx = context_text

    async def retrieve_context(self, **kwargs):
        return self._ctx

    async def graph_augmented_retrieve(self, **kwargs):
        return self._ctx


@pytest.mark.asyncio
async def test_execute_rag_no_hit_returns_failure_no_sources(monkeypatch):
    """无命中：success=False、无 sources、提示中文无命中，绝不泄漏英文道歉句。"""
    monkeypatch.setattr("core.rag.get_retriever", lambda name: _FakeRetriever(""))
    result = await _execute_rag(course_id="c1", query="扫地机怎么退货")

    assert result.success is False
    assert result.sources == []  # 不带 sources → 前端不弹来源卡片
    assert "未检索到" in result.content
    assert "[no-context]" not in result.content
    assert "Sorry" not in result.content


@pytest.mark.asyncio
async def test_execute_rag_hit_returns_sources(monkeypatch):
    """有命中：success=True、带 rag 来源、内容含证据。"""
    monkeypatch.setattr(
        "core.rag.get_retriever",
        lambda name: _FakeRetriever("KVL：沿任意闭合回路电压代数和为零。"),
    )
    result = await _execute_rag(course_id="c1", query="KVL 是什么")

    assert result.success is True
    assert len(result.sources) == 1
    assert result.sources[0]["type"] == "rag"
    assert "KVL" in result.content


@pytest.mark.asyncio
async def test_execute_rag_general_course_short_circuits(monkeypatch):
    """自由问答（course_id=general）短路：不查库，success=False，无 sources。"""
    called = {"n": 0}

    def _should_not_call(name):
        called["n"] += 1
        return _FakeRetriever("不应被取到")

    monkeypatch.setattr("core.rag.get_retriever", _should_not_call)
    result = await _execute_rag(course_id="general", query="x")

    assert result.success is False
    assert result.sources == []
    assert called["n"] == 0  # 短路，未触达检索器
