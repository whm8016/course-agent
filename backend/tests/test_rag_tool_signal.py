"""RAG 工具层信号 + 双后端 auto 路由回归测试。

覆盖：
1. 无命中 → success=False + 中文提示 + 无 sources（不泄漏英文道歉句）。
2. 有命中 → success=True + rag 来源。
3. 自由问答（general）短路，不触达检索器。
4. auto 路由（Phase 2）：
   - 手动 mix/naive/local → 透传 LightRAG，不受 strategy 影响；
   - auto + relationship + lightrag 就绪 → LightRAG 图增强（多跳）；
   - auto + fact + pg 就绪 → pgvector 向量；
   - 手动 mix 但课程未建 lightrag → success=False，不降级到 pg。

_get_ready_backends（查 DB）在本测试用 monkeypatch 替换为固定集合，隔离 DB。
"""
from __future__ import annotations

import pytest

from core.agent.tool_registry import _execute_rag


class _RecordingRetriever:
    """假检索器：记录每次调用 (method, kwargs)，返回固定字符串。"""

    def __init__(self, context_text: str = "HIT") -> None:
        self._ctx = context_text
        self.calls: list[tuple[str, dict]] = []

    async def retrieve_context(self, **kwargs):
        self.calls.append(("retrieve", kwargs))
        return self._ctx

    async def graph_augmented_retrieve(self, **kwargs):
        self.calls.append(("graph", kwargs))
        return self._ctx


def _patch(monkeypatch, ready: set[str], lightrag=None, pg=None) -> list[str]:
    """打桩：_get_ready_backends 返回 ready；get_retriever 按 name 分发，并记录请求的后端名。"""
    async def _ready(_cid):
        return set(ready)
    monkeypatch.setattr("core.agent.tool_registry._get_ready_backends", _ready)

    table: dict[str, _RecordingRetriever] = {}
    if lightrag is not None:
        table["lightrag"] = lightrag
    if pg is not None:
        table["llamaindex_pg"] = pg
    requested: list[str] = []

    def _get_retriever(name):
        requested.append(name)
        return table.get(name)

    monkeypatch.setattr("core.rag.get_retriever", _get_retriever)
    return requested


@pytest.mark.asyncio
async def test_execute_rag_no_hit_returns_failure_no_sources(monkeypatch):
    """无命中：success=False、无 sources、提示中文无命中，绝不泄漏英文道歉句。"""
    lr = _RecordingRetriever("")
    _patch(monkeypatch, ready={"lightrag"}, lightrag=lr)
    result = await _execute_rag(course_id="c1", query="扫地机怎么退货")

    assert result.success is False
    assert result.sources == []
    assert "未检索到" in result.content
    assert "[no-context]" not in result.content
    assert "Sorry" not in result.content


@pytest.mark.asyncio
async def test_execute_rag_hit_returns_sources(monkeypatch):
    """有命中：success=True、带 rag 来源、内容含证据。"""
    lr = _RecordingRetriever("KVL：沿任意闭合回路电压代数和为零。")
    _patch(monkeypatch, ready={"lightrag"}, lightrag=lr)
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
        return _RecordingRetriever("不应被取到")

    monkeypatch.setattr("core.rag.get_retriever", _should_not_call)
    result = await _execute_rag(course_id="general", query="x")

    assert result.success is False
    assert result.sources == []
    assert called["n"] == 0


# ── Phase 2：auto 路由 ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mode_mix_passthrough_to_lightrag(monkeypatch):
    """手动 mix：透传 LightRAG retrieve_context(mode=mix)，strategy=relationship 被忽略。"""
    lr = _RecordingRetriever("证据")
    pg = _RecordingRetriever("PG")
    _patch(monkeypatch, ready={"lightrag", "llamaindex_pg"}, lightrag=lr, pg=pg)
    await _execute_rag(course_id="c1", query="q", mode="mix", strategy="relationship")

    assert len(lr.calls) == 1 and lr.calls[0][0] == "retrieve"
    assert lr.calls[0][1]["mode"] == "mix"  # 透传
    assert pg.calls == []                    # 手动模式不走 pg


@pytest.mark.asyncio
async def test_auto_relationship_uses_lightrag_graph(monkeypatch):
    """auto + relationship + lightrag 就绪 → LightRAG 图增强（多跳），不走 pg。"""
    lr = _RecordingRetriever("图谱证据")
    pg = _RecordingRetriever("PG")
    _patch(monkeypatch, ready={"lightrag", "llamaindex_pg"}, lightrag=lr, pg=pg)
    await _execute_rag(course_id="c1", query="q", strategy="relationship")  # mode 默认 auto

    assert lr.calls and lr.calls[0][0] == "graph"
    assert pg.calls == []


@pytest.mark.asyncio
async def test_auto_fact_uses_pgvector(monkeypatch):
    """auto + fact + pg 就绪 → pgvector 向量检索，不走 lightrag。"""
    lr = _RecordingRetriever("LR")
    pg = _RecordingRetriever("PG 向量证据")
    _patch(monkeypatch, ready={"lightrag", "llamaindex_pg"}, lightrag=lr, pg=pg)
    result = await _execute_rag(course_id="c1", query="q", strategy="fact")  # mode 默认 auto

    assert pg.calls and pg.calls[0][0] == "retrieve"
    assert lr.calls == []
    assert result.success is True
    assert "PG 向量证据" in result.content


@pytest.mark.asyncio
async def test_manual_mix_without_lightrag_fails(monkeypatch):
    """手动 mix 但课程未建 lightrag（仅 pg 就绪）→ success=False，不降级到 pg。"""
    pg = _RecordingRetriever("PG")
    _patch(monkeypatch, ready={"llamaindex_pg"}, pg=pg)
    result = await _execute_rag(course_id="c1", query="q", mode="mix")

    assert result.success is False
    assert "LightRAG" in result.content
    assert pg.calls == []  # 手动 LightRAG 模式不退到 pg
