"""P0：mem0 读侧精排——候选数超槽上限才触发 qwen3-rerank，None/异常降级原序。

覆盖路径：
- 候选数 > 上限 + 开关开 → 触发精排且注入顺序按 rerank 序（截断吃精排头部）
- 候选数 ≤ 上限（含 event 主题过滤后回落到上限内）→ 不触发（省一次热路径 API 往返）
- 开关关（默认）→ 不触发；build_rerank_fn 返回 None → 原序；rerank 抛异常 → 原序
"""
from unittest.mock import MagicMock, patch

import pytest

from core.memory import mem0_client


def _mem_settings(monkeypatch, rerank_enabled=False):
    from settings import get_settings

    cfg = get_settings().mem0
    monkeypatch.setattr(cfg, "conflict_detect_enabled", False)
    monkeypatch.setattr(cfg, "search_threshold", 0.0)
    monkeypatch.setattr(cfg, "rerank_enabled", rerank_enabled)


def _mk(results):
    m = MagicMock()

    async def fake_search(query, filters=None, top_k=5, **kw):
        kind = (filters or {}).get("kind")
        return {"results": [r for r in results if r["kind"] == kind]}

    m.search = fake_search
    return m


def _items(kind, n, topic_ids=None):
    return [
        {"kind": kind, "id": f"{kind}-{i}", "memory": f"{kind}-{i} 内容",
         "metadata": {"topic_ids": topic_ids or []}}
        for i in range(n)
    ]


def _fake_rerank_factory(order, calls):
    """构造假 rerank_fn：记录每次调用的候选 docs，按 order 下标序返回（模拟精排翻转）。"""

    async def rerank_fn(query, docs, top_n=5):
        calls.append(docs)
        return [docs[i] for i in order]

    return rerank_fn


@pytest.mark.asyncio
async def test_rerank_reorders_when_over_limit(monkeypatch):
    """persona 5 条 > 上限 3 → 精排触发，注入吃精排头部（persona-4,2,0）。"""
    _mem_settings(monkeypatch, rerank_enabled=True)
    results = _items("persona", 5)
    calls: list = []
    fn = _fake_rerank_factory([4, 2, 0, 1, 3], calls)
    with (
        patch.object(mem0_client, "get_memory", lambda: _mk(results)),
        patch("core.rag.rerank.build_rerank_fn", lambda: fn),
    ):
        out = await mem0_client.build_memory_context("u1", "问", course_id="")
    persona_lines = [ln for ln in out.splitlines() if ln.startswith("- ") and "persona-" in ln]
    assert len(persona_lines) == 3
    assert persona_lines[0].startswith("- persona-4")
    assert persona_lines[1].startswith("- persona-2")
    assert persona_lines[2].startswith("- persona-0")
    # 契约：送进 rerank 的候选已包装出 content 字段（build_rerank_fn 只认 content）
    assert len(calls) == 1
    assert all(d.get("content") == d.get("memory") for d in calls[0])


@pytest.mark.asyncio
async def test_no_rerank_when_within_limit(monkeypatch):
    """两槽候选都 ≤ 上限（event 主题过滤后回落）→ 不构造 rerank_fn。"""
    _mem_settings(monkeypatch, rerank_enabled=True)
    results = (
        _items("persona", 3)                                  # = 上限 3，不触发
        + _items("event", 2, topic_ids=["ohm"])               # 过滤后 = 上限 2，不触发
        + _items("event", 6, topic_ids=["thevenin"])          # 跨主题：过滤掉，不进精排
    )
    with (
        patch.object(mem0_client, "get_memory", lambda: _mk(results)),
        patch("core.rag.rerank.build_rerank_fn") as mock_build,
    ):
        out = await mem0_client.build_memory_context(
            "u1", "问", course_id="c1", matched_topic="ohm"
        )
    mock_build.assert_not_called()
    assert "persona-0" in out  # 行为不变：原序照常注入


@pytest.mark.asyncio
async def test_rerank_disabled_by_default(monkeypatch):
    """开关关（默认 False）→ 即便候选超上限也不碰 rerank。"""
    _mem_settings(monkeypatch, rerank_enabled=False)
    results = _items("persona", 5) + _items("event", 4)
    with (
        patch.object(mem0_client, "get_memory", lambda: _mk(results)),
        patch("core.rag.rerank.build_rerank_fn") as mock_build,
    ):
        await mem0_client.build_memory_context("u1", "问", course_id="")
    mock_build.assert_not_called()


@pytest.mark.asyncio
async def test_rerank_fn_none_falls_back(monkeypatch):
    """build_rerank_fn 返回 None（无 key / RERANK__ENABLED=false）→ 原序注入。"""
    _mem_settings(monkeypatch, rerank_enabled=True)
    results = _items("persona", 5)
    with (
        patch.object(mem0_client, "get_memory", lambda: _mk(results)),
        patch("core.rag.rerank.build_rerank_fn", lambda: None),
    ):
        out = await mem0_client.build_memory_context("u1", "问", course_id="")
    persona_lines = [ln for ln in out.splitlines() if ln.startswith("- ") and "persona-" in ln]
    # 向量原序前 3（精排缺席不改变行为）
    assert [ln.split()[1] for ln in persona_lines] == ["persona-0", "persona-1", "persona-2"]


@pytest.mark.asyncio
async def test_rerank_exception_falls_back(monkeypatch):
    """rerank API 抛异常 → 降级向量原序，不拖垮记忆注入（每轮对话热路径）。"""
    _mem_settings(monkeypatch, rerank_enabled=True)
    results = _items("persona", 5)

    async def boom(query, docs, top_n=5):
        raise RuntimeError("DashScope rerank HTTP 500")

    with (
        patch.object(mem0_client, "get_memory", lambda: _mk(results)),
        patch("core.rag.rerank.build_rerank_fn", lambda: boom),
    ):
        out = await mem0_client.build_memory_context("u1", "问", course_id="")
    persona_lines = [ln for ln in out.splitlines() if ln.startswith("- ") and "persona-" in ln]
    assert len(persona_lines) == 3
    assert persona_lines[0].startswith("- persona-0")  # 原序头部
