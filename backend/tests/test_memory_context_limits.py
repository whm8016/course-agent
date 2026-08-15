"""读侧限权（r1/r2）：第一路 mem0 检索的每槽上限、主题过滤与门控去重。

验收口径（plan §4）：
- 给定 5 条 persona + 3 条跨主题 event → 注入后 persona ≤3、跨主题 event 被过滤
- 门控本轮开口的证据主题 event 被剔除（同一件事不在两路里说两遍）
- 每条 ≤80 字；header 含三条禁令；事件段含免责句
"""
from unittest.mock import MagicMock, patch

import pytest

from core.memory import mem0_client
from core.memory.mem0_client import _ITEM_MAX_CHARS


def _mem_settings(monkeypatch):
    from settings import get_settings

    cfg = get_settings().mem0
    monkeypatch.setattr(cfg, "conflict_detect_enabled", False)
    monkeypatch.setattr(cfg, "search_threshold", 0.45)


def _mk(results):
    m = MagicMock()

    async def fake_search(query, filters=None, top_k=5, **kw):
        kind = (filters or {}).get("kind")
        return {"results": [r for r in results if r["kind"] == kind]}

    m.search = fake_search
    return m


_LONG = "偏好" * 100  # 200 字，超每条 80 字上限


def _items(kind, n, topic_ids=None):
    return [
        {"kind": kind, "memory": f"{kind}-{i} {_LONG if i == 0 else '短内容'}",
         "metadata": {"topic_ids": topic_ids or []}}
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_slot_limits_and_topic_filter(monkeypatch):
    """5 条 persona → ≤3；跨主题 event 被过滤；门控证据主题 event 被剔除。"""
    _mem_settings(monkeypatch)
    results = (
        _items("persona", 5)
        + _items("event", 2, topic_ids=["ohm"])       # 本轮命中主题：保留（≤2）
        + _items("event", 1, topic_ids=["thevenin"])  # 跨主题：过滤
    )
    with patch.object(mem0_client, "get_memory", lambda: _mk(results)):
        out = await mem0_client.build_memory_context(
            "u1", "欧姆定律怎么用", course_id="c1",
            matched_topic="ohm", gate_evidence_topic=None,
        )
    persona_lines = [ln for ln in out.splitlines() if ln.startswith("- ") and "persona-" in ln]
    event_lines = [ln for ln in out.splitlines() if ln.startswith("- ") and "event-" in ln]
    assert len(persona_lines) == 3          # persona ≤3
    assert len(event_lines) == 2            # 本主题 event ≤2
    assert "thevenin" not in out or all("thevenin" not in ln for ln in event_lines)
    # 每条 ≤80 字（+2 是 "- " 前缀）
    assert all(len(ln) <= _ITEM_MAX_CHARS + 2 for ln in out.splitlines() if ln.startswith("- "))


@pytest.mark.asyncio
async def test_gate_evidence_topic_dedup(monkeypatch):
    """门控本轮开口（evidence=ohm）→ ohm 的 event 行被剔除，避免同主题说两遍。"""
    _mem_settings(monkeypatch)
    results = (
        _items("persona", 1)
        + _items("event", 2, topic_ids=["ohm"])
        + _items("event", 1, topic_ids=["kcl"])
    )
    with patch.object(mem0_client, "get_memory", lambda: _mk(results)):
        out = await mem0_client.build_memory_context(
            "u1", "问欧姆", course_id="c1",
            matched_topic="ohm", gate_evidence_topic="ohm",
        )
    event_lines = [ln for ln in out.splitlines() if ln.startswith("- ") and "event-" in ln]
    assert event_lines == []  # ohm 被门控去重剔除，kcl 不匹配主题早被过滤


@pytest.mark.asyncio
async def test_no_matched_topic_keeps_legacy_behavior(monkeypatch):
    """matched_topic 空 + course_id 也空（IM 无课程概念）→ 不做主题过滤，保留旧行为。"""
    _mem_settings(monkeypatch)
    results = _items("persona", 1) + _items("event", 3, topic_ids=["x"])
    with patch.object(mem0_client, "get_memory", lambda: _mk(results)):
        out = await mem0_client.build_memory_context("u1", "问", course_id="")
    event_lines = [ln for ln in out.splitlines() if ln.startswith("- ") and "event-" in ln]
    assert len(event_lines) == 2  # 无主题过滤但仍有每槽上限


@pytest.mark.asyncio
async def test_course_without_topic_match_empties_event_slot(monkeypatch):
    """有课程但问句没命中主题 → event 槽留空（persona 仍注入）。

    与门控「没命中就沉默」对齐：门控刚决定这轮不提跨主题学情，event 槽若照旧
    全量注入跨主题历史，等于另一条路把话又提了。旧 bug 只区分「matched_topic
    有无」，把「有课程没命中」混进「IM 无课程」的全量旧行为里。
    """
    _mem_settings(monkeypatch)
    results = _items("persona", 1) + _items("event", 3, topic_ids=["x"])
    with patch.object(mem0_client, "get_memory", lambda: _mk(results)):
        out = await mem0_client.build_memory_context(
            "u1", "问", course_id="c1", matched_topic=""
        )
    event_lines = [ln for ln in out.splitlines() if ln.startswith("- ") and "event-" in ln]
    assert event_lines == []                       # event 槽留空
    assert any("persona-0" in ln for ln in out.splitlines())  # persona 不受影响


@pytest.mark.asyncio
async def test_header_restrictions(monkeypatch):
    """header 三禁令 + 事件段免责句。"""
    _mem_settings(monkeypatch)
    results = _items("persona", 1) + _items("event", 1, topic_ids=["ohm"])
    with patch.object(mem0_client, "get_memory", lambda: _mk(results)):
        out = await mem0_client.build_memory_context(
            "u1", "问", course_id="c1", matched_topic="ohm"
        )
    assert "不得作为该生掌握程度的结论" in out
    assert "不得据此主动展开学生本轮未问的知识点" in out
    assert "与本轮问题无关时忽略" in out
    assert "不代表当前掌握程度" in out
