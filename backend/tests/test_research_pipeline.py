"""ResearchPipeline 数据结构 / 引用 / 解析单测（Phase 4）。

验证：
- DynamicTopicQueue 状态机（add/get_pending/all_done/is_full/find_similar/mark_*）
- CitationManager 去重 / 编号 / inline_marker / render_references
- _parse_sub_topics JSON 解析与回退
- _extract_traces_from_knowledge 从 [来源: ...] 标记抽 ToolTrace（url vs rag）
- 端到端 pipeline.run：mock run_agent_loop，验证 4 阶段编排 + 引用附录产出
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from core.agentic.types import LoopOutcome
from core.context import UnifiedContext
from core.research.citation_manager import CitationManager
from core.research.data_structures import (
    DEFAULT_QUEUE_MAX_LENGTH,
    DynamicTopicQueue,
    TopicStatus,
)
from core.research.pipeline import (
    ResearchPipeline,
    _extract_traces_from_knowledge,
    _parse_sub_topics,
    _strip_source_markers,
)


# ── DynamicTopicQueue 状态机 ────────────────────────────────────────────────────


def test_queue_add_and_pending():
    q = DynamicTopicQueue(max_length=8)
    b1 = q.add_block("导数定义")
    b2 = q.add_block("积分应用", "定积分的几何意义")
    assert b1.block_id == "block_1"
    assert b2.block_id == "block_2"
    assert b2.overview == "定积分的几何意义"
    assert [b.block_id for b in q.get_pending()] == ["block_1", "block_2"]
    assert q.all_done() is False


def test_queue_status_transitions():
    q = DynamicTopicQueue()
    b = q.add_block("t")
    assert q.mark_researching(b.block_id) is True
    assert b.status == TopicStatus.RESEARCHING
    assert q.get_pending() == []
    assert q.mark_completed(b.block_id) is True
    assert b.status == TopicStatus.COMPLETED
    assert q.all_done() is True


def test_queue_all_done_with_failed():
    q = DynamicTopicQueue()
    a = q.add_block("a")
    q.add_block("b")
    q.mark_completed(a.block_id)
    assert q.all_done() is False
    q.mark_failed("block_2")
    assert q.all_done() is True  # 终态即可


def test_queue_is_full_and_overflow():
    q = DynamicTopicQueue(max_length=2)
    q.add_block("a")
    q.add_block("b")
    assert q.is_full() is True
    with pytest.raises(RuntimeError):
        q.add_block("c")


def test_queue_default_max_length_is_8():
    q = DynamicTopicQueue()
    assert q.max_length == DEFAULT_QUEUE_MAX_LENGTH == 8


def test_queue_find_similar_dedup():
    q = DynamicTopicQueue()
    q.add_block("  导数 定义  ")
    # 归一化相等
    assert q.find_similar("导数定义") is not None
    # 子串匹配
    assert q.find_similar("导数 定义与应用") is not None
    # 不相关
    assert q.find_similar("矩阵特征值") is None


def test_queue_empty_all_done_is_false():
    assert DynamicTopicQueue().all_done() is False


def test_queue_statistics():
    from core.research.data_structures import ToolTrace

    q = DynamicTopicQueue()
    b = q.add_block("t")
    b.add_source(ToolTrace(tool_type="web_search", query="q", summary="s", source="u"))
    b.add_source(ToolTrace(tool_type="rag", query="q2", summary="s2", source="doc.pdf"))
    stats = q.statistics()
    assert stats["total"] == 1
    assert stats["sources"] == 2
    assert stats["completed"] == 0


# ── CitationManager ─────────────────────────────────────────────────────────────


def test_citation_dedup_by_url():
    cm = CitationManager()
    n1 = cm.add_source(url="https://a.com/1", title="A", tool_type="web_search")
    n2 = cm.add_source(url="https://a.com/1", title="A 重命名", tool_type="web_search")
    assert n1 == n2  # 同 url 去重
    assert len(cm) == 1


def test_citation_dedup_by_title_for_rag():
    cm = CitationManager()
    n1 = cm.add_source(title="导数定义.pdf", tool_type="rag", query="导数")
    n2 = cm.add_source(title="  导数定义.pdf ", tool_type="rag", query="其它")
    assert n1 == n2  # 归一化 title 去重
    n3 = cm.add_source(title="积分.pdf", tool_type="rag")
    assert n3 == 2
    assert len(cm) == 2


def test_citation_inline_marker_and_render():
    cm = CitationManager()
    a = cm.add_source(url="https://a.com", title="A", tool_type="web_search")
    b = cm.add_source(title="doc.pdf", tool_type="rag")
    assert cm.inline_marker(a) == "[1]"
    assert cm.inline_marker(b) == "[2]"
    assert cm.inline_marker(999) == ""
    rendered = cm.render_references()
    assert "## 参考资料" in rendered
    assert "1. [A](https://a.com)" in rendered
    assert "2. doc.pdf" in rendered


def test_citation_render_empty():
    assert CitationManager().render_references() == ""


# ── _parse_sub_topics ───────────────────────────────────────────────────────────


def test_parse_sub_topics_from_json():
    text = '{"sub_topics":[{"title":"导数","overview":"定义与计算"},{"title":"积分","overview":"应用"}]}'
    items = _parse_sub_topics(text, "主题", num_subtopics=5)
    assert len(items) == 2
    assert items[0] == {"title": "导数", "overview": "定义与计算"}


def test_parse_sub_topics_count_cap():
    text = (
        '{"sub_topics":['
        '{"title":"a"},{"title":"b"},{"title":"c"}'
        "]}"
    )
    assert len(_parse_sub_topics(text, "主题", num_subtopics=2)) == 2


def test_parse_sub_topics_fallback_on_garbage():
    items = _parse_sub_topics("这不是 json", "原始主题", num_subtopics=3)
    assert len(items) == 1
    assert items[0]["title"] == "原始主题"  # 解析失败回退为单主题


def test_parse_sub_topics_drops_empty_title():
    text = '{"sub_topics":[{"title":""},{"title":"ok"}]}'
    items = _parse_sub_topics(text, "主题", num_subtopics=5)
    assert len(items) == 1
    assert items[0]["title"] == "ok"


# ── 来源标记提取 ────────────────────────────────────────────────────────────────


def test_extract_traces_url_and_rag():
    knowledge = (
        "导数是变化率 [来源: https://en.wikipedia.org/wiki/Derivative]。"
        "知识库补充 [来源: 高数教材.pdf]。"
    )
    traces = _extract_traces_from_knowledge(knowledge, block=None)  # type: ignore[arg-type]
    assert len(traces) == 2
    assert traces[0].tool_type == "web_search"
    assert traces[0].source.startswith("https://")
    assert traces[1].tool_type == "rag"
    assert traces[1].source == "高数教材.pdf"


def test_extract_traces_empty():
    assert _extract_traces_from_knowledge("无来源的纯文本", block=None) == []  # type: ignore[arg-type]


def test_strip_source_markers():
    text = "结论 A [来源: https://x.com]。结论 B [来源: doc.pdf]。"
    cleaned = _strip_source_markers(text)
    assert "来源" not in cleaned
    assert cleaned.startswith("结论 A")
    assert "结论 B" in cleaned


# ── 端到端 pipeline.run（mock run_agent_loop）───────────────────────────────────


def _make_context() -> UnifiedContext:
    return UnifiedContext(
        course_id="course_1",
        user_message="研究导数与积分",
        enabled_tools=["rag", "web_search"],
        mode="research",
        session_id="sess_1",
    )


def _mock_loop_factory():
    """返回一个按调用顺序产出阶段产物的 mock。

    调用顺序固定：rephrase → decompose → block_1 → block_2 → outline → intro →
    section_1 → section_2 → conclusion。每次调用消费序列中下一个产物。
    """
    # research_step 产物需随 block 标题变化，故用占位按 block_1/2 顺序填充
    seq: list[str] = [
        # 0 rephrase
        "精炼主题：研究微积分中导数与积分的核心概念及应用。",
        # 1 decompose
        (
            '{"sub_topics":['
            '{"title":"导数定义","overview":"变化率与极限"},'
            '{"title":"积分应用","overview":"面积与累积"}'
            "]}"
        ),
        # 2 block_1 (导数定义)
        "导数是瞬时变化率 [来源: https://en.wikipedia.org/wiki/Derivative]。",
        # 3 block_2 (积分应用)
        "积分用于求面积与累积 [来源: 积分教材.pdf]。",
        # 4 outline
        (
            '{"title":"导数与积分研究报告","sections":['
            '{"id":"S1","title":"导数","intent":"变化率","block_ids":["block_1"]},'
            '{"id":"S2","title":"积分","intent":"累积","block_ids":["block_2"]}'
            "]}"
        ),
        # 5 intro
        "## 1. 引言\n微积分是研究变化与累积的数学语言。",
        # 6 section_1
        "## 2. 导数\n导数刻画瞬时变化率 [1]。",
        # 7 section_2
        "## 3. 积分\n积分刻画累积 [2]。",
        # 8 conclusion
        "## 4. 结论\n导数与积分互为逆运算。",
    ]
    idx: dict[str, int] = {"i": 0}

    async def _fake(*, context, stream, system_prompt, tool_schemas=None, model=None,
                    client=None, max_iterations=10, **kw):
        i = idx["i"]
        idx["i"] = i + 1
        text = seq[i] if i < len(seq) else f"## 段落 {i}"
        return LoopOutcome(final_text=text, rounds=1, tools_used=[], completed=True)

    return _fake


def test_pipeline_run_end_to_end():
    pipeline = ResearchPipeline(num_subtopics=2, max_parallel=2)
    ctx = _make_context()

    async def _go():
        from core.stream_bus import StreamBus

        stream = StreamBus()
        with patch("core.research.pipeline.run_agent_loop", new=_mock_loop_factory()):
            result = await pipeline.run(
                topic="研究导数与积分", context=ctx, stream=stream
            )
        return result

    result = asyncio.run(_go())

    assert result["research_id"].startswith("research_")
    assert result["report"].startswith("# 导数与积分研究报告")
    # 引言 + 两节 + 结论 + 参考资料
    assert "## 1." in result["report"] or "## 1 " in result["report"]
    assert "参考资料" in result["report"]
    # 两个来源都进了附录
    assert "wikipedia.org" in result["report"]
    assert "积分教材.pdf" in result["report"]
    # metadata
    assert result["metadata"]["block_count"] == 2
    assert result["metadata"]["sources"] >= 2
    assert result["metadata"]["stages"] == ["rephrase", "decompose", "researching", "reporting"]
