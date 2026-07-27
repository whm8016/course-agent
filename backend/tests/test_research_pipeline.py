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
from unittest.mock import AsyncMock, patch

import pytest

from core.agentic.types import LoopOutcome
from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.stream_bus import StreamBus
from core.research.citation_manager import CitationManager
from core.research.data_structures import (
    DEFAULT_QUEUE_MAX_LENGTH,
    DynamicTopicQueue,
    TopicStatus,
)
from core.research.pipeline import (
    DEFAULT_BLOCK_MAX_TRIES,
    ResearchPipeline,
    _BLOCK_MIN_KNOWLEDGE_CHARS,
    _block_self_check,
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
        # 2 block_1 (导数定义) —— 够长 + 有来源标记，通过块自检（_BLOCK_MIN_KNOWLEDGE_CHARS=50）
        "导数刻画函数在某点的瞬时变化率，是微积分的核心概念，几何上表示切线斜率 [来源: https://en.wikipedia.org/wiki/Derivative]。",
        # 3 block_2 (积分应用) —— 同样够长 + 有来源标记
        "积分用于求面积与累积量，是微积分的基本运算，与导数互为逆运算（微积分基本定理） [来源: 积分教材.pdf]。",
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
        with (
            patch("core.research.pipeline.run_agent_loop", new=_mock_loop_factory()),
            patch(
                "core.research.pipeline.resolve_profile_runtime",
                new=AsyncMock(return_value=ProfileRuntime()),
            ),
            patch(
                "core.research.pipeline.build_common_context_layers",
                new=AsyncMock(return_value=CommonContextLayers()),
            ),
            patch(
                "core.research.pipeline.describe_images",
                new=AsyncMock(side_effect=lambda c, t, r: t),
            ),
        ):
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


# ── 块自检 + 重试（plan 第三批-1）────────────────────────────────────────────────


def test_block_self_check_rejects_empty_and_short():
    assert _block_self_check("", has_retrieval=True) is False
    assert _block_self_check("   ", has_retrieval=True) is False
    # 过短（< _BLOCK_MIN_KNOWLEDGE_CHARS）判失败
    assert _block_self_check("X" * (_BLOCK_MIN_KNOWLEDGE_CHARS - 1), has_retrieval=False) is False


def test_block_self_check_rejects_no_citation_when_retrieval_available():
    # 够长但无 [来源: ...] 标记、且有检索工具 → 判失败（应检索却未引证）
    assert _block_self_check("Y" * 200, has_retrieval=True) is False


def test_block_self_check_passes():
    # 够长 + 有来源标记 + 有检索工具 → 通过
    ok = "Z" * 60 + " [来源: doc.pdf]"
    assert _block_self_check(ok, has_retrieval=True) is True
    # 纯推理块（无检索工具）只校验长度
    assert _block_self_check("Z" * 60, has_retrieval=False) is True


def _new_pipeline_for_block_test() -> ResearchPipeline:
    """构造一个 ResearchPipeline 并手动设好 _rt/_layers（_research_block 直接测，跳过 run()）。"""
    pipe = ResearchPipeline()
    pipe._rt = ProfileRuntime()
    pipe._layers = CommonContextLayers()
    return pipe


@pytest.mark.asyncio
async def test_research_block_retries_then_succeeds():
    """第 1 次过短（自检不过）→ 重跑第 2 次产出合格摘要 → mark_completed，run_agent_loop 被调 2 次。"""
    pipe = _new_pipeline_for_block_test()
    q = DynamicTopicQueue()
    block = q.add_block("导数定义")
    ctx = _make_context()
    stream = StreamBus()
    cfg = {"research_step": {
        "system": "{topic}{block_title}{block_overview}{kb_note}",
        "user_template": "{sibling_topics}",
    }}

    calls = {"n": 0}

    async def _fake_loop(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return LoopOutcome(final_text="太短", rounds=1, tools_used=[], completed=True)
        return LoopOutcome(
            final_text=(
                "这是一个足够长的有效研究摘要，详细说明了导数的定义、几何意义与基本计算例子，"
                "可供后续报告引用 [来源: 高数教材.pdf]。"
            ),
            rounds=1, tools_used=[], completed=True,
        )

    with (
        patch("core.research.pipeline.run_agent_loop", new=_fake_loop),
        patch("core.research.pipeline.get_tool_schemas", return_value=[]),
    ):
        await pipe._research_block(
            block=block, queue=q, topic="t", context=ctx, stream=stream, cfg=cfg,
        )

    assert calls["n"] == 2  # 重试了一次
    assert block.status == TopicStatus.COMPLETED
    assert block.knowledge.startswith("这是一个足够长的有效研究摘要")
    assert any(s.source == "高数教材.pdf" for s in block.sources)


@pytest.mark.asyncio
async def test_research_block_marks_failed_after_self_check_exhausted():
    """每次都过短 → 用尽重试 → mark_failed，run_agent_loop 被调 DEFAULT_BLOCK_MAX_TRIES 次。"""
    pipe = _new_pipeline_for_block_test()
    q = DynamicTopicQueue()
    block = q.add_block("空主题")
    ctx = _make_context()
    stream = StreamBus()
    cfg = {"research_step": {
        "system": "{topic}{block_title}{block_overview}{kb_note}",
        "user_template": "{sibling_topics}",
    }}

    calls = {"n": 0}

    async def _always_short(**kw):
        calls["n"] += 1
        return LoopOutcome(final_text="太短", rounds=1, tools_used=[], completed=True)

    with (
        patch("core.research.pipeline.run_agent_loop", new=_always_short),
        patch("core.research.pipeline.get_tool_schemas", return_value=[]),
    ):
        await pipe._research_block(
            block=block, queue=q, topic="t", context=ctx, stream=stream, cfg=cfg,
        )

    assert calls["n"] == DEFAULT_BLOCK_MAX_TRIES
    assert block.status == TopicStatus.FAILED


@pytest.mark.asyncio
async def test_research_block_retries_on_exception_then_marks_failed():
    """每次都抛异常 → 用尽重试 → mark_failed（不向上抛），run_agent_loop 被调 DEFAULT_BLOCK_MAX_TRIES 次。"""
    pipe = _new_pipeline_for_block_test()
    q = DynamicTopicQueue()
    block = q.add_block("异常主题")
    ctx = _make_context()
    stream = StreamBus()
    cfg = {"research_step": {
        "system": "{topic}{block_title}{block_overview}{kb_note}",
        "user_template": "{sibling_topics}",
    }}

    calls = {"n": 0}

    async def _raises(**kw):
        calls["n"] += 1
        raise RuntimeError("boom")

    with (
        patch("core.research.pipeline.run_agent_loop", new=_raises),
        patch("core.research.pipeline.get_tool_schemas", return_value=[]),
    ):
        # 不应抛出（_research_block 内部消化异常）
        await pipe._research_block(
            block=block, queue=q, topic="t", context=ctx, stream=stream, cfg=cfg,
        )

    assert calls["n"] == DEFAULT_BLOCK_MAX_TRIES
    assert block.status == TopicStatus.FAILED


# ── Observer 汇总质量门（消融开关默认关）──────────────────────────────────────


@pytest.mark.asyncio
async def test_observe_and_refill_only_under_sourced_blocks():
    """只对「有内容但 sources < _OBSERVER_MIN_SOURCES」的块补检索；来源够 / 无内容跳过。"""
    from core.research.data_structures import ToolTrace
    from core.research.pipeline import _OBSERVER_MIN_SOURCES

    pipe = _new_pipeline_for_block_test()
    q = DynamicTopicQueue()
    a = q.add_block("导数")          # 有内容 + 0 来源 → 候选
    a.knowledge = "导数是变化率 [来源: doc.pdf]"
    a.sources = []
    b = q.add_block("积分")          # 2 来源 → 跳过
    b.knowledge = "积分..."
    b.add_source(ToolTrace(tool_type="rag", query="q", summary="s", source="x"))
    b.add_source(ToolTrace(tool_type="rag", query="q2", summary="s2", source="y"))
    c = q.add_block("空")            # 无内容 → 跳过
    c.knowledge = ""

    calls: list[str] = []

    async def _fake_block(**kw):
        calls.append(kw["block"].block_id)

    with patch.object(pipe, "_research_block", _fake_block):
        await pipe._observe_and_refill(
            queue=q, topic="t", context=_make_context(), stream=StreamBus(), cfg={},
        )
    assert calls == ["block_1"]  # 只补 a
    assert _OBSERVER_MIN_SOURCES == 2


@pytest.mark.asyncio
async def test_observe_and_refill_caps_candidates():
    """候选数超过 _OBSERVER_MAX_REFILL_BLOCKS 时截断，防队列异常膨胀。"""
    from core.research.pipeline import _OBSERVER_MAX_REFILL_BLOCKS

    pipe = _new_pipeline_for_block_test()
    q = DynamicTopicQueue(max_length=_OBSERVER_MAX_REFILL_BLOCKS + 5)
    for i in range(_OBSERVER_MAX_REFILL_BLOCKS + 5):
        blk = q.add_block(f"t{i}")
        blk.knowledge = f"内容{i}"  # 有内容 + 0 来源 → 候选

    calls: list[str] = []

    async def _fake_block(**kw):
        calls.append(kw["block"].block_id)

    with patch.object(pipe, "_research_block", _fake_block):
        await pipe._observe_and_refill(
            queue=q, topic="t", context=_make_context(), stream=StreamBus(), cfg={},
        )
    assert len(calls) == _OBSERVER_MAX_REFILL_BLOCKS


@pytest.mark.asyncio
async def test_observer_gate_off_skips_refill():
    """开关关（metadata 无 research_observer）→ run 不调 _observe_and_refill。"""
    pipeline = ResearchPipeline(num_subtopics=2, max_parallel=2)
    ctx = _make_context()  # 默认关
    stream = StreamBus()
    invoked: list[dict] = []
    with (
        patch("core.research.pipeline.run_agent_loop", new=_mock_loop_factory()),
        patch("core.research.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=ProfileRuntime())),
        patch("core.research.pipeline.build_common_context_layers", new=AsyncMock(return_value=CommonContextLayers())),
        patch("core.research.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
        patch.object(pipeline, "_observe_and_refill", new=AsyncMock(side_effect=lambda **kw: invoked.append(kw))),
    ):
        await pipeline.run(topic="研究导数与积分", context=ctx, stream=stream)
    assert invoked == []


@pytest.mark.asyncio
async def test_observer_gate_on_invokes_refill():
    """开关开（metadata research_observer=True）→ run 调一次 _observe_and_refill。"""
    pipeline = ResearchPipeline(num_subtopics=2, max_parallel=2)
    ctx = _make_context()
    ctx.metadata["research_observer"] = True
    stream = StreamBus()
    invoked: list[dict] = []
    with (
        patch("core.research.pipeline.run_agent_loop", new=_mock_loop_factory()),
        patch("core.research.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=ProfileRuntime())),
        patch("core.research.pipeline.build_common_context_layers", new=AsyncMock(return_value=CommonContextLayers())),
        patch("core.research.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
        patch.object(pipeline, "_observe_and_refill", new=AsyncMock(side_effect=lambda **kw: invoked.append(kw))),
    ):
        await pipeline.run(topic="研究导数与积分", context=ctx, stream=stream)
    assert len(invoked) == 1
