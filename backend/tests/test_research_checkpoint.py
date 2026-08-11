"""深度研究阶段级 checkpoint 回归（plan 阶段 2B）。

验证：
1. 全新 run：每阶段完成后调 checkpoint.save_phase（rephrase/decompose/researching/reporting），done 后 clear。
2. resume_research_id：checkpoint 命中已完成阶段 → 跳过 rephrase/decompose/researching，只重放 reporting。
3. awaiting_user：上次在 rephrase 的 ask_user 暂停期间崩溃 → 重发同一份卡片、等回复、把回复当 refined_topic，
   不重跑 rephrase。

checkpoint 读写整体 mock（不依赖 DB）；run_agent_loop 用序贯 fake 出阶段产物。
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.agentic.types import LoopOutcome
from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.research.data_structures import (
    DynamicTopicQueue,
    TopicStatus,
    ToolTrace,
)
from core.research.pipeline import ResearchPipeline
from core.stream_bus import StreamBus


# ── helpers ────────────────────────────────────────────────────────────────────


def _ctx(with_waiter=False, resume_id: str = "") -> UnifiedContext:
    ctx = UnifiedContext(
        course_id="c1", user_message="研究X", enabled_tools=["rag", "web_search"],
        mode="research", session_id="s1",
    )
    if resume_id:
        ctx.metadata["resume_research_id"] = resume_id
    if with_waiter:
        async def _w():
            return {"text": "用户精炼后的主题", "answers": None}
        ctx.metadata["wait_for_user_reply"] = _w
    return ctx


def _seq_loop(seq: list[str]):
    """按调用顺序产出 seq 中下一个文本的 fake run_agent_loop。"""
    idx = {"i": 0}

    async def _fake(**kw):
        i = idx["i"]
        idx["i"] = i + 1
        return LoopOutcome(final_text=seq[i] if i < len(seq) else f"## 段落 {i}",
                           rounds=1, tools_used=[], completed=True)

    return _fake


# 全新 run 的序贯产物（rephrase/decompose/block×2/outline/intro/sec×2/conclusion）
_FRESH_SEQ = [
    "精炼主题：研究微积分中导数与积分的核心概念及应用。",
    '{"sub_topics":[{"title":"导数","overview":"定义"},{"title":"积分","overview":"应用"}]}',
    # 够长（≥50）+ 有来源标记，过块自检
    "导数刻画函数在某点的瞬时变化率，是微积分的核心概念，几何上表示切线斜率，联结极限与微分 [来源: https://en.wikipedia.org/wiki/Derivative]。",
    "积分用于求面积与累积量，是微积分的基本运算，与导数互为逆运算并由微积分基本定理统一 [来源: 积分教材.pdf]。",
    '{"title":"微积分报告","sections":[{"id":"S1","title":"导数","intent":"","block_ids":["block_1"]},{"id":"S2","title":"积分","intent":"","block_ids":["block_2"]}]}',
    "## 1. 引言\n微积分研究变化与累积。",
    "## 2. 导数\n导数是变化率 [1]。",
    "## 3. 积分\n积分是累积 [2]。",
    "## 4. 结论\n二者互逆。",
]
# reporting-only 序贯产物（resume 跳过 rephrase/decompose/researching 后）
_REPORT_SEQ = _FRESH_SEQ[4:]


def _patch_stack(loop_fake, ckpt_load=None):
    """返回 patcher 列表：run_agent_loop + profile/layers/describe + checkpoint.load/save/clear。"""
    save = AsyncMock()
    clear = AsyncMock()
    load = AsyncMock(return_value=ckpt_load)
    patchers = [
        patch("core.research.pipeline.run_agent_loop", new=loop_fake),
        patch("core.research.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=ProfileRuntime())),
        patch("core.research.pipeline.build_common_context_layers", new=AsyncMock(return_value=CommonContextLayers())),
        patch("core.research.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
        patch("core.research.pipeline.get_tool_schemas", return_value=[]),
        patch("core.research.checkpoint.load", new=load),
        patch("core.research.checkpoint.save_phase", new=save),
        patch("core.research.checkpoint.clear", new=clear),
    ]
    return patchers, save, clear, load


def _event_types(stream: StreamBus) -> set[str]:
    out: set[str] = set()
    for e in stream._history:
        if isinstance(e, dict):
            out.add(str(e.get("type")))
        elif hasattr(e, "type"):
            out.add(e.type.value if hasattr(e.type, "value") else str(e.type))
    return out


# ── tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fresh_run_saves_checkpoint_each_phase():
    """全新 run：save_phase 被调，phase 覆盖四个阶段；done 后 clear。"""
    pipe = ResearchPipeline(num_subtopics=2, max_parallel=2)
    ctx = _ctx()
    stream = StreamBus()
    patchers, save, clear, _load = _patch_stack(_seq_loop(_FRESH_SEQ), ckpt_load=None)
    for p in patchers:
        p.start()
    try:
        result = await pipe.run(topic="研究导数与积分", context=ctx, stream=stream)
    finally:
        for p in patchers:
            p.stop()

    phases = {call.kwargs.get("phase") for call in save.call_args_list}
    assert {"rephrase", "decompose", "researching", "reporting"}.issubset(phases)
    # reporting 存 status=done，之后 clear
    assert any(call.kwargs.get("phase") == "reporting" and call.kwargs.get("status") == "done"
               for call in save.call_args_list)
    clear.assert_called_once()
    # research_id 提前下发（前端保存用）
    assert "research_id" in result
    assert "stage_start" in _event_types(stream)


@pytest.mark.asyncio
async def test_resume_skips_completed_phases():
    """resume_research_id 命中 phase=researching（队列已完成）→ 跳过 rephrase/decompose/researching，只跑 reporting。"""
    # 造一个「researching 已完成」的 queue：2 个 completed 块带 knowledge + sources
    q = DynamicTopicQueue("r_resume")
    b1 = q.add_block("导数", "定义")
    b1.status = TopicStatus.COMPLETED
    b1.knowledge = "导数知识摘要 [来源: https://x.com]"
    b1.add_source(ToolTrace(tool_type="web_search", query="q", summary="s", source="https://x.com"))
    b2 = q.add_block("积分", "应用")
    b2.status = TopicStatus.COMPLETED
    b2.knowledge = "积分知识摘要 [来源: doc.pdf]"
    b2.add_source(ToolTrace(tool_type="rag", query="q", summary="s", source="doc.pdf"))

    state = {"refined_topic": "已精炼主题", "queue": q.to_dict()}
    ckpt = SimpleNamespace(
        research_id="r_resume", phase="researching", status="running",
        state_json=json.dumps(state), pending_question_json="", topic="已精炼主题",
    )

    pipe = ResearchPipeline(num_subtopics=2, max_parallel=2)
    ctx = _ctx(resume_id="r_resume")
    stream = StreamBus()

    rephrase = AsyncMock()
    decompose = AsyncMock()
    drive = AsyncMock()
    patchers, _save, _clear, _load = _patch_stack(_seq_loop(_REPORT_SEQ), ckpt_load=ckpt)
    patchers.append(patch.object(pipe, "_rephrase", rephrase))
    patchers.append(patch.object(pipe, "_decompose", decompose))
    patchers.append(patch.object(pipe, "_drive_queue", drive))
    for p in patchers:
        p.start()
    try:
        result = await pipe.run(topic="原始", context=ctx, stream=stream)
    finally:
        for p in patchers:
            p.stop()

    rephrase.assert_not_called()   # 跳过
    decompose.assert_not_called()  # 跳过
    drive.assert_not_called()      # researching 已完成，跳过
    assert result["research_id"] == "r_resume"           # 复用 research_id
    assert result["topic"] == "已精炼主题"                # 用 saved refined_topic


@pytest.mark.asyncio
async def test_awaiting_user_restores_card_and_uses_reply():
    """上次 rephrase ask_user 暂停时崩溃：重发卡片、等回复、回复当 refined_topic，不重跑 rephrase。"""
    pending = {"intro": "请澄清范围", "questions": [{"questionId": "q1", "text": "受众是谁？"}]}
    ckpt = SimpleNamespace(
        research_id="r_au", phase="rephrase", status="awaiting_user",
        state_json=json.dumps({}), pending_question_json=json.dumps(pending),
        topic="原始主题",
    )
    # rephrase 走 awaiting_user 分支跳过；后续 decompose/researching/reporting 用序贯 fake
    seq = _FRESH_SEQ[1:]  # 去掉 rephrase 产物

    pipe = ResearchPipeline(num_subtopics=2, max_parallel=2)
    ctx = _ctx(with_waiter=True, resume_id="r_au")
    stream = StreamBus()

    rephrase = AsyncMock()
    patchers, _save, _clear, _load = _patch_stack(_seq_loop(seq), ckpt_load=ckpt)
    patchers.append(patch.object(pipe, "_rephrase", rephrase))
    for p in patchers:
        p.start()
    try:
        result = await pipe.run(topic="原始主题", context=ctx, stream=stream)
    finally:
        for p in patchers:
            p.stop()

    rephrase.assert_not_called()                      # 不重跑 rephrase
    assert result["topic"] == "用户精炼后的主题"        # waiter 回的 text 当 refined_topic
    types = _event_types(stream)
    assert "ask_user_card" in types                   # 重发了同一份卡片
