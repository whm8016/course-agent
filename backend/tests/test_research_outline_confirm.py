"""深度研究「大纲确认」回归（decompose 后暂停，用户过目/编辑子主题再执行 research）。

镜像 test_research_checkpoint / test_research_clarify 的搭架子方式：run_agent_loop 用序贯 fake，
checkpoint 读写整体 mock，get_settings stub 控制 outline_confirm_enabled / clarify_enabled。

验证分支：
1. happy path：waiter 回带编辑后大纲 -> 出 outline_card + 队列用编辑后的标题建。
2. 超时/skip：waiter 回无 outline 的 skip dict -> 用原 decompose 大纲续跑。
3. 空编辑回退：waiter 回 outline=[] -> 用原大纲。
4. 开关关：outline_confirm_enabled=False -> 不出卡、不调 waiter、直接执行。
5. 无 waiter（HTTP 降级）：不注入 wait_for_user_reply -> 不出卡、直接执行。
6. resume awaiting_user@decompose：重发同一份 outline_card、应用回复、不重跑 decompose。
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.agentic.types import LoopOutcome
from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.research.pipeline import ResearchPipeline
from core.stream_bus import StreamBus


# ── helpers ────────────────────────────────────────────────────────────────────


def _settings(*, outline_on: bool = True, clarify_on: bool = False):
    """stub settings：默认关 clarify（避免 rephrase ask_user 干扰），单独控 outline_confirm_enabled。"""
    return type("S", (), {"research": type("R", (), {
        "clarify_enabled": clarify_on,
        "clarify_wait_timeout_s": 120,
        "clarify_max_questions": 3,
        "outline_confirm_enabled": outline_on,
    })()})()


def _ctx(waiter=None, resume_id: str = "") -> UnifiedContext:
    ctx = UnifiedContext(
        course_id="c1", user_message="研究X", enabled_tools=["rag", "web_search"],
        mode="research", session_id="s1",
    )
    if resume_id:
        ctx.metadata["resume_research_id"] = resume_id
    if waiter is not None:
        ctx.metadata["wait_for_user_reply"] = waiter
    return ctx


def _waiter(reply):
    """构造记录调用次数的 waiter。"""
    state = {"called": 0}

    async def _w():
        state["called"] += 1
        return reply

    return _w, state


def _seq_loop(seq: list[str]):
    idx = {"i": 0}

    async def _fake(**kw):
        i = idx["i"]
        idx["i"] = i + 1
        return LoopOutcome(final_text=seq[i] if i < len(seq) else f"## 段落 {i}",
                           rounds=1, tools_used=[], completed=True)

    return _fake


# rephrase / decompose 产物；后续 reporting 由 _seq_loop 兜底段落（_drive_queue 被 spy 不消耗）
_SEQ = [
    "精炼主题：研究导数与积分。",
    '{"sub_topics":[{"title":"导数","overview":"定义"},{"title":"积分","overview":"应用"}]}',
    '{"title":"报告","sections":[]}',
    "## 引言\n概述。",
    "## 结论\n总结。",
]


def _drive_spy(captured: dict):
    """spy _drive_queue：记录入队标题（即建队列用的 sub_topics），不真跑检索。"""
    async def _spy(**kw):
        captured["titles"] = list(kw["queue"].list_titles())
    return AsyncMock(side_effect=_spy)


def _patchers(loop_fake, *, outline_on=True, ckpt_load=None):
    """run_agent_loop + profile/layers/describe/tool_schemas + checkpoint mock + get_settings stub。"""
    return [
        patch("core.research.pipeline.run_agent_loop", new=loop_fake),
        patch("core.research.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=ProfileRuntime())),
        patch("core.research.pipeline.build_common_context_layers", new=AsyncMock(return_value=CommonContextLayers())),
        patch("core.research.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
        patch("core.research.pipeline.get_tool_schemas", return_value=[]),
        patch("core.research.checkpoint.load", new=AsyncMock(return_value=ckpt_load)),
        patch("core.research.checkpoint.save_phase", new=AsyncMock()),
        patch("core.research.checkpoint.clear", new=AsyncMock()),
        patch("core.research.pipeline.get_settings", new=lambda: _settings(outline_on=outline_on)),
    ]


def _event_types(stream: StreamBus) -> set[str]:
    out: set[str] = set()
    for e in stream._history:
        if isinstance(e, dict):
            out.add(str(e.get("type")))
        elif hasattr(e, "type"):
            out.add(e.type.value if hasattr(e.type, "value") else str(e.type))
    return out


async def _run_fresh(waiter, captured, *, outline_on=True):
    """跑一次全新 pipeline.run，spy _drive_queue 捕获建队列标题，返回 stream。"""
    pipe = ResearchPipeline(num_subtopics=2, max_parallel=2)
    ctx = _ctx(waiter=waiter)
    stream = StreamBus()
    patchers = _patchers(_seq_loop(_SEQ), outline_on=outline_on)
    drive_patch = patch.object(pipe, "_drive_queue", _drive_spy(captured))
    for p in patchers + [drive_patch]:
        p.start()
    try:
        await pipe.run(topic="研究导数与积分", context=ctx, stream=stream)
    finally:
        for p in patchers + [drive_patch]:
            p.stop()
    return stream


# ── tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_outline_confirm_applies_edited_outline():
    """WS + 开关开 + waiter 回编辑后大纲 -> 出 outline_card，队列用编辑后标题建。"""
    edited = [{"title": "导数概念", "overview": ""}, {"title": "极限", "overview": ""}, {"title": "微分", "overview": ""}]
    waiter, state = _waiter({"outline": edited})
    captured: dict = {}
    stream = await _run_fresh(waiter, captured)

    assert "outline_card" in _event_types(stream)
    assert state["called"] == 1                      # waiter 被调一次
    assert captured["titles"] == ["导数概念", "极限", "微分"]


@pytest.mark.asyncio
async def test_outline_confirm_skip_keeps_original():
    """waiter 超时/skip（回无 outline 的 dict）-> 用原 decompose 大纲续跑。"""
    waiter, _ = _waiter({"text": "", "answers": None})
    captured: dict = {}
    await _run_fresh(waiter, captured)
    assert captured["titles"] == ["导数", "积分"]      # _SEQ 里 decompose 的原大纲


@pytest.mark.asyncio
async def test_outline_confirm_empty_edit_falls_back():
    """waiter 回 outline=[]（学生全删空）-> 保持原大纲。"""
    waiter, _ = _waiter({"outline": []})
    captured: dict = {}
    await _run_fresh(waiter, captured)
    assert captured["titles"] == ["导数", "积分"]


@pytest.mark.asyncio
async def test_outline_confirm_disabled_skips_card():
    """outline_confirm_enabled=False -> 不出 outline_card、不调 waiter、直接执行。"""
    waiter, state = _waiter({"outline": [{"title": "不应被用"}]})
    captured: dict = {}
    stream = await _run_fresh(waiter, captured, outline_on=False)
    assert "outline_card" not in _event_types(stream)
    assert state["called"] == 0
    assert captured["titles"] == ["导数", "积分"]


@pytest.mark.asyncio
async def test_outline_confirm_no_waiter_degrades():
    """无 wait_for_user_reply（HTTP 入口）-> 不出卡、直接执行（同 rephrase ask_user 降级）。"""
    captured: dict = {}
    stream = await _run_fresh(waiter=None, captured=captured)
    assert "outline_card" not in _event_types(stream)
    assert captured["titles"] == ["导数", "积分"]


@pytest.mark.asyncio
async def test_outline_resume_awaiting_user_reposts_and_applies():
    """resume 命中「上次卡在大纲确认」(phase=decompose,status=awaiting_user)：
    重发同一份 outline_card、应用回复、不重跑 decompose。"""
    state = {"refined_topic": "已精炼",
             "sub_topics": [{"title": "原A", "overview": ""}, {"title": "原B", "overview": ""}]}
    pending = {"topic": "已精炼", "sub_topics": state["sub_topics"]}
    ckpt = SimpleNamespace(
        research_id="r_od", phase="decompose", status="awaiting_user",
        state_json=json.dumps(state), pending_question_json=json.dumps(pending),
        topic="已精炼",
    )
    waiter, _ = _waiter({"outline": [{"title": "新A", "overview": "x"}]})
    captured: dict = {}

    pipe = ResearchPipeline(num_subtopics=2, max_parallel=2)
    ctx = _ctx(waiter=waiter, resume_id="r_od")
    stream = StreamBus()
    decompose = AsyncMock()
    # resume 跳过 rephrase/decompose，seq 只需 reporting 三段
    patchers = _patchers(_seq_loop(_SEQ[2:]), outline_on=True, ckpt_load=ckpt)
    patchers.append(patch.object(pipe, "_decompose", decompose))
    drive_patch = patch.object(pipe, "_drive_queue", _drive_spy(captured))
    for p in patchers + [drive_patch]:
        p.start()
    try:
        await pipe.run(topic="原始", context=ctx, stream=stream)
    finally:
        for p in patchers + [drive_patch]:
            p.stop()

    decompose.assert_not_called()                   # resume 跳过 decompose
    assert "outline_card" in _event_types(stream)   # 重发了同一份大纲卡
    assert captured["titles"] == ["新A"]             # 用了编辑后的标题
