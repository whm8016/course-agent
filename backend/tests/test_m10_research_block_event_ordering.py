"""M-10：Research 并行 block 事件在共享 StreamBus 上交错 → 块级隔离 + flush。

修复前：_drive_queue 用 asyncio.gather 并行跑多个 _research_block，各块共享同一个主
StreamBus，run_agent_loop 的事件按 await 调度交错塞进 _history，前端拿到的 token 序列
跨子主题混在一起，无法归属。

修复后：每块用独立子 StreamBus 缓冲自身事件，块结束后整体 flush 回主 stream（顺序保持、
不交错），并以 progress 事件标记块边界（block_id / sub_topic）。

测试直接驱动 _drive_queue / _research_block（聚焦 block 路径，避免 rephrase/decompose/
reporting 阶段的事件干扰断言）。
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from core.agentic.types import LoopOutcome
from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.research.data_structures import DynamicTopicQueue
from core.research.pipeline import ResearchPipeline
from core.stream_bus import StreamBus


async def _collect(bus: StreamBus):
    if not bus._closed:
        await bus.close()
    return [e.to_dict() async for e in bus.subscribe()]


def _pipeline_ready() -> tuple[ResearchPipeline, dict[str, Any]]:
    """构造一个已就绪（_rt/_layers 已设）的 pipeline + 一个 research_step cfg。"""
    pipeline = ResearchPipeline(num_subtopics=2, max_parallel=2, block_max_iterations=2)
    pipeline._rt = ProfileRuntime()
    pipeline._layers = CommonContextLayers()
    cfg = {
        "research_step": {
            "system": "研究 {topic} 的子主题：{block_title}。{block_overview}.{kb_note}",
            "user_template": "兄弟主题：{sibling_topics}",
        }
    }
    return pipeline, cfg


def test_parallel_blocks_events_not_interleaved():
    """两 block 并行：各自 token 在主 stream 上连续成段，不交错，且有块边界。"""
    pipeline, cfg = _pipeline_ready()
    ctx = UnifiedContext(
        course_id="C1", user_message="t",
        enabled_tools=["web_search"], mode="research",
    )
    queue = DynamicTopicQueue("rid", max_length=8)
    queue.add_block("导数", "")
    queue.add_block("积分", "")

    # 按子主题区分 emit 内容：导数块→A1/A2，积分块→B1/B2
    async def _fake_loop(*, context, stream, system_prompt, **kw):
        if "导数" in system_prompt:
            await stream.emit({"type": "token", "content": "A1"})
            await stream.emit({"type": "token", "content": "A2"})
            return LoopOutcome(final_text="导数刻画函数在某一点的瞬时变化率，是微积分的核心概念，几何上表示切线的斜率 [来源: https://a.com]", rounds=1, completed=True)
        if "积分" in system_prompt:
            await stream.emit({"type": "token", "content": "B1"})
            await stream.emit({"type": "token", "content": "B2"})
            return LoopOutcome(final_text="积分用于求面积与累积量，是微积分的基本运算，与导数互为逆运算（微积分基本定理） [来源: https://b.com]", rounds=1, completed=True)
        return LoopOutcome(final_text="这是一个足够长的段落占位以通过块自检 [来源: https://default.com]", rounds=1, completed=True)

    async def _go():
        stream = StreamBus()
        with (
            patch("core.research.pipeline.run_agent_loop", new=_fake_loop),
            patch("core.research.pipeline.describe_images",
                  new=AsyncMock(side_effect=lambda c, t, r: t)),
        ):
            await pipeline._drive_queue(
                queue=queue, topic="微积分", context=ctx, stream=stream, cfg=cfg
            )
        return stream

    stream = asyncio.run(_go())
    events = asyncio.run(_collect(stream))

    token_contents = [
        e["content"] for e in events
        if e.get("type") == "token" and e.get("content") in ("A1", "A2", "B1", "B2")
    ]
    assert len(token_contents) == 4, f"应有 4 个块 token，实际 {token_contents}"
    # 修复后：A 连续 + B 连续（不交错）
    a_indices = [i for i, t in enumerate(token_contents) if t.startswith("A")]
    b_indices = [i for i, t in enumerate(token_contents) if t.startswith("B")]
    assert max(a_indices) - min(a_indices) == 1, f"A token 不连续（交错）：{token_contents}"
    assert max(b_indices) - min(b_indices) == 1, f"B token 不连续（交错）：{token_contents}"

    # 块边界 progress 事件存在且含 block_id
    block_starts = [e for e in events
                    if e.get("type") == "progress" and e.get("status") == "block_start"]
    block_ends = [e for e in events
                  if e.get("type") == "progress" and e.get("status") == "block_end"]
    assert len(block_starts) == 2 and len(block_ends) == 2
    assert all("block_id" in e for e in block_starts)


def test_block_boundary_wraps_its_tokens():
    """单个 block：progress(block_start) → 该块 token → progress(block_end)，顺序正确。"""
    pipeline, cfg = _pipeline_ready()
    ctx = UnifiedContext(
        course_id="C1", user_message="t",
        enabled_tools=["web_search"], mode="research",
    )
    queue = DynamicTopicQueue("rid2", max_length=8)
    queue.add_block("单主题", "")

    async def _fake_loop(*, context, stream, system_prompt, **kw):
        await stream.emit({"type": "token", "content": "X1"})
        await stream.emit({"type": "token", "content": "X2"})
        return LoopOutcome(final_text="结论部分给出完整的推导与说明，包含主要定理及其几何意义，可供后续报告引用 [来源: https://x.com]", rounds=1, completed=True)

    async def _go():
        stream = StreamBus()
        with (
            patch("core.research.pipeline.run_agent_loop", new=_fake_loop),
            patch("core.research.pipeline.describe_images",
                  new=AsyncMock(side_effect=lambda c, t, r: t)),
        ):
            await pipeline._drive_queue(
                queue=queue, topic="单主题", context=ctx, stream=stream, cfg=cfg
            )
        return stream

    stream = asyncio.run(_go())
    events = asyncio.run(_collect(stream))

    # researching 阶段的事件序列：block_start → X1 → X2 → block_end
    researching_seq = [
        e for e in events
        if (e.get("type") == "progress" and e.get("stage") == "researching")
        or (e.get("type") == "token" and e.get("content") in ("X1", "X2"))
    ]
    statuses = [(e.get("status") or e.get("content")) for e in researching_seq]
    assert statuses[0] == "block_start", f"应以 block_start 开头：{statuses}"
    assert statuses[-1] == "block_end", f"应以 block_end 结尾：{statuses}"
    assert statuses[1:3] == ["X1", "X2"], f"块 token 应夹在边界内：{statuses}"


def test_child_stream_is_not_the_main_stream():
    """验证修复确实用了独立子 bus：run_agent_loop 收到的 stream 不是主 stream（是子 bus）。

    修复前：run_agent_loop 收到的就是主 stream（事件直接进主 stream → 交错）。
    修复后：run_agent_loop 收到的是独立子 bus（事件先缓冲）。
    """
    pipeline, cfg = _pipeline_ready()
    ctx = UnifiedContext(
        course_id="C1", user_message="t",
        enabled_tools=["web_search"], mode="research",
    )
    queue = DynamicTopicQueue("rid3", max_length=8)
    queue.add_block("T", "")

    captured_streams: list[Any] = []

    async def _fake_loop(*, context, stream, system_prompt, **kw):
        captured_streams.append(stream)
        # 在子 bus 上 emit，验证它不会直接出现在主 stream（直到 flush）
        await stream.emit({"type": "token", "content": "Z"})
        return LoopOutcome(final_text="这是一个足够长的有效研究摘要，包含关键结论与来源标注，供断言子 bus 隔离 [来源: https://z.com]", rounds=1, completed=True)

    main_stream = StreamBus()

    async def _go():
        with (
            patch("core.research.pipeline.run_agent_loop", new=_fake_loop),
            patch("core.research.pipeline.describe_images",
                  new=AsyncMock(side_effect=lambda c, t, r: t)),
        ):
            await pipeline._drive_queue(
                queue=queue, topic="T", context=ctx, stream=main_stream, cfg=cfg
            )

    asyncio.run(_go())

    assert len(captured_streams) == 1
    child = captured_streams[0]
    # run_agent_loop 收到的是独立子 bus，不是主 stream
    assert child is not main_stream, "应使用独立子 bus 隔离，而非直接传主 stream"
