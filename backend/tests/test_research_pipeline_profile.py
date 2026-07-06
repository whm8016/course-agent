"""步骤7：ResearchPipeline 三修测试。

验证 research pipeline 经 pipeline_common 注入（修复三重"没拼全"）：
  1. resolve_profile_runtime 的 model/binding 透传全部 run_agent_loop 调用点
     （_rephrase / _decompose / _research_block / _gen_report_outline / _one_shot_report）
  2. describe_images 收到 self._rt（_rephrase 阶段，视觉判断用用户选模型）
  3. system_prompt 叠加通用上下文层（course_prompt 出现在 system 里）
  4. build_common_context_layers 以 include_skills=False 调用

复用 test_research_pipeline 的 4 阶段 seq 驱动 + asyncio.run。
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.agentic.types import LoopOutcome
from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.research.pipeline import ResearchPipeline


def _make_context() -> UnifiedContext:
    return UnifiedContext(
        course_id="course_1",
        user_message="研究导数与积分",
        enabled_tools=["rag", "web_search"],
        mode="research",
        session_id="sess_1",
    )


def _fake_loop_factory(captured: dict):
    """按调用顺序产出阶段产物（复用 test_research_pipeline 的 seq），并捕获 kwargs。

    research blocks 用 asyncio.gather 并行，seq[2]/seq[3] 内容对称故乱序无影响；
    outline/intro/section/conclusion 在 research 后串行，idx 继续递增。
    """
    seq = [
        # 0 rephrase
        "精炼主题：研究微积分中导数与积分的核心概念。",
        # 1 decompose
        ('{"sub_topics":['
         '{"title":"导数定义","overview":"变化率"},'
         '{"title":"积分应用","overview":"累积"}]}'),
        # 2/3 research blocks（并行，顺序不定但内容对称）
        "导数是瞬时变化率 [来源: https://wikipedia.org/Derivative]。",
        "积分用于求面积 [来源: 积分教材.pdf]。",
        # 4 outline
        ('{"title":"报告","sections":['
         '{"id":"S1","title":"导数","intent":"","block_ids":["block_1"]},'
         '{"id":"S2","title":"积分","intent":"","block_ids":["block_2"]}]}'),
        # 5 intro / 6 section1 / 7 section2 / 8 conclusion
        "## 1. 引言\n微积分。",
        "## 2. 导数\n导数刻画变化率 [1]。",
        "## 3. 积分\n积分刻画累积 [2]。",
        "## 4. 结论\n互为逆运算。",
    ]
    idx = {"i": 0}

    async def _fake(*, context, stream, system_prompt, tool_schemas=None, model=None,
                    client=None, binding=None, emit_terminal_events=True, max_iterations=10, **kw):
        i = idx["i"]
        idx["i"] = i + 1
        captured["loops"].append({
            "system_prompt": system_prompt, "model": model, "binding": binding,
            "emit_terminal_events": emit_terminal_events,
        })
        text = seq[i] if i < len(seq) else f"## 段落 {i}"
        return LoopOutcome(final_text=text, rounds=1, tools_used=[], completed=True)

    return _fake


def test_research_threads_profile_and_injects_layers():
    pipeline = ResearchPipeline(num_subtopics=2, max_parallel=2)
    ctx = _make_context()
    rt = ProfileRuntime(client=MagicMock(name="client"), text_model="gpt-4o", binding="openai")
    layers = CommonContextLayers(course_prompt="COURSE_MARKER_数学课")
    captured: dict = {"loops": [], "describe_rt": None, "include_skills": None}

    async def _fake_describe(_ctx, base_text, _rt):
        captured["describe_rt"] = _rt
        return base_text

    async def _go():
        from core.stream_bus import StreamBus
        stream = StreamBus()
        with (
            patch("core.research.pipeline.run_agent_loop", new=_fake_loop_factory(captured)),
            patch(
                "core.research.pipeline.resolve_profile_runtime",
                new=AsyncMock(return_value=rt),
            ),
            patch(
                "core.research.pipeline.build_common_context_layers",
                new=AsyncMock(return_value=layers),
            ) as m_build,
            patch(
                "core.research.pipeline.describe_images",
                new=AsyncMock(side_effect=_fake_describe),
            ),
        ):
            result = await pipeline.run(topic="研究导数与积分", context=ctx, stream=stream)
        captured["include_skills"] = m_build.call_args.kwargs.get("include_skills", False)
        await stream.close()
        return result

    result = asyncio.run(_go())

    # 全部 run_agent_loop 调用点都透传用户选的 model/binding（不再退回全局 TEXT_MODEL）
    assert len(captured["loops"]) >= 5  # rephrase/decompose/2 blocks/outline + reporting 段
    # 多阶段 pipeline 的中间 loop 必须不 emit done（emit_terminal_events=False），
    # 否则 rephrase 结束的 done 会让前端误判整个 research 完成而断开 ws（turn.cancel）
    assert all(kw["emit_terminal_events"] is False for kw in captured["loops"])
    for kw in captured["loops"]:
        assert kw["model"] == "gpt-4o"
        assert kw["binding"] == "openai"
    # describe_images 收到同一 self._rt（_rephrase 阶段）
    assert captured["describe_rt"] is rt
    # build_common_context_layers 以 include_skills=False 调用（research 不挂 read_skill）
    assert captured["include_skills"] is False
    # 通用层叠加进 system_prompt（course_prompt marker 至少出现在一处）
    assert any("COURSE_MARKER_数学课" in kw["system_prompt"] for kw in captured["loops"])
    # 报告正常产出
    assert "report" in result and result["report"]


def test_research_kb_note_only_when_rag_selected():
    """未选知识库（enabled_tools 不含 rag）时，research block 不应提示'已挂载知识库'。

    根因：_research_block 的 kb_note 旧条件是 `if context.course_id`（任何课程都成立），
    导致即使本轮 enabled_tools 无 rag、tool_schemas=None，仍提示 LLM"调用 rag 检索"——
    有提示无工具，LLM 困惑（用户愤怒的 bug 第二面）。修复后仅在 rag ∈ tools 时注入。
    """

    async def _run_with(tools: list[str]) -> list[dict]:
        from core.stream_bus import StreamBus
        pipeline = ResearchPipeline(num_subtopics=2, max_parallel=2)
        rt = ProfileRuntime(client=MagicMock(name="client"), text_model="m", binding="openai")
        layers = CommonContextLayers()
        captured: dict = {"loops": []}
        with (
            patch("core.research.pipeline.run_agent_loop", new=_fake_loop_factory(captured)),
            patch("core.research.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=rt)),
            patch("core.research.pipeline.build_common_context_layers", new=AsyncMock(return_value=layers)),
            patch("core.research.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, b, r: b)),
        ):
            ctx = UnifiedContext(
                course_id="course_1",
                user_message="研究导数与积分",
                enabled_tools=tools,
                mode="research",
                session_id="sess_1",
            )
            stream = StreamBus()
            await pipeline.run(topic="研究导数与积分", context=ctx, stream=stream)
            await stream.close()
        return captured["loops"]

    # 含 rag：至少 research block 应注入"已挂载知识库"提示
    loops_with_rag = asyncio.run(_run_with(["rag", "web_search"]))
    assert any("已挂载知识库" in kw["system_prompt"] for kw in loops_with_rag)
    # 不含 rag：所有阶段都不应出现"已挂载知识库"（无工具却提示会让 LLM 困惑）
    loops_no_rag = asyncio.run(_run_with(["web_search"]))
    assert all("已挂载知识库" not in kw["system_prompt"] for kw in loops_no_rag)
