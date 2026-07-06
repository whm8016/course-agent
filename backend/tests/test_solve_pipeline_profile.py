"""步骤5：DeepSolvePipeline 三修测试。

验证 solve pipeline 经 pipeline_common 注入（修复三重"没拼全"）：
  1. resolve_profile_runtime 解析的 model/binding 透传 run_agent_loop —— profile 真生效，
     不再退回全局 TEXT_MODEL（用户选 gpt-4o 解题就用 gpt-4o）
  2. describe_images 收到 rt —— 视觉判断用用户选的模型，不再只传 user_id 误走两阶段
  3. system_prompt 叠加通用上下文层（course_prompt / memory），solve YAML 的 system 不再孤立
  4. build_common_context_layers 以 include_skills=False 调用 —— solve 不挂 read_skill，不注入 always_skills

asyncio_mode=auto，async def test 自动按 asyncio 跑。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.solve.pipeline import DeepSolvePipeline
from core.stream_bus import StreamBus


def _ctx() -> UnifiedContext:
    return UnifiedContext(
        user_message="解 x+1=3",
        course_id="C1",
        user_id="U1",
        enabled_tools=["rag"],
        llm_profile_id=None,
    )


async def test_solve_threads_profile_model_and_binding():
    """resolve_profile_runtime → gpt-4o/openai 透传 run_agent_loop（profile 真生效）。"""
    ctx = _ctx()
    bus = StreamBus()
    rt = ProfileRuntime(client=MagicMock(name="client"), text_model="gpt-4o", binding="openai")

    captured: dict = {}

    async def _fake_describe(_ctx, base_text, _rt):
        captured["describe_rt"] = _rt
        return base_text  # 不改文字

    fake_outcome = MagicMock(rounds=1, tools_used=["solve_plan"], final_text="x=2")

    with (
        patch("core.solve.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=rt)),
        patch(
            "core.solve.pipeline.build_common_context_layers",
            new=AsyncMock(return_value=CommonContextLayers()),
        ) as m_build,
        patch(
            "core.solve.pipeline.describe_images", new=AsyncMock(side_effect=_fake_describe)
        ),
        patch("core.solve.pipeline._get_tool_schemas", return_value=[]),
        patch(
            "core.solve.pipeline.run_agent_loop", new=AsyncMock(return_value=fake_outcome)
        ) as m_loop,
    ):
        result = await DeepSolvePipeline().run("解 x+1=3", ctx, bus)

    # run_agent_loop 收到用户选的 model/binding（不再退回全局 TEXT_MODEL）
    kw = m_loop.call_args.kwargs
    assert kw["model"] == "gpt-4o"
    assert kw["binding"] == "openai"
    assert kw["client"] is rt.client
    # describe_images 收到同一 rt（视觉判断用 gpt-4o/openai，不再只传 user_id）
    assert captured["describe_rt"] is rt
    # build_common_context_layers 以 include_skills=False 调用（solve 不挂 read_skill）
    assert m_build.call_args.kwargs.get("include_skills", False) is False
    # 返回结构完整
    assert result["final_answer"] == "x=2"
    await bus.close()


async def test_solve_injects_common_context_layers():
    """通用上下文层（course_prompt / memory）叠加进 system_prompt，原先 solve 看不到。"""
    ctx = _ctx()
    bus = StreamBus()
    rt = ProfileRuntime()
    layers = CommonContextLayers(
        course_prompt="你是数学课助教",  # 课程设定，原先 solve pipeline 看不到
        memory_context="[记忆] 该生偏好详细步骤",
    )

    with (
        patch("core.solve.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=rt)),
        patch(
            "core.solve.pipeline.build_common_context_layers",
            new=AsyncMock(return_value=layers),
        ),
        patch(
            "core.solve.pipeline.describe_images",
            new=AsyncMock(side_effect=lambda c, t, r: t),
        ),
        patch("core.solve.pipeline._get_tool_schemas", return_value=[]),
        patch(
            "core.solve.pipeline.run_agent_loop",
            new=AsyncMock(return_value=MagicMock(rounds=1, tools_used=[], final_text="ok")),
        ) as m_loop,
    ):
        await DeepSolvePipeline().run("解 x+1=3", ctx, bus)

    sys_prompt = m_loop.call_args.kwargs["system_prompt"]
    # task_system（solve YAML 的 system 键）在前，通用层叠加在后
    assert "数学课助教" in sys_prompt
    assert "该生偏好详细步骤" in sys_prompt
    # layers 里 now_text/always_skills 为空 → 被过滤，不出现在 system 里
    assert "【当前时间】" not in sys_prompt
    await bus.close()
