"""步骤6：QuizPipeline 三修测试。

验证 quiz pipeline 经 pipeline_common 注入（修复三重"没拼全"）：
  1. resolve_profile_runtime 的 model/binding 透传全部 3 处 run_agent_loop（explore/plan/quiz）
  2. describe_images 收到 rt —— 视觉判断用用户选模型，不再只传 user_id
  3. 3 处 system_prompt 叠加通用上下文层（course_prompt 出现在 system 里）
  4. build_common_context_layers 以 include_skills=False 调用 —— quiz 不挂 read_skill

asyncio_mode=auto，async def test 自动按 asyncio 跑。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.question.pipeline import QuizPipeline
from core.stream_bus import StreamBus


def _ctx() -> UnifiedContext:
    return UnifiedContext(
        user_message="出 1 道导数题",
        course_id="C1",
        user_id="U1",
        enabled_tools=["rag"],
    )


def _outcome(text: str) -> MagicMock:
    m = MagicMock()
    m.final_text = text
    m.tools_used = []
    return m


async def test_quiz_threads_profile_and_injects_layers():
    """3 处 run_agent_loop 都收到 model/binding + 通用层叠加；describe_images 收到 rt。"""
    ctx = _ctx()
    bus = StreamBus()
    rt = ProfileRuntime(client=MagicMock(name="client"), text_model="gpt-4o", binding="openai")
    layers = CommonContextLayers(course_prompt="你是数学课助教")

    captured: dict = {"loops": [], "describe_rt": None}

    async def _fake_describe(_ctx, base_text, _rt):
        captured["describe_rt"] = _rt
        return base_text

    plan_text = (
        '{"templates":[{"topic":"导数定义",'
        '"question_type":"written","difficulty":"easy"}]}'
    )
    quiz_text = '{"question":"求 f(x)=x² 的导数","correct_answer":"2x","explanation":"幂函数法则"}'

    call_n = {"i": 0}

    async def _fake_loop(**kw):
        captured["loops"].append(kw)
        call_n["i"] += 1
        if call_n["i"] == 2:  # plan → JSON 蓝图
            return _outcome(plan_text)
        if call_n["i"] == 3:  # quiz → JSON 题目
            return _outcome(quiz_text)
        return _outcome("探索摘要：导数描述变化率")  # explore

    with (
        patch(
            "core.question.pipeline.resolve_profile_runtime",
            new=AsyncMock(return_value=rt),
        ),
        patch(
            "core.question.pipeline.build_common_context_layers",
            new=AsyncMock(return_value=layers),
        ) as m_build,
        patch(
            "core.question.pipeline.describe_images",
            new=AsyncMock(side_effect=_fake_describe),
        ),
        patch("core.question.pipeline._get_tool_schemas", return_value=[{"name": "rag"}]),
        patch(
            "core.question.pipeline.run_agent_loop", new=AsyncMock(side_effect=_fake_loop)
        ),
    ):
        result = await QuizPipeline().run("出 1 道导数题", ctx, bus, count=1)

    # 3 处 run_agent_loop（explore + plan + 1 quiz）都收到用户选的 model/binding
    assert len(captured["loops"]) == 3
    for kw in captured["loops"]:
        assert kw["model"] == "gpt-4o"
        assert kw["binding"] == "openai"
        assert kw["client"] is rt.client
    # describe_images 收到同一 rt（视觉判断用 gpt-4o/openai）
    assert captured["describe_rt"] is rt
    # build_common_context_layers 以 include_skills=False 调用（quiz 不挂 read_skill）
    assert m_build.call_args.kwargs.get("include_skills", False) is False
    # 通用层叠加进全部 3 处 system_prompt（course_prompt）
    for kw in captured["loops"]:
        assert "数学课助教" in kw["system_prompt"]
    # quiz 产出 1 题
    assert len(result["questions"]) == 1
    assert result["questions"][0]["question"] == "求 f(x)=x² 的导数"
    await bus.close()
