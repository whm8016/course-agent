"""步骤6：QuizPipeline 三修测试。

验证 quiz pipeline 经 pipeline_common 注入（修复三重"没拼全"）：
  1. resolve_profile_runtime 的 model/binding 透传全部 run_agent_loop（explore/plan/quiz/critic）
  2. describe_images 收到 rt —— 视觉判断用用户选模型，不再只传 user_id
  3. 各处 system_prompt 叠加通用上下文层（course_prompt 出现在 system 里）
  4. build_common_context_layers 以 include_skills=False 调用 —— quiz 不挂 read_skill
  5. 出题定向薄弱点：plan 轮注入清单 + target_topic_id 透传 + metadata 计数

asyncio_mode=auto，async def test 自动按 asyncio 跑。
"""
from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.question.pipeline import QuizPipeline
from core.stream_bus import StreamBus

_PLAN_TEXT = (
    '{"templates":[{"topic":"导数定义",'
    '"question_type":"written","difficulty":"easy"}]}'
)
_QUIZ_TEXT = '{"question":"求 f(x)=x² 的导数","correct_answer":"2x","explanation":"幂函数法则"}'


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
    # total_usage 显式给真值：pipeline 的 usage 聚合做 samples 比较，MagicMock 属性会炸
    from core.observability.cost import TokenUsage
    m.total_usage = TokenUsage(input_tokens=100, output_tokens=20)
    return m


async def _run_pipeline(
    *,
    plan_text: str = _PLAN_TEXT,
    quiz_text: str = _QUIZ_TEXT,
    layers: CommonContextLayers | None = None,
    weak: list[dict] | None = None,
) -> tuple[dict, dict, ProfileRuntime]:
    """共享 mock 栈跑一次 QuizPipeline.run，返回 (result, captured, rt)。

    fake loop 按消息内容路由（并发安全，同 test_m8_m9 的 _mock_loop_dispatch 口径）：
    出题要求 → explore、请规划 → plan、题目蓝图 → quiz 生成、待校验题目 → critic
    （critic 回探索摘要文本 = 不可解析 → fail-open 放行）。captured 记录各轮
    kwargs、plan 轮 user_message、describe_images 收到的 rt、build mock。
    """
    ctx = _ctx()
    rt = ProfileRuntime(client=MagicMock(name="client"), text_model="gpt-4o", binding="openai")
    layers = layers or CommonContextLayers()
    build_mock = AsyncMock(return_value=layers)
    captured: dict = {"loops": [], "plan_user_message": "", "describe_rt": None, "build": build_mock}

    async def _fake_describe(_ctx, base_text, _rt):
        captured["describe_rt"] = _rt
        return base_text

    async def _fake_loop(**kw):
        captured["loops"].append(kw)
        um = kw["context"].user_message
        if "请规划" in um:
            captured["plan_user_message"] = um
            return _outcome(plan_text)
        if um.startswith("题目蓝图："):
            return _outcome(quiz_text)
        if um.startswith("待校验题目："):
            return _outcome("探索摘要：导数描述变化率")  # 非 JSON → fail-open 放行
        return _outcome("探索摘要：导数描述变化率")  # explore

    bus = StreamBus()
    patches = [
        patch("core.question.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=rt)),
        patch("core.question.pipeline.build_common_context_layers", new=build_mock),
        patch("core.question.pipeline.describe_images", new=AsyncMock(side_effect=_fake_describe)),
        patch("core.question.pipeline.get_tool_schemas", return_value=[{"name": "rag"}]),
        patch("core.question.pipeline.run_agent_loop", new=AsyncMock(side_effect=_fake_loop)),
    ]
    if weak is not None:
        patches.append(
            patch("core.memory.mastery.get_weak_topics", new=AsyncMock(return_value=weak))
        )
    try:
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = await QuizPipeline().run("出 1 道导数题", ctx, bus, count=1)
    finally:
        await bus.close()
    return result, captured, rt


async def test_quiz_threads_profile_and_injects_layers():
    """全部 run_agent_loop 都收到 model/binding + 通用层叠加；describe_images 收到 rt。"""
    result, captured, rt = await _run_pipeline(
        layers=CommonContextLayers(course_prompt="你是数学课助教")
    )

    # 4 处 run_agent_loop（explore + plan + quiz 生成 + critic 校验轮）都收到
    # 用户选的 model/binding。critic 轮 mock 返回非 JSON → fail-open 放行（题保留）。
    assert len(captured["loops"]) == 4
    for kw in captured["loops"]:
        assert kw["model"] == "gpt-4o"
        assert kw["binding"] == "openai"
        assert kw["client"] is rt.client
        # 通用层叠加进全部 system_prompt（course_prompt）
        assert "数学课助教" in kw["system_prompt"]
    # describe_images 收到同一 rt（视觉判断用 gpt-4o/openai）
    assert captured["describe_rt"] is rt
    # build_common_context_layers 以 include_skills=False 调用（quiz 不挂 read_skill）
    assert captured["build"].call_args.kwargs.get("include_skills", False) is False
    # quiz 产出 1 题
    assert len(result["questions"]) == 1
    assert result["questions"][0]["question"] == "求 f(x)=x² 的导数"


async def test_quiz_plan_injects_weak_topics():
    """出题定向薄弱点：清单非空 → plan 轮 user_message 含 label，metadata 计数正确，
    target_topic_id 从蓝图透传到题目。"""
    weak = [{"topic_id": "deriv", "label": "导数定义", "risk": 0.81, "observation_count": 3}]
    plan_text = (
        '{"templates":[{"topic":"导数定义","question_type":"written",'
        '"difficulty":"easy","target_topic_id":"deriv"}]}'
    )
    result, captured, _rt = await _run_pipeline(plan_text=plan_text, weak=weak)

    # plan 轮消息含薄弱点清单（label + topic_id 都在，LLM 才能定向）
    assert "导数定义" in captured["plan_user_message"]
    assert "topic_id=deriv" in captured["plan_user_message"]
    # metadata 计数（评测读）
    assert result["metadata"]["weak_topics_considered"] == 1
    assert result["metadata"]["weak_topics_targeted"] == 1
    # target_topic_id 从蓝图透传到题目（闭环评测对账的落点）
    assert result["questions"][0]["target_topic_id"] == "deriv"
