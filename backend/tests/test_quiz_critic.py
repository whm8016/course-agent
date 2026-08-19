"""出题 critic 校验轮 + coding 题最小沙箱测试。

覆盖（plan §quiz-critic-sandbox）：
  1. critic 先 fail 再 pass → issues 拼回生成器重试，regenerated/passed 计数正确
  2. critic 连续 fail → 重试用尽走 M-8/M-9 同款容错（reason="critic_failed"）
  3. sandbox.run_code：通过 / 非零退出 / 超时 三态 + 围栏代码块抽取
  4. critic LLM 不可用 → fail-open（仅沙箱程序化结论生效，不误杀好题）
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
        course_id="",
        user_id="",  # 空 user/course：跳过薄弱点查询，聚焦 critic 路径
        enabled_tools=[],
    )


def _outcome(text: str) -> MagicMock:
    m = MagicMock()
    m.final_text = text
    m.tools_used = []
    return m


def _harness(critic_texts: list[str], quiz_texts: list[str]):
    """fake loop 按消息前缀分流：待校验题目 → critic 轮，其余 → 生成轮。

    critic 轮按序消费 critic_texts（耗尽后复用最后一条），生成轮同吃 quiz_texts。
    用作 context manager，yield captured（每轮 user_message，供断言重试回填）。
    """
    import contextlib

    captured: dict = {"messages": [], "regen_messages": []}
    n_critic = {"i": 0}
    n_gen = {"i": 0}

    async def _fake_loop(**kw):
        um = kw["context"].user_message
        captured["messages"].append(um)
        if um.startswith("待校验题目："):
            n_critic["i"] += 1
            return _outcome(critic_texts[min(n_critic["i"], len(critic_texts)) - 1])
        if um.startswith("题目蓝图："):
            n_gen["i"] += 1
            if n_gen["i"] > 1:  # 第 2+ 次生成轮 = critic 触发的重试
                captured["regen_messages"].append(um)
            return _outcome(quiz_texts[min(n_gen["i"], len(quiz_texts)) - 1])
        # explore / plan 轮（出题要求 / 请规划）
        return _outcome(_PLAN_TEXT if "请规划" in um else "探索摘要：导数描述变化率")

    rt = ProfileRuntime(client=MagicMock(name="client"), text_model="gpt-4o", binding="openai")
    layers = CommonContextLayers()
    patches = (
        patch("core.question.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=rt)),
        patch("core.question.pipeline.build_common_context_layers",
              new=AsyncMock(return_value=layers)),
        patch("core.question.pipeline.describe_images", new=AsyncMock(return_value="出题要求")),
        patch("core.question.pipeline.get_tool_schemas", return_value=[]),
        patch("core.question.pipeline.run_agent_loop", new=AsyncMock(side_effect=_fake_loop)),
    )

    @contextlib.contextmanager
    def _ctxs():
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield captured

    return _ctxs()


_PLAN_TEXT = '{"templates":[{"topic":"导数定义","question_type":"written","difficulty":"easy"}]}'
_QUIZ_TEXT = '{"question":"求 f(x)=x² 的导数","correct_answer":"2x","explanation":"幂函数法则"}'
_CRITIC_FAIL = '{"pass": false, "issues": ["解析说幂函数法则但未说明 x 为任意实数的前提"]}'
_CRITIC_PASS = '{"pass": true, "issues": []}'


async def test_critic_fail_then_pass_retries_with_issues():
    """critic 首 fail → issues 拼回生成器重试 → pass：计数与回填内容正确。"""
    with _harness([_CRITIC_FAIL, _CRITIC_PASS], [_QUIZ_TEXT, _QUIZ_TEXT]) as captured:
        result = await QuizPipeline().run("出 1 道导数题", _ctx(), StreamBus(), count=1)

    assert len(result["questions"]) == 1  # 重试后过 critic，题保留
    md = result["metadata"]
    assert md["critic_regenerated"] == 1
    assert md["critic_passed"] == 1
    assert md["critic_failed"] == 0
    # 重试轮消息：原蓝图 + 校验问题回填（生成器才知道改什么）
    assert len(captured["regen_messages"]) == 1
    regen = captured["regen_messages"][0]
    assert "上一版未通过校验" in regen
    assert "幂函数法则" in regen  # critic 的 issues 原文进了重试消息
    assert "请修正后重新给出完整 JSON" in regen


async def test_critic_fail_all_marks_error():
    """critic 连续 fail（初始 + 2 次重试全失败）→ critic_failed 容错，整批不中断。"""
    with _harness([_CRITIC_FAIL], [_QUIZ_TEXT]) as captured:  # critic 恒 fail；重试产物同题
        result = await QuizPipeline().run("出 1 道导数题", _ctx(), StreamBus(), count=1)

    assert result["questions"] == []
    assert len(result["metadata"]["errors"]) == 1
    err = result["metadata"]["errors"][0]
    assert err["reason"] == "critic_failed"
    assert err["critic_issues"]  # 失败原因带出（排查用）
    md = result["metadata"]
    assert md["critic_failed"] == 1
    assert md["critic_passed"] == 0
    assert md["critic_regenerated"] == 2  # 初始 + 2 次重试 = 3 次 critic，前 2 次触发重试
    # 3 次 critic（1 初始 + 2 重试后复验），全部 fail
    assert sum(1 for m in captured["messages"] if m.startswith("待校验题目：")) == 3


async def test_critic_fail_without_issues_not_silently_passed():
    """critic 判 pass=false 但 issues 缺失 → 兜底 issue 触发重试，不静默放行。

    fail-open 只覆盖「不知道」（异常/不可解析）；「明确判否」被放行会让 critic 轮
    空转（日志全绿、计数照加）。
    """
    with _harness(['{"pass": false, "issues": []}'], [_QUIZ_TEXT, _QUIZ_TEXT]) as captured:
        result = await QuizPipeline().run("出 1 道导数题", _ctx(), StreamBus(), count=1)

    # issues 空被兜底 → 两次 critic 都判否 → 重试用尽 critic_failed（不是静默出题）
    assert result["questions"] == []
    assert result["metadata"]["errors"][0]["reason"] == "critic_failed"
    assert result["metadata"]["critic_failed"] == 1
    assert any("未给出具体问题清单" in m for m in captured["regen_messages"])


# ── sandbox.run_code 三态 + 抽取 ──────────────────────────────────────────────


def test_extract_code_blocks():
    from core.question.sandbox import extract_code_blocks

    text = (
        "答案如下：\n```python\nprint(1)\n```\n解析见\n```\nx = 2\n```\n"
        "```Python\nprint(3)\n```\n```python3\nprint(4)\n```\n```text\n不是代码\n```"
    )
    blocks = extract_code_blocks(text)
    # python/裸/Python/python3 围栏都抽（围栏头大小写不敏感），text 块跳过
    assert blocks == ["print(1)", "x = 2", "print(3)", "print(4)"]
    assert extract_code_blocks("无代码块") == []


def test_run_code_pass_fail_timeout():
    from core.question.sandbox import run_code

    ok, detail = run_code("print(1 + 1)")
    assert ok is True and detail == ""

    bad, detail = run_code("raise ValueError('boom')")
    assert bad is False
    assert "boom" in detail

    slow, detail = run_code("while True:\n    pass", timeout=0.3)
    assert slow is False
    assert "timeout" in detail


async def test_critic_llm_down_fails_open_sandbox_still_counts():
    """critic LLM 不可用（异常）→ fail-open 不误杀；沙箱结论仍生效（coding 跑不过照样拦）。"""
    from core.question.pipeline import _run_critic

    q = {
        "question_type": "coding",
        "topic": "python 基础",
        "difficulty": "easy",
        "question": "写代码抛出 ValueError",
        "options": None,
        "correct_answer": "```python\nraise ValueError('boom')\n```",
        "explanation": "见答案",
    }
    rt = ProfileRuntime(client=MagicMock(name="client"), text_model="m", binding="openai")
    bus = StreamBus()
    with patch("core.question.pipeline.run_agent_loop",
               AsyncMock(side_effect=RuntimeError("LLM down"))):
        issues = await _run_critic(
            q, "critic system", _ctx(), CommonContextLayers(), rt, bus
        )
    assert len(issues) == 1
    assert "执行失败" in issues[0] and "boom" in issues[0]
    await bus.close()


# ── A2：coding 题测试用例的 critic 校验（run_tests 全过才放行） ────────────────


def _coding_q_with_tests() -> dict:
    return {
        "question_type": "coding",
        "topic": "python 基础",
        "difficulty": "easy",
        "question": "实现 add(a, b) 返回两数之和",
        "options": None,
        "correct_answer": "```python\ndef add(a, b):\n    return a + b\n```",
        "explanation": "见答案",
        "language": "python",
        "starter_code": "def add(a, b):\n",
        "test_cases": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    }


async def test_critic_coding_reference_fails_tests():
    """参考答案未通过 test_cases → 逐条计 issue（重试依据），LLM down 时沙箱结论独立生效。"""
    from core.question.pipeline import _run_critic
    from core.question.sandbox import TestRunResult

    rt = ProfileRuntime(client=MagicMock(name="client"), text_model="m", binding="openai")
    bus = StreamBus()
    fake = TestRunResult(passed=[1], failed=[(0, "exit=1: AssertionError")], total=2)
    with patch("core.question.sandbox.run_tests", return_value=fake) as m:
        with patch("core.question.pipeline.run_agent_loop",
                   AsyncMock(side_effect=RuntimeError("LLM down"))):
            issues = await _run_critic(
                _coding_q_with_tests(), "critic system", _ctx(),
                CommonContextLayers(), rt, bus,
            )
    m.assert_called_once()  # critic 轮确实跑了用例校验
    assert len(issues) == 1
    assert "第1条测试用例" in issues[0] and "AssertionError" in issues[0]
    await bus.close()


async def test_critic_coding_reference_passes_tests():
    """参考答案全过 test_cases → 无用例 issue；空 test_cases（存量题形状）不调 run_tests。"""
    from core.question.pipeline import _run_critic
    from core.question.sandbox import TestRunResult

    rt = ProfileRuntime(client=MagicMock(name="client"), text_model="m", binding="openai")
    bus = StreamBus()
    with patch("core.question.sandbox.run_tests",
               return_value=TestRunResult(passed=[0, 1], total=2)) as m:
        with patch("core.question.pipeline.run_agent_loop",
                   AsyncMock(side_effect=RuntimeError("LLM down"))):
            issues = await _run_critic(
                _coding_q_with_tests(), "critic system", _ctx(),
                CommonContextLayers(), rt, bus,
            )
    m.assert_called_once()
    assert issues == []  # 代码块可跑 + 用例全过 + LLM down fail-open

    # 存量题形状：无 test_cases → 只走既有围栏块校验，不触发用例校验
    legacy = _coding_q_with_tests()
    legacy["test_cases"] = []
    with patch("core.question.sandbox.run_tests") as m2:
        with patch("core.question.pipeline.run_agent_loop",
                   AsyncMock(side_effect=RuntimeError("LLM down"))):
            issues = await _run_critic(
                legacy, "critic system", _ctx(), CommonContextLayers(), rt, bus,
            )
    m2.assert_not_called()
    assert issues == []
    await bus.close()
