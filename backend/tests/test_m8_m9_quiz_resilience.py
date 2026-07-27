"""M-8 + M-9：Quiz 单题容错 + JSON schema 校验。

M-8：单题生成（LLM 异常）不中断整批，失败题记 error 事件并跳过，其余题继续。
M-9：_validate_question 对内容残缺题（空题干/空答案/choice 答案不命中 options）判无效，
同样走容错路径（标记 schema_invalid 跳过），不产出空题目给前端。

二者协同：单题无论是抛异常还是内容不合法，都局部失败，不影响其它题。

注：load_prompt_dict 被 mock，因为 core/question/prompts/zh/pipeline.yaml 存在独立的
YAML 缩进 bug（块标量内容顶格，导致 safe_load 失败、prompt_loader 吞异常返回空 cfg）——
那是预先存在的配置问题（见本批次报告「连带发现」），不在本测试覆盖范围。这里注入合法
cfg 以聚焦验证 M-8/M-9 的容错逻辑本身。
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.question.pipeline import QuizPipeline, _validate_question
from core.stream_bus import StreamBus


async def _collect(bus: StreamBus):
    if not bus._closed:
        await bus.close()
    return [e.to_dict() async for e in bus.subscribe()]


def _ctx() -> UnifiedContext:
    return UnifiedContext(user_message="出 3 道题", course_id="C1", user_id="U1")


# 合法 cfg（结构对齐 pipeline.yaml 的 explore/plan/quiz.system）
_FAKE_CFG = {
    "explore": {"system": "你是出题探索模块。"},
    "plan": {"system": "你是出题规划模块。规划 {count} 道，题型：{valid_types}。"},
    "quiz": {"system": "你是出题生成模块。"},
}


def _mock_loop_dispatch(explore_text, plan_payload, qid_outputs):
    """按 user_message 内容分发 run_agent_loop 返回值（并发安全）。

    explore/plan 按 stage 关键词识别；quiz 阶段从「题目蓝图」里抽 question_id，按题分发对应输出。
    分发依据是各题自己的输入而非全局调用顺序——Stage 3 改并行后调用顺序不确定，必须这样分发
    才不会假阳性（旧的 _mock_loop_seq 按调用计数返回，并行下会把 RAISE 随机落到任意题）。
    qid_outputs: {question_id: "<json 文本>" 或 "RAISE"}。
    """
    async def _fake(**kwargs):
        um = (kwargs.get("context").user_message or "")
        if "请生成第" in um:  # quiz：从 blueprint 抽 question_id
            try:
                bp_raw = um.split("题目蓝图：", 1)[1].split("\n\n", 1)[0]
                qid = json.loads(bp_raw).get("question_id", "")
            except Exception:
                qid = ""
            out = qid_outputs.get(qid, "")
            if out == "RAISE":
                raise RuntimeError(f"模拟 {qid} LLM 超时")
            return MagicMock(rounds=1, tools_used=[], final_text=out)
        if "请规划" in um:  # plan
            return MagicMock(rounds=1, tools_used=[], final_text=plan_payload)
        return MagicMock(rounds=1, tools_used=[], final_text=explore_text)  # explore

    return _fake


# ── M-9：_validate_question 单测 ────────────────────────────────────────────────

def test_validate_rejects_empty_question():
    assert _validate_question(
        {"question_type": "written", "question": "  ", "correct_answer": "ans"}
    ) is False


def test_validate_rejects_empty_answer():
    assert _validate_question(
        {"question_type": "written", "question": "题干", "correct_answer": ""}
    ) is False


def test_validate_accepts_well_formed_written():
    assert _validate_question(
        {"question_type": "written", "question": "解释熵", "correct_answer": "熵增"}
    ) is True


def test_validate_choice_requires_options_and_matching_answer():
    # choice 但 options 缺失
    assert _validate_question(
        {"question_type": "choice", "question": "q", "correct_answer": "A", "options": None}
    ) is False
    # choice 只有 1 个选项（不足 2）
    assert _validate_question(
        {"question_type": "choice", "question": "q", "correct_answer": "A",
         "options": {"A": "1"}}
    ) is False
    # choice 答案不命中 options（指向不存在的 D）
    assert _validate_question(
        {"question_type": "choice", "question": "q", "correct_answer": "D",
         "options": {"A": "1", "B": "2"}}
    ) is False
    # choice 答案大小写不一也能命中（校验内部 upper）
    assert _validate_question(
        {"question_type": "choice", "question": "q", "correct_answer": "b",
         "options": {"A": "1", "B": "2"}}
    ) is True


# ── M-8：单题 LLM 异常不中断整批 ────────────────────────────────────────────────

async def test_quiz_one_question_failure_does_not_abort_batch():
    """3 题：第 1、3 正常，第 2 题 LLM 调用抛异常 → 整批不中断，返回 2 题 + 1 error。"""
    ctx = _ctx()
    bus = StreamBus()
    # Stage 3 改并行后调用顺序不确定，故按 question_id 分发（不依赖调用次序）。
    plan_payload = json.dumps({"templates": [
        {"question_id": "q1", "topic": "T1", "question_type": "written", "difficulty": "easy"},
        {"question_id": "q2", "topic": "T2", "question_type": "written", "difficulty": "easy"},
        {"question_id": "q3", "topic": "T3", "question_type": "written", "difficulty": "easy"},
    ]})
    qid_outputs = {
        "q1": json.dumps({"question_type": "written", "question": "题1", "correct_answer": "答1", "explanation": "e1"}),  # ok
        "q2": "RAISE",  # 异常
        "q3": json.dumps({"question_type": "written", "question": "题3", "correct_answer": "答3", "explanation": "e3"}),  # ok
    }
    fake = _mock_loop_dispatch("探索摘要", plan_payload, qid_outputs)

    with (
        patch("core.question.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=ProfileRuntime())),
        patch("core.question.pipeline.build_common_context_layers", new=AsyncMock(return_value=CommonContextLayers())),
        patch("core.question.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
        patch("core.question.pipeline.load_prompt_dict", return_value=_FAKE_CFG),
        patch("core.question.pipeline.run_agent_loop", new=AsyncMock(side_effect=fake)),
    ):
        result = await QuizPipeline().run("出 3 道题", ctx, bus, count=3)

    # 修复前：第 2 题异常会让整个 run 抛出，result 拿不到。
    # 修复后：返回 2 道成功题 + 1 个失败标记。
    assert len(result["questions"]) == 2, f"应保留 2 道成功题，实际 {len(result['questions'])}"
    qids = [q["question_id"] for q in result["questions"]]
    assert "q1" in qids and "q3" in qids and "q2" not in qids
    assert result["metadata"]["count_failed"] == 1
    assert result["metadata"]["errors"][0]["reason"] == "generation_error"
    assert result["metadata"]["count_requested"] == 3
    assert result["metadata"]["count_generated"] == 2

    events = await _collect(bus)
    # 失败题发了 quiz_question_error 事件
    err_events = [e for e in events if e["type"] == "quiz_question_error"]
    assert len(err_events) == 1 and err_events[0]["reason"] == "generation_error"
    await bus.close()


# ── M-9 + M-8 协同：残缺内容题被判无效跳过，不产出空题目 ──────────────────────

async def test_quiz_schema_invalid_question_skipped():
    """第 1 题 choice 答案不命中 options（M-9 校验失败）→ 跳过不发 quiz_question，第 2 题照常。"""
    ctx = _ctx()
    bus = StreamBus()
    plan_payload = json.dumps({"templates": [
        {"question_id": "q1", "topic": "T1", "question_type": "choice", "difficulty": "easy"},
        {"question_id": "q2", "topic": "T2", "question_type": "written", "difficulty": "easy"},
    ]})
    qid_outputs = {
        # q1：choice 题，但 correct_answer=D 不命中 options（A/B/C）→ 无效
        "q1": json.dumps({"question_type": "choice", "question": "1+1?",
                          "options": {"A": "1", "B": "2", "C": "3"}, "correct_answer": "D",
                          "explanation": "e"}),
        # q2：正常 written
        "q2": json.dumps({"question_type": "written", "question": "题2", "correct_answer": "答2", "explanation": "e2"}),
    }
    fake = _mock_loop_dispatch("探索摘要", plan_payload, qid_outputs)

    with (
        patch("core.question.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=ProfileRuntime())),
        patch("core.question.pipeline.build_common_context_layers", new=AsyncMock(return_value=CommonContextLayers())),
        patch("core.question.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
        patch("core.question.pipeline.load_prompt_dict", return_value=_FAKE_CFG),
        patch("core.question.pipeline.run_agent_loop", new=AsyncMock(side_effect=fake)),
    ):
        result = await QuizPipeline().run("出 2 道题", ctx, bus, count=2)

    # 修复前：q1 虽校验失败也会被原样 append（前端渲染出答案指向不存在选项的题）
    # 修复后：q1 被判无效跳过，只返回 q2
    assert len(result["questions"]) == 1
    assert result["questions"][0]["question_id"] == "q2"
    assert result["metadata"]["count_failed"] == 1
    assert result["metadata"]["errors"][0]["reason"] == "schema_invalid"

    events = await _collect(bus)
    # 无效题未发 quiz_question（不发空题目给前端）
    q_events = [e for e in events if e["type"] == "quiz_question"]
    assert len(q_events) == 1 and q_events[0]["index"] == 1  # 只有 q2（index=1）
    err_events = [e for e in events if e["type"] == "quiz_question_error"]
    assert len(err_events) == 1 and err_events[0]["reason"] == "schema_invalid"
    await bus.close()
