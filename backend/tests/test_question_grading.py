"""A3 判分闭环测试：answer_judge 校验层 + grading 分派 + fair 等级回流 + 防污染守卫 + 判分端点。

Mock 边界：沙箱 run_tests 传可控小代码（subprocess 真跑零依赖）；LLM 一律 mock
（answer_judge.judge_answer / client）；观测回流走 db fixture（in-memory SQLite）。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ── answer_judge._validate（纯函数） ─────────────────────────────────────────


_UA = "光合作用是植物利用光能把二氧化碳和水转化为有机物并释放氧气的过程"
_CA = "光合作用：光能→化学能，原料二氧化碳和水，产物有机物（葡萄糖）和氧气"


def test_validate_accepts_valid_grades():
    from core.question.answer_judge import _validate

    out = _validate({"grade": "good", "confidence": 0.9,
                     "evidence_quote": "转化为有机物并释放氧气", "feedback": "要点齐全"}, _UA, _CA)
    assert out is not None and out.grade == "good" and out.confidence == 0.9

    out = _validate({"grade": "fair", "confidence": 2.0,  # clamp 到 1.0
                     "evidence_quote": "原料二氧化碳和水", "feedback": "缺产物"}, _UA, _CA)
    assert out is not None and out.grade == "fair" and out.confidence == 1.0


def test_validate_unknown_abstains_and_bad_inputs_drop():
    from core.question.answer_judge import _validate

    assert _validate({"grade": "unknown"}, _UA, _CA) is None  # 弃权
    assert _validate({"grade": "excellent"}, _UA, _CA) is None  # 越闭集丢弃
    # 引用非「学生作答/参考答案」子串 → 整条作废（judge.py 同口径）
    assert _validate({"grade": "good", "evidence_quote": "这句话原文里没有"}, _UA, _CA) is None


# ── grading.grade_answer 分派矩阵 ────────────────────────────────────────────


def _coding_payload(answer_code: str, cases: list[str] | None = None) -> dict:
    return {
        "question_id": "q_1", "question_type": "coding",
        "question": "实现 add", "correct_answer": "```python\ndef add(a, b):\n    return a + b\n```",
        "user_answer": answer_code,
        "test_cases": cases if cases is not None else [
            "assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 0",
        ],
        "starter_code": "def add(a, b):", "language": "python",
    }


async def test_grade_coding_all_pass_good():
    from core.question.grading import grade_answer

    r = await grade_answer(_coding_payload("def add(a, b):\n    return a + b\n"))
    assert r.grade == "good" and r.source == "sandbox"
    assert r.score == 1.0 and r.is_correct is True
    assert r.test_results is not None and len(r.test_results) == 3
    assert "3/3" in r.feedback


async def test_grade_coding_ratio_thresholds():
    """0.75→good（≥0.7）；2/3=0.67→fair；0→poor。"""
    from core.question.grading import grade_answer

    r = await grade_answer(_coding_payload("def add(a, b):\n    return a + b + 1\n"))
    assert r.grade == "poor" and r.score == 0.0 and r.is_correct is False
    assert any(not t["passed"] for t in r.test_results)
    assert "未通过" in r.feedback


async def test_grade_coding_no_tests_ungraded():
    from core.question.grading import grade_answer

    r = await grade_answer(_coding_payload("def add(a, b):\n    return a + b\n", cases=[]))
    assert r.grade == "ungraded" and r.source == "" and r.is_correct is None
    assert "暂无自动判分" in r.feedback


async def test_grade_coding_extracts_fenced_answer():
    """学生作答带 ```python 围栏（含解释文字）→ 抽块判分。"""
    from core.question.grading import grade_answer

    payload = _coding_payload("我的答案：\n```python\ndef add(a, b):\n    return a + b\n```")
    r = await grade_answer(payload)
    assert r.grade == "good"


async def test_grade_short_answer_llm_and_fail_safe():
    from core.question import grading as g
    from core.question.answer_judge import AnswerJudgeResult

    payload = {"question_id": "q_2", "question_type": "short_answer",
               "question": "简述光合作用", "correct_answer": _CA, "user_answer": _UA}
    good = AnswerJudgeResult("good", 0.9, "释放氧气", "要点齐全")
    with patch.object(g, "judge_answer", AsyncMock(return_value=good)) as m:
        r = await g.grade_answer(payload, track={"user_id": "u1"})
    m.assert_awaited_once()
    assert r.grade == "good" and r.source == "llm_judge" and r.is_correct is True

    with patch.object(g, "judge_answer", AsyncMock(return_value=None)):
        r = await g.grade_answer(payload)
    assert r.grade == "ungraded" and "未能自动判分" in r.feedback


async def test_grade_empty_answer_short_circuits():
    """空作答不判分不耗资源（沙箱与 LLM 都不调）。"""
    from core.question import grading as g

    with patch.object(g, "run_tests") as m_sandbox, patch.object(g, "judge_answer") as m_llm:
        r = await g.grade_answer(_coding_payload("   "))
        await g.grade_answer({"question_type": "short_answer", "user_answer": "", "question": "", "correct_answer": ""})
    m_sandbox.assert_not_called()
    m_llm.assert_not_called()
    assert r.grade == "ungraded"


def test_grade_result_is_correct_semantics():
    from core.question.grading import GradeResult

    assert GradeResult(grade="good").is_correct is True
    assert GradeResult(grade="fair").is_correct is False   # 部分正确=未全对→进错题本
    assert GradeResult(grade="poor").is_correct is False
    assert GradeResult(grade="ungraded").is_correct is None


# ── record_quiz_observation：fair 等级 + feedback 证据（db） ──────────────────


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def test_quiz_observation_fair_level_outcome_half(db):
    """fair 等级 → outcome 0.5、weight 0.6（W_JUDGE）落库——部分正确通道。"""
    from core.db.database import AsyncSessionLocal, CourseTopic, MasteryObservation
    from core.memory.quiz_observe import record_quiz_observation
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id="c1", topic_id="ohm", label="欧姆定律"))
        await s.commit()
    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            ok = await record_quiz_observation(
                s, "u1", "c1", "q1", "欧姆定律的公式", "fair",
                user_answer="I=UR", correct_answer="I=U/R",
                weight=0.6, feedback="公式记反了",
            )
        await s.commit()
    assert ok is True
    async with AsyncSessionLocal() as s:
        obs = (await s.execute(select(MasteryObservation))).scalars().all()
    assert len(obs) == 1
    assert obs[0].outcome == 0.5
    assert obs[0].weight == 0.6
    assert obs[0].evidence_quote == "公式记反了"  # 判分 feedback 优先当证据


async def test_quiz_observation_bool_compat_maps_to_good_poor(db):
    """bool 兼容路径：False→poor（旧调用方/旧测试语义不变）。"""
    from core.db.database import AsyncSessionLocal, CourseTopic, MasteryObservation
    from core.memory.quiz_observe import record_quiz_observation
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id="c1", topic_id="ohm", label="欧姆定律"))
        await s.commit()
    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            ok = await record_quiz_observation(s, "u1", "c1", "q9", "欧姆定律的公式", False)
        await s.commit()
    assert ok is True
    async with AsyncSessionLocal() as s:
        obs = (await s.execute(select(MasteryObservation))).scalars().all()
    assert obs[0].outcome == 0.0 and obs[0].weight == 1.0


# ── 防污染守卫 + 判分端点（HTTP） ────────────────────────────────────────────


def _upsert_payload(qtype: str, **kw) -> dict:
    base = {
        "session_id": "sess_g", "question_id": "q_g", "question": "1+1=?",
        "question_type": qtype, "correct_answer": "2", "explanation": "...",
        "difficulty": "easy", "user_answer": "3", "is_correct": False,
    }
    base.update(kw)
    return base


async def test_guard_skips_observations_without_grade_source(client, auth_headers):
    """非本地可判题型 + 无 grade_source → 只写 entry，不回流学情事件/mastery 观测。"""
    with patch("core.analytics.events.record_learning_event", new=AsyncMock()) as m_ev, \
         patch("core.memory.quiz_observe.record_quiz_observation", new=AsyncMock()) as m_ob:
        r = await client.post(
            "/api/question/notebook/entries/upsert",
            json=_upsert_payload("written"), headers=auth_headers,
        )
    assert r.status_code == 200, r.text
    assert r.json()["grade"] == ""  # 新字段贯通（迁移后列存在）
    m_ev.assert_not_called()
    m_ob.assert_not_called()


async def test_guard_allows_with_grade_source(client, auth_headers):
    """带 grade_source=sandbox + grade=fair → 观测回流且 level/weight 正确。"""
    with patch("core.analytics.events.record_learning_event", new=AsyncMock()), \
         patch("core.memory.quiz_observe.record_quiz_observation",
               new=AsyncMock(return_value=True)) as m_ob:
        r = await client.post(
            "/api/question/notebook/entries/upsert",
            json=_upsert_payload("coding", grade="fair", grade_source="sandbox",
                                 score=0.67, feedback="通过 2/3 条用例"),
            headers=auth_headers,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grade"] == "fair" and body["grade_source"] == "sandbox"
    assert body["score"] == 0.67 and "2/3" in body["feedback"]
    m_ob.assert_awaited_once()
    # 位置参数：args = (db, uid, course_id, question_id, question, level)
    assert m_ob.await_args.args[5] == "fair"


async def test_guard_choice_compat_still_flows(client, auth_headers):
    """choice 本地判（options 非空）→ 观测照旧回流（旧行为回归）。"""
    with patch("core.analytics.events.record_learning_event", new=AsyncMock()), \
         patch("core.memory.quiz_observe.record_quiz_observation",
               new=AsyncMock(return_value=True)) as m_ob:
        r = await client.post(
            "/api/question/notebook/entries/upsert",
            json=_upsert_payload("choice", options={"A": "1", "B": "2"}, is_correct=True),
            headers=auth_headers,
        )
    assert r.status_code == 200, r.text
    m_ob.assert_awaited_once()
    assert m_ob.await_args.args[5] == "good"  # is_correct=True 兼容映射


async def test_guard_empty_type_legacy_flows(client, auth_headers):
    """空 question_type（legacy 最小载荷）→ 照旧回流：answered 学情信号不能悄悄
    依赖客户端必传题型字段（test_learning_events 的 P0-a 回归口径）。"""
    with patch("core.analytics.events.record_learning_event", new=AsyncMock()) as m_ev:
        r = await client.post(
            "/api/question/notebook/entries/upsert",
            json=_upsert_payload(""), headers=auth_headers,
        )
    assert r.status_code == 200, r.text
    m_ev.assert_awaited_once()


async def test_grade_endpoint_coding_sandbox(client, auth_headers):
    """判分端点：coding → 沙箱路径真跑（mock run_tests 桩控制结果形状）。"""
    from core.question.sandbox import TestRunResult

    payload = dict(_coding_payload("def add(a, b):\n    return a + b\n"),
                   session_id="sess_g", language="python")
    fake = TestRunResult(passed=[0, 1], failed=[(2, "exit=1: AssertionError")], total=3)
    with patch("core.question.grading.run_tests", return_value=fake) as m:
        r = await client.post(
            "/api/question/notebook/grade", json=payload, headers=auth_headers,
        )
    m.assert_called_once()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grade"] == "fair" and body["source"] == "sandbox"
    assert body["score"] == pytest.approx(2 / 3, abs=1e-3)  # 后端 round 4 位
    assert body["is_correct"] is False
    assert len(body["test_results"]) == 3


async def test_grade_endpoint_empty_answer_ungraded(client, auth_headers):
    payload = dict(_coding_payload(""), session_id="sess_g")
    with patch("core.question.grading.run_tests") as m:
        r = await client.post(
            "/api/question/notebook/grade", json=payload, headers=auth_headers,
        )
    m.assert_not_called()
    assert r.status_code == 200
    body = r.json()
    assert body["grade"] == "ungraded" and body["is_correct"] is None


async def test_grade_endpoint_short_answer_llm(client, auth_headers):
    from core.question.answer_judge import AnswerJudgeResult

    payload = {"session_id": "s", "question_id": "q_2", "question_type": "short_answer",
               "question": "简述光合作用", "correct_answer": _CA, "user_answer": _UA}
    with patch("core.question.grading.judge_answer",
               AsyncMock(return_value=AnswerJudgeResult("fair", 0.7, "释放氧气", "缺原料"))):
        r = await client.post(
            "/api/question/notebook/grade", json=payload, headers=auth_headers,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grade"] == "fair" and body["source"] == "llm_judge"
    assert body["feedback"] == "缺原料"
