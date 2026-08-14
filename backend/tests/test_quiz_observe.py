"""阶段F：判分观测接入 + rollup mastery 填充 + 校准指标（零 LLM）。"""
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def _seed(user_id="u1", course_id="c1"):
    from core.db.database import AsyncSessionLocal, CourseTopic

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id=course_id, topic_id="ohm", label="欧姆定律"))
        await s.commit()


# ── record_quiz_observation ─────────────────────────────────────────────────


async def test_quiz_observation_written_and_folds(db):
    """答错 → source=quiz 观测落库 + knowledge_mastery 折叠 + b_topic 上移。"""
    await _seed()
    from core.db.database import AsyncSessionLocal, CourseTopic, KnowledgeMastery, MasteryObservation
    from core.memory.quiz_observe import record_quiz_observation

    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            ok = await record_quiz_observation(s, "u1", "c1", "q1", "欧姆定律的公式", False)
        await s.commit()

    assert ok is True
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        obs = (await s.execute(select(MasteryObservation))).scalars().all()
        km = (await s.execute(select(KnowledgeMastery))).scalars().all()
        ct = (await s.execute(select(CourseTopic))).scalars().first()

    assert len(obs) == 1
    assert obs[0].source == "quiz" and obs[0].outcome == 0.0 and obs[0].question_id == "q1"
    assert km[0].n_obs == 1 and km[0].theta < 0  # 答错 → theta 下移
    assert ct.difficulty_b > 0.0 and ct.n_item == 1  # 题目侧：全班难度上移


async def test_quiz_observation_idempotent(db):
    """同 (user, question) 重复提交 → 不重复写观测。"""
    await _seed()
    from core.db.database import AsyncSessionLocal, MasteryObservation
    from sqlalchemy import select

    from core.memory.quiz_observe import record_quiz_observation

    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            first = await record_quiz_observation(s, "u1", "c1", "q1", "欧姆定律的公式", False)
            second = await record_quiz_observation(s, "u1", "c1", "q1", "欧姆定律的公式", True)
        await s.commit()

    assert first is True and second is False  # 重答改注不重复计
    async with AsyncSessionLocal() as s:
        n = len((await s.execute(select(MasteryObservation))).scalars().all())
    assert n == 1


async def test_quiz_observation_no_topic_no_write(db):
    """题面映射不到主题 → 不写（常态，非错误）。"""
    await _seed()
    from core.db.database import AsyncSessionLocal, MasteryObservation
    from core.memory.quiz_observe import record_quiz_observation
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[])):
            ok = await record_quiz_observation(s, "u1", "c1", "q1", "今天天气如何", False)
        await s.commit()

    assert ok is False
    async with AsyncSessionLocal() as s:
        assert len((await s.execute(select(MasteryObservation))).scalars().all()) == 0


# ── rollup mastery 填充 ─────────────────────────────────────────────────────


async def test_rollup_fills_mastery_avg_risk(db):
    """student_course_rollup 的 mastery_avg/risk 从折叠状态聚合填上（阶段F）。"""
    from core.db.database import (
        AsyncSessionLocal, CourseTopic, Enrollment, KnowledgeMastery,
    )

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id="c1", topic_id="ohm", label="欧姆定律"))
        s.add(Enrollment(student_id="u1", course_id="c1"))
        # 一个 theta=-0.9（poor 史）的主题 → risk 高
        s.add(KnowledgeMastery(
            user_id="u1", course_id="c1", kp_id="ohm", label="欧姆定律",
            theta=-0.9, n_obs=4, observation_count=4, risk=0.71, mastery=0.29,
        ))
        await s.commit()

    from core.analytics.learning_rollup import rollup_student_course

    n = await rollup_student_course(["c1"])
    assert n == 1
    from core.db.database import AsyncSessionLocal as ASL, StudentCourseRollup
    from sqlalchemy import select

    async with ASL() as s:
        row = (await s.execute(
            select(StudentCourseRollup).where(StudentCourseRollup.user_id == "u1")
        )).scalars().first()
    # risk = 1 - sigmoid(-0.9) ≈ 0.71（b=0）
    assert row.risk is not None and 0.65 < row.risk < 0.75
    assert row.mastery_avg is not None and 0.25 < row.mastery_avg < 0.35


# ── 校准指标纯函数 ──────────────────────────────────────────────────────────


def test_auc_perfect():
    from scripts.eval_memory.run_quiz_calibration import auc

    assert auc([(0.2, 0), (0.8, 1)]) == 1.0
    assert auc([(0.8, 0), (0.2, 1)]) == 0.0
    assert auc([(0.5, 0), (0.5, 1)]) == 0.5  # 并列


def test_brier():
    from scripts.eval_memory.run_quiz_calibration import brier

    assert brier([(1.0, 1), (0.0, 0)]) == 0.0  # 完美
    assert brier([(0.5, 1), (0.5, 0)]) == 0.25  # 无信息
    assert brier([(1.0, 0)]) == 1.0  # 最差


def test_calibration_skipped_without_quiz():
    """无 quiz 观测 → skipped=True 不产假数字。"""
    import asyncio

    from scripts.eval_memory.run_quiz_calibration import run

    out = asyncio.run(run())
    assert out["skipped"] is True
