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
    """答错 → source=quiz 观测落库 + knowledge_mastery 折叠；b 由 cron 重折（不在写路径动）。"""
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
    # 答错不强化：strength 保持默认 5.0
    assert km[0].strength == 5.0
    # 难度不在写路径更新（plan「班级难度估计修复」）：写完观测 b 仍为 0，等 cron 日频重折
    assert ct.difficulty_b == 0.0 and ct.n_item == 0


async def test_quiz_correct_reinforces_strength(db):
    """答对 → strength +1（MemoryBank spacing effect，论文式(4) S 递增）。

    回归根因：旧实现唯一写 strength += 1 的 append_mastery_from_topics 无生产调用方，
    巩固主路径不碰 strength → p_forget 只随时间变，答对不减缓遗忘。
    """
    await _seed()
    from core.db.database import AsyncSessionLocal, KnowledgeMastery
    from core.memory.quiz_observe import record_quiz_observation
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            ok1 = await record_quiz_observation(s, "u1", "c1", "q1", "欧姆定律的公式", True)
            ok2 = await record_quiz_observation(s, "u1", "c1", "q2", "欧姆定律单位", True)
        await s.commit()

    assert ok1 and ok2
    async with AsyncSessionLocal() as s:
        km = (await s.execute(select(KnowledgeMastery))).scalars().all()

    # 首答对建行 strength=6（初值5+1），第二次答对 +1 → 7
    assert km[0].strength == 7.0


async def test_strength_capped_at_15(db):
    """多轮 good 强化封顶 _STRENGTH_CAP=15，防 p_forget 恒 0。"""
    from core.db.database import AsyncSessionLocal, CourseTopic, KnowledgeMastery
    from core.memory.consolidation import _STRENGTH_CAP, _write_mastery_observations
    from core.memory.judge import Judgement
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id="c1", topic_id="ohm", label="欧姆定律"))
        s.add(KnowledgeMastery(
            user_id="u1", course_id="c1", kp_id="ohm", label="欧姆定律",
            mastery=0.8, risk=0.2, observation_count=10, strength=15.0,
            theta=1.0, n_obs=10,
        ))
        await s.commit()

        # 已封顶的行再来一条 good 观测 → strength 仍 15（min 封顶）
        await _write_mastery_observations(
            s, "u1", "c1", [Judgement("ohm", "good", 1.0, "引用")], []
        )

    async with AsyncSessionLocal() as s:
        km = (await s.execute(select(KnowledgeMastery))).scalars().one()
    assert km.strength == _STRENGTH_CAP


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


async def test_quiz_second_answer_no_double_count(db):
    """难度重折的幂等回归（迁到 cron 口径）：两次跑 fold_class_difficulty 结果一致，
    且同一份全班观测每条只计一次。

    旧在线路径（写观测时全班 quiz 重折）已删——难度由 cron_refit_difficulty 日频
    全班重折，全量重放天然幂等。本测试直接验证折叠函数的幂等性 + 非零 b 上移。
    """
    from core.memory.mastery_state import fold_class_difficulty

    await _seed()
    from core.db.database import AsyncSessionLocal, MasteryObservation
    from core.memory.quiz_observe import record_quiz_observation
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            await record_quiz_observation(s, "u1", "c1", "q1", "欧姆定律的公式", False)
        await s.commit()
    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            await record_quiz_observation(s, "u1", "c1", "q2", "欧姆定律单位换算", True)
        await s.commit()

    async with AsyncSessionLocal() as s:
        obs = (await s.execute(select(MasteryObservation))).scalars().all()
        rows = [
            {"user_id": o.user_id, "created_at": o.created_at,
             "outcome": o.outcome, "weight": o.weight}
            for o in obs
        ]

    b1, n1 = fold_class_difficulty(rows)
    b2, n2 = fold_class_difficulty(rows)
    assert n1 == n2 == 2  # 两条观测各计一次（无起点叠加）
    assert abs(b1 - b2) < 1e-12  # 幂等：同一份日志折两次结果一致
    assert b1 < 0.0  # 一错一对 → 题偏难一点点（theta 被第一条 poor 拉低后第二条 good 的
    # err 贡献变小，净效应 b 略负）；关键锚是非零且可重放，不是具体符号


async def test_cron_refit_difficulty_writes_b(db):
    """cron_refit_difficulty：对近期有观测的 (course, topic) 全班重折写回
    difficulty_b / n_item；写路径本身不动 b。"""
    await _seed()
    from core.db.database import AsyncSessionLocal, CourseTopic, MasteryObservation
    from core.memory.quiz_observe import record_quiz_observation
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            await record_quiz_observation(s, "u1", "c1", "q1", "欧姆定律的公式", False)
        await s.commit()
    async with AsyncSessionLocal() as s:
        ct_before = (await s.execute(select(CourseTopic))).scalars().first()
        assert ct_before.difficulty_b == 0.0  # 写路径不动 b

    import worker as worker_mod

    await worker_mod.cron_refit_difficulty({})

    async with AsyncSessionLocal() as s:
        ct_after = (await s.execute(select(CourseTopic))).scalars().first()
        n_obs = len((await s.execute(select(MasteryObservation))).scalars().all())
    assert ct_after.n_item == n_obs == 1
    assert ct_after.difficulty_b > 0.0  # 答错 → cron 重折后全班难度上移


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


# ── 错因闭环：evidence_quote 存具体错因（复用现有字段，不建表）───────────────


async def test_quiz_wrong_answer_stores_error_evidence(db):
    """答错且带作答 → evidence_quote 是「学生答 vs 正确答案」，不再是纯题面。"""
    await _seed()
    from core.db.database import AsyncSessionLocal, MasteryObservation
    from core.memory.quiz_observe import record_quiz_observation
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            ok = await record_quiz_observation(
                s, "u1", "c1", "q9", "欧姆定律的公式", False,
                user_answer="I=UR", correct_answer="I=U/R",
            )
        await s.commit()

    assert ok is True
    async with AsyncSessionLocal() as s:
        obs = (await s.execute(select(MasteryObservation))).scalars().all()
    assert len(obs) == 1
    assert "I=UR" in obs[0].evidence_quote  # 学生的错误作答摘录
    assert "I=U/R" in obs[0].evidence_quote
    assert obs[0].evidence_quote.startswith("学生答：")


async def test_quiz_evidence_fallback_without_answer_text(db):
    """答对 / 未传作答文本 → 维持题面截断回退（现有行为不变）。"""
    await _seed()
    from core.db.database import AsyncSessionLocal, MasteryObservation
    from core.memory.quiz_observe import record_quiz_observation
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        with patch("core.memory.topic_match.match_topics",
                   AsyncMock(return_value=[("ohm", 0.88)])):
            ok_wrong_no_text = await record_quiz_observation(
                s, "u1", "c1", "q10", "欧姆定律的公式", False,
            )
            ok_right = await record_quiz_observation(
                s, "u1", "c1", "q11", "欧姆定律单位", True,
                user_answer="欧姆", correct_answer="欧姆",
            )
        await s.commit()

    assert ok_wrong_no_text and ok_right
    async with AsyncSessionLocal() as s:
        obs = (await s.execute(select(MasteryObservation))).scalars().all()
    by_q = {o.question_id: o for o in obs}
    assert by_q["q10"].evidence_quote == "欧姆定律的公式"  # 答错但无文本 → 题面回退
    assert by_q["q11"].evidence_quote == "欧姆定律单位"  # 答对 → 题面回退


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
