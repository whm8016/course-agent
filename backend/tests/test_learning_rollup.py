"""学情读模型聚合测试（学情分析四模块设计 §第二期 p2-rollup）。

验证 rollup_course_daily / rollup_student_course 的删后重算（幂等）+ 聚合正确性 +
课程过滤（quiz 不跨课程，同 P1-a 口径）。
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from core.analytics.learning_rollup import (
    _day_range,
    rollup_course_daily,
    rollup_student_course,
)


@pytest.mark.asyncio
async def test_rollup_course_daily_idempotent(client):
    from core.db.database import AsyncSessionLocal, CourseDailyRollup, LearningEvent, User

    day = "20260101"
    start, _end = _day_range(day)
    async with AsyncSessionLocal() as db:
        u1 = User(username="rd_u1", password_hash="x")
        u2 = User(username="rd_u2", password_hash="x")
        db.add_all([u1, u2])
        await db.flush()
        # u1: 2 asked + 1 answered；u2: 1 asked → 共 3 asked、1 answered、2 去重学生
        for uid, verb, n in [(u1.id, "asked", 2), (u2.id, "asked", 1), (u1.id, "answered", 1)]:
            for _ in range(n):
                db.add(LearningEvent(
                    actor_user_id=uid, course_id="c_rd", verb=verb, created_at=start + 100,
                ))
        await db.commit()

    assert await rollup_course_daily([day]) == 1
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            select(CourseDailyRollup).where(CourseDailyRollup.day == day)
        )).scalar_one()
    assert row.course_id == "c_rd"
    assert row.active_students == 2
    assert row.questions == 3
    assert row.answers == 1

    # 幂等：再跑一次，行数/数值不变（删后重插，不重复不漂移）
    assert await rollup_course_daily([day]) == 1
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(CourseDailyRollup).where(CourseDailyRollup.day == day)
        )).scalars().all()
    assert len(rows) == 1
    assert rows[0].questions == 3


@pytest.mark.asyncio
async def test_rollup_student_course_scoped_and_idempotent(client):
    from core.db.database import (
        AsyncSessionLocal, Enrollment, Message, NotebookEntry, Session, StudentCourseRollup, User,
    )

    async with AsyncSessionLocal() as db:
        stu = User(username="rs_u1", password_hash="x")
        db.add(stu)
        await db.flush()
        sid = stu.id
        db.add(Enrollment(student_id=sid, course_id="c_rs"))
        s_in1 = Session(id="rs_s1", course_id="c_rs", user_id=sid)
        s_in2 = Session(id="rs_s2", course_id="c_rs", user_id=sid)
        s_out = Session(id="rs_s3", course_id="c_other", user_id=sid)
        db.add_all([s_in1, s_in2, s_out])
        await db.flush()
        db.add(Message(session_id=s_in1.id, role="user", content="hi"))
        db.add(Message(session_id=s_in2.id, role="user", content="yo"))
        db.add(Message(session_id=s_in1.id, role="user", content="again"))  # 本课程共 3 条 user 消息
        # 本课程 quiz：1 对 1 错；他课程 quiz：1 对（须被排除）
        db.add(NotebookEntry(user_id=sid, course_id="c_rs", session_id=s_in1.id, question_id="q1", is_correct=True))
        db.add(NotebookEntry(user_id=sid, course_id="c_rs", session_id=s_in2.id, question_id="q2", is_correct=False))
        db.add(NotebookEntry(user_id=sid, course_id="c_other", session_id=s_out.id, question_id="q3", is_correct=True))
        await db.commit()

    assert await rollup_student_course(["c_rs"]) == 1
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            select(StudentCourseRollup).where(StudentCourseRollup.course_id == "c_rs")
        )).scalar_one()
    assert row.user_id == sid
    assert row.sessions == 2          # 只本课程
    assert row.messages == 3          # 本课程 user 消息
    assert row.quiz_total == 2        # 排除他课程
    assert row.quiz_correct == 1
    assert row.last_active_at is not None
    assert row.mastery_avg is None    # Phase 4 BKT 占位

    # 幂等
    assert await rollup_student_course(["c_rs"]) == 1
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(StudentCourseRollup).where(StudentCourseRollup.course_id == "c_rs")
        )).scalars().all()
    assert len(rows) == 1


def test_recent_days_window():
    """recent_days(n) 返回 n 个连续 UTC 日（今日在前、无重复）。

    P0-b：cron 滚动窗口由 2 天（today_yesterday）扩到 7 天，覆盖展示层趋势 + 首次回填历史。
    """
    from datetime import datetime, timezone

    from core.analytics.learning_rollup import recent_days

    days = recent_days(7)
    assert len(days) == 7
    assert days[0] == datetime.now(timezone.utc).strftime("%Y%m%d")  # 今日在前
    parsed = [datetime.strptime(d, "%Y%m%d").date() for d in days]
    assert parsed == sorted(parsed, reverse=True)  # 连续递减
    assert len(set(days)) == 7                       # 无重复
