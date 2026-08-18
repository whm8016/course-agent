"""教师学情统计 API 回归测试。

覆盖 student-stats 的 quiz 聚合口径：
- NotebookEntry 必须按课程过滤（via Session.course_id），不能仅按 user_id 全局聚合，
  否则学生在他课程的答题会把别的课程的答题也算进来，导致 list 与 detail 两个页面
  正确率对不上（详见 学情分析四模块设计 §模块二）。
"""
from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_student_stats_quiz_scoped_to_course(client, admin_headers, course_with_code):
    """student-stats 的 quiz 统计只计本课程答题，不含学生在他课程的答题。

    场景：一名学生在「本课程」答 1 对 1 错，在「他课程」又答对 1 题。
    旧实现按 ``NotebookEntry.user_id`` 全局聚合，会把 3 题全算进本课程
    （total=3, correct=2, acc≈0.667）；正确口径应只计本课程 2 题
    （total=2, correct=1, acc=0.5），与 student/{id}/detail 一致。
    """
    from core.db.database import AsyncSessionLocal, Enrollment, NotebookEntry, Session, User

    course_id = course_with_code["course_id"]

    async with AsyncSessionLocal() as db:
        stu = User(username=f"qz_{os.urandom(3).hex()}", password_hash="x")
        db.add(stu)
        await db.flush()
        stu_id = stu.id

        db.add(Enrollment(student_id=stu_id, course_id=course_id))
        # 本课程会话 + 他课程会话（同一名学生）
        s_in = Session(id=f"in_{os.urandom(2).hex()}", course_id=course_id, user_id=stu_id)
        s_out = Session(
            id=f"out_{os.urandom(2).hex()}",
            course_id=f"other_{os.urandom(2).hex()}",
            user_id=stu_id,
        )
        db.add_all([s_in, s_out])
        await db.flush()

        # 本课程：1 对 1 错
        db.add(NotebookEntry(
            user_id=stu_id, course_id=course_id, session_id=s_in.id, question_id="in_q1",
            question="本课对", is_correct=True, user_answer="y", correct_answer="y",
        ))
        db.add(NotebookEntry(
            user_id=stu_id, course_id=course_id, session_id=s_in.id, question_id="in_q2",
            question="本课错", is_correct=False, user_answer="x", correct_answer="y",
        ))
        # 他课程：又答对 1 题（旧 bug 会把它也算进本课程）
        db.add(NotebookEntry(
            user_id=stu_id, course_id=s_out.course_id, session_id=s_out.id, question_id="out_q1",
            question="他课对", is_correct=True, user_answer="y", correct_answer="y",
        ))
        await db.commit()

    r = await client.get(
        f"/api/teacher/courses/{course_id}/analytics/student-stats",
        headers=admin_headers,
    )
    assert r.status_code == 200, r.text
    summaries = r.json()["student_summaries"]
    me = next(s for s in summaries if s["student_id"] == stu_id)
    # 只有本课程的 2 题（1 对），他课程的答题不计入
    assert me["total_questions"] == 2, me
    assert me["correct_count"] == 1, me
    assert me["accuracy_rate"] == 0.5, me


@pytest.mark.asyncio
async def test_overview_message_count_scoped_by_course_id(client, admin_headers, course_with_code):
    """P1-step2：analytics/overview 按 Message.course_id 过滤（免 JOIN Session），
    只计本课程消息，他课程消息不计入。"""
    from core.db.database import AsyncSessionLocal, Enrollment, Message, Session, User

    course_id = course_with_code["course_id"]
    async with AsyncSessionLocal() as db:
        stu = User(username=f"ov_{os.urandom(3).hex()}", password_hash="x")
        db.add(stu)
        await db.flush()
        db.add(Enrollment(student_id=stu.id, course_id=course_id))
        s_in = Session(id=f"in_{os.urandom(2).hex()}", course_id=course_id, user_id=stu.id)
        s_out = Session(
            id=f"out_{os.urandom(2).hex()}",
            course_id=f"other_{os.urandom(2).hex()}",
            user_id=stu.id,
        )
        db.add_all([s_in, s_out])
        await db.flush()
        # 本课程 2 条 user 消息；他课程 1 条（不应计入 total_messages）
        db.add(Message(session_id=s_in.id, course_id=course_id, role="user", content="a"))
        db.add(Message(session_id=s_in.id, course_id=course_id, role="user", content="b"))
        db.add(Message(session_id=s_out.id, course_id=s_out.course_id, role="user", content="c"))
        await db.commit()

    r = await client.get(
        f"/api/teacher/courses/{course_id}/analytics/overview", headers=admin_headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["total_messages"] == 2  # 只计本课程


# ── student_course_rollup 读路径接通（P0 收口）──────────────────────────────

async def _seed_student_with_activity(course_id: str) -> tuple[str, float]:
    """种一个学生：1 会话 + 2 条 user 消息 + 0 答题。返回 (user_id, last_active近似值)。"""
    from core.db.database import AsyncSessionLocal, Enrollment, Message, Session, User

    async with AsyncSessionLocal() as db:
        stu = User(username=f"ru_{os.urandom(3).hex()}", password_hash="x")
        db.add(stu)
        await db.flush()
        db.add(Enrollment(student_id=stu.id, course_id=course_id))
        s = Session(id=f"rs_{os.urandom(2).hex()}", course_id=course_id, user_id=stu.id)
        db.add(s)
        await db.flush()
        db.add(Message(session_id=s.id, course_id=course_id, role="user", content="a"))
        db.add(Message(session_id=s.id, course_id=course_id, role="user", content="b"))
        await db.commit()
        return stu.id, s.updated_at


async def _insert_rollup(course_id: str, user_id: str, *, updated_at: float, **cols):
    from core.db.database import AsyncSessionLocal, StudentCourseRollup

    async with AsyncSessionLocal() as db:
        db.add(StudentCourseRollup(
            user_id=user_id, course_id=course_id, updated_at=updated_at, **cols
        ))
        await db.commit()


@pytest.mark.asyncio
async def test_student_stats_reads_fresh_rollup(client, admin_headers, course_with_code):
    """新鲜 rollup（<6h）→ sessions/messages/last_active/quiz 四列直接读 rollup。

    rollup 里故意写与现算不同的数字：响应反映 rollup 数字即证明走了读路径
    （而非对每生现算 4 条聚合）。risk_score 仍走图谱、7 天趋势仍现算（不在
    断言范围，rollup 无此数据）。
    """
    import time as _time

    course_id = course_with_code["course_id"]
    stu_id, _ = await _seed_student_with_activity(course_id)
    await _insert_rollup(
        course_id, stu_id, updated_at=_time.time(),
        sessions=7, messages=42, quiz_total=9, quiz_correct=8, last_active_at=111.0,
    )

    r = await client.get(
        f"/api/teacher/courses/{course_id}/analytics/student-stats",
        headers=admin_headers,
    )
    assert r.status_code == 200, r.text
    me = next(s for s in r.json()["student_summaries"] if s["student_id"] == stu_id)
    assert me["total_sessions"] == 7, me
    assert me["total_messages"] == 42, me
    assert me["total_questions"] == 9, me
    assert me["correct_count"] == 8, me
    assert me["last_active_at"] == 111.0, me


@pytest.mark.asyncio
async def test_student_stats_falls_back_when_rollup_stale(client, admin_headers, course_with_code):
    """rollup 过期（updated_at 7h 前）→ 降级现算，rollup 里的失真数字被忽略。"""
    import time as _time

    course_id = course_with_code["course_id"]
    stu_id, live_last = await _seed_student_with_activity(course_id)
    await _insert_rollup(
        course_id, stu_id, updated_at=_time.time() - 7 * 3600,
        sessions=99, messages=99, quiz_total=99, quiz_correct=99, last_active_at=111.0,
    )

    r = await client.get(
        f"/api/teacher/courses/{course_id}/analytics/student-stats",
        headers=admin_headers,
    )
    assert r.status_code == 200, r.text
    me = next(s for s in r.json()["student_summaries"] if s["student_id"] == stu_id)
    # 现算口径：1 会话 / 2 条 user 消息 / 0 答题；99 系失真数字未生效
    assert me["total_sessions"] == 1, me
    assert me["total_messages"] == 2, me
    assert me["total_questions"] == 0, me
    assert me["last_active_at"] != 111.0, me


@pytest.mark.asyncio
async def test_student_stats_live_for_new_enrollee_with_fresh_rollup(client, admin_headers, course_with_code):
    """rollup 新鲜但 cron 后新入课的学生无行 → 该生走现算（不全班降级）。"""
    import time as _time

    course_id = course_with_code["course_id"]
    # 旧生：有 rollup 行（新鲜），读 rollup
    old_id, _ = await _seed_student_with_activity(course_id)
    await _insert_rollup(
        course_id, old_id, updated_at=_time.time(),
        sessions=7, messages=42, quiz_total=9, quiz_correct=8, last_active_at=111.0,
    )
    # 新生：cron 后入课，无 rollup 行 → 现算（1 会话 / 2 消息）
    new_id, _ = await _seed_student_with_activity(course_id)

    r = await client.get(
        f"/api/teacher/courses/{course_id}/analytics/student-stats",
        headers=admin_headers,
    )
    assert r.status_code == 200, r.text
    summaries = {s["student_id"]: s for s in r.json()["student_summaries"]}
    assert summaries[old_id]["total_sessions"] == 7
    assert summaries[old_id]["total_messages"] == 42
    assert summaries[new_id]["total_sessions"] == 1, summaries[new_id]
    assert summaries[new_id]["total_messages"] == 2, summaries[new_id]
