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
