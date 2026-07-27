"""3 个只读学业工具回归测试（plan §2 / §4）。

核心安全断言（OWASP LLM06 / IDOR 防线）：
- query_grades：A 学生查成绩，返回里不含 B 学生数据（强制 WHERE student_id==注入值）。
- query_timetable：未选课的课程不出现在课表（JOIN Enrollment 限定「我选的课」）。
- query_mistakes：只取本人答错的题（is_correct=False + user_id 隔离）。
- 身份只走注入的 user_id：空 user_id 直接拒；schema 绝不含身份参数。

身份注入机制（fail-closed）：若模型幻觉出 user_id 参数，tool_dispatch.py 的 **call_kwargs
会与注入的 user_id 撞成 TypeError → registry 兜底「工具执行失败」。schema 不暴露身份参数
是预防手段，本测试固化该契约。
"""
from __future__ import annotations

import pytest


# ── 身份注入契约 ──────────────────────────────────────────────────────────────

def test_academic_schemas_have_no_identity_params():
    """schema 绝不暴露 user_id/student_id 等身份参数（身份只走注入）。"""
    from core.academic.tools import (
        QUERY_GRADES_SCHEMA,
        QUERY_MISTAKES_SCHEMA,
        QUERY_TIMETABLE_SCHEMA,
    )

    forbidden = {"user_id", "student_id", "uid", "owner_id", "studentId"}
    for schema in (QUERY_TIMETABLE_SCHEMA, QUERY_GRADES_SCHEMA, QUERY_MISTAKES_SCHEMA):
        props = set(schema["function"]["parameters"].get("properties", {}).keys())
        leaked = props & forbidden
        assert not leaked, f"{schema['function']['name']} schema 泄漏身份参数: {leaked}"


def test_academic_tools_registered():
    """3 个工具已注册进 ToolRegistry（名字 + schema 名一致）。"""
    from core.agent.registry import get_tool_registry

    reg = get_tool_registry()
    for name in ("query_timetable", "query_grades", "query_mistakes"):
        assert reg.has(name), f"工具 {name} 未注册"
        entry = reg.get(name)
        assert entry.schema["function"]["name"] == name


@pytest.mark.asyncio
async def test_academic_tools_reject_empty_user_id():
    """空 user_id 直接拒（身份只走注入，无身份即不可用）。"""
    from core.academic.tools import (
        execute_query_grades,
        execute_query_mistakes,
        execute_query_timetable,
    )

    for fn in (execute_query_grades, execute_query_timetable, execute_query_mistakes):
        res = await fn(user_id="", course_id="c1")
        assert res.success is False, f"{fn.__name__} 空 user_id 应拒绝"


# ── 数据隔离（DB 集成）────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_grades_isolation_between_students(client):
    """A 查成绩：含 A 的分数，不含 B 的分数（IDOR 防线）。"""
    from core.academic.tools import execute_query_grades
    from core.db.database import AsyncSessionLocal, Grade, User

    async with AsyncSessionLocal() as db:
        a = User(username="gA", password_hash="x", display_name="A")
        b = User(username="gB", password_hash="x", display_name="B")
        db.add_all([a, b])
        await db.flush()
        db.add(Grade(student_id=a.id, course_id="c1", item_name="期中", score=80, full_score=100))
        db.add(Grade(student_id=b.id, course_id="c1", item_name="期中", score=55, full_score=100))
        await db.commit()
        a_id = a.id

    res = await execute_query_grades(course_id="c1", user_id=a_id)
    assert res.success is True
    assert "80" in res.content
    assert "55" not in res.content  # B 的分数不泄漏给 A


@pytest.mark.asyncio
async def test_query_grades_item_keyword_filter(client):
    from core.academic.tools import execute_query_grades
    from core.db.database import AsyncSessionLocal, Grade, User

    async with AsyncSessionLocal() as db:
        a = User(username="gK", password_hash="x")
        db.add(a)
        await db.flush()
        db.add(Grade(student_id=a.id, course_id="c1", item_name="期中考试", score=80, full_score=100))
        db.add(Grade(student_id=a.id, course_id="c1", item_name="作业1", score=90, full_score=100))
        await db.commit()
        a_id = a.id

    res = await execute_query_grades(course_id="c1", user_id=a_id, item_keyword="期中")
    assert res.success is True
    assert "期中" in res.content
    assert "作业1" not in res.content


@pytest.mark.asyncio
async def test_query_timetable_only_enrolled_courses(client):
    """课表只含已选课程（JOIN Enrollment）；未选课的课程不出现。"""
    from core.academic.tools import execute_query_timetable
    from core.db.database import AsyncSessionLocal, CourseSchedule, Enrollment, User

    async with AsyncSessionLocal() as db:
        a = User(username="tT", password_hash="x")
        db.add(a)
        await db.flush()
        db.add(CourseSchedule(course_id="c1", weekday=1, start_time="08:00", end_time="09:40", location="A101"))
        db.add(CourseSchedule(course_id="c2", weekday=2, start_time="10:00", end_time="11:40", location="B202"))
        db.add(Enrollment(student_id=a.id, course_id="c1"))  # 只选 c1
        await db.commit()
        a_id = a.id

    res = await execute_query_timetable(user_id=a_id)
    assert res.success is True
    assert "08:00" in res.content  # c1 出现
    assert "c1" in res.content
    assert "c2" not in res.content  # 未选课不出现
    assert "B202" not in res.content


@pytest.mark.asyncio
async def test_query_timetable_weekday_filter(client):
    from core.academic.tools import execute_query_timetable
    from core.db.database import AsyncSessionLocal, CourseSchedule, Enrollment, User

    async with AsyncSessionLocal() as db:
        a = User(username="tW", password_hash="x")
        db.add(a)
        await db.flush()
        db.add(Enrollment(student_id=a.id, course_id="c1"))
        db.add(CourseSchedule(course_id="c1", weekday=1, start_time="08:00", end_time="09:40", location="A101"))
        await db.commit()
        a_id = a.id

    res_mon = await execute_query_timetable(user_id=a_id, weekday=1)
    assert res_mon.success is True
    assert "周一" in res_mon.content

    # 周五无课 → 空结果（success=False 明确无命中）
    res_fri = await execute_query_timetable(user_id=a_id, weekday=5)
    assert res_fri.success is False


@pytest.mark.asyncio
async def test_query_mistakes_isolation_and_only_wrong(client):
    """错题本：只含本人答错的题（不含别人的、不含自己做对的）。"""
    from core.academic.tools import execute_query_mistakes
    from core.db.database import AsyncSessionLocal, NotebookEntry, User

    async with AsyncSessionLocal() as db:
        a = User(username="mA", password_hash="x")
        b = User(username="mB", password_hash="x")
        db.add_all([a, b])
        await db.flush()
        # question_id 必填唯一键（user_id,session_id,question_id），故各给不同 question_id
        db.add(NotebookEntry(user_id=a.id, question_id="q1", question="A的错题", is_correct=False, user_answer="x", correct_answer="y"))
        db.add(NotebookEntry(user_id=b.id, question_id="q1", question="B的错题", is_correct=False, user_answer="x", correct_answer="y"))
        db.add(NotebookEntry(user_id=a.id, question_id="q2", question="A做对的题", is_correct=True, user_answer="y", correct_answer="y"))
        await db.commit()
        a_id = a.id

    res = await execute_query_mistakes(user_id=a_id)
    assert res.success is True
    assert "A的错题" in res.content
    assert "B的错题" not in res.content  # 他人数据隔离
    assert "A做对的题" not in res.content  # 只取错题


@pytest.mark.asyncio
async def test_query_empty_data_returns_failure_not_exception(client):
    """无数据时返回明确无命中（success=False），不抛异常。"""
    from core.academic.tools import (
        execute_query_grades,
        execute_query_mistakes,
        execute_query_timetable,
    )
    from core.db.database import AsyncSessionLocal, User

    async with AsyncSessionLocal() as db:
        a = User(username="eE", password_hash="x")
        db.add(a)
        await db.commit()
        a_id = a.id

    assert (await execute_query_grades(user_id=a_id, course_id="c1")).success is False
    assert (await execute_query_mistakes(user_id=a_id)).success is False
    assert (await execute_query_timetable(user_id=a_id)).success is False
