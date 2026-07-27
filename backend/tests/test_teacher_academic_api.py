"""教师课表/成绩录入 API 回归测试（plan §2.5 / §4）。

覆盖：
- 课表 PUT 整表替换（幂等）、GET 读取
- 成绩 POST 批量 upsert（同 (student,course,item) 更新不新增）、GET、DELETE
- owner 校验：非 owner 教师 → 403；课程不存在 → 404（读写分离 + JWT 角色校验）
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy import update


async def _make_teacher(client) -> dict:
    """注册一名教师（升级 role=teacher）并返回 auth headers。"""
    from core.db.database import AsyncSessionLocal, User

    username = f"te_{os.urandom(3).hex()}"
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "testpass123", "display_name": "T"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    async with AsyncSessionLocal() as db:
        await db.execute(update(User).where(User.username == username).values(role="teacher"))
        await db.commit()
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_schedule_put_get_replace_idempotent(client, admin_headers, course_with_code):
    course_id = course_with_code["course_id"]

    # PUT 初始课表
    r = await client.put(
        f"/api/teacher/courses/{course_id}/schedule",
        headers=admin_headers,
        json={"items": [
            {"weekday": 1, "start_time": "08:00", "end_time": "09:40", "location": "A101", "teacher_name": "张老师"},
            {"weekday": 3, "start_time": "10:00", "end_time": "11:40", "location": "A102"},
        ]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2

    # GET 回读
    r = await client.get(f"/api/teacher/courses/{course_id}/schedule", headers=admin_headers)
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    weekdays = {row["weekday"] for row in rows}
    assert weekdays == {1, 3}

    # PUT 替换为 1 条 → 幂等整表替换（旧 2 条被删）
    r = await client.put(
        f"/api/teacher/courses/{course_id}/schedule",
        headers=admin_headers,
        json={"items": [{"weekday": 5, "start_time": "14:00", "end_time": "15:40", "location": "C201"}]},
    )
    assert r.status_code == 200
    assert r.json()["count"] == 1
    r = await client.get(f"/api/teacher/courses/{course_id}/schedule", headers=admin_headers)
    assert len(r.json()) == 1
    assert r.json()[0]["weekday"] == 5

    # PUT 空数组 → 清空（幂等）
    r = await client.put(
        f"/api/teacher/courses/{course_id}/schedule",
        headers=admin_headers,
        json={"items": []},
    )
    assert r.status_code == 200
    r = await client.get(f"/api/teacher/courses/{course_id}/schedule", headers=admin_headers)
    assert r.json() == []


@pytest.mark.asyncio
async def test_grades_upsert_get_delete(client, admin_headers, course_with_code):
    course_id = course_with_code["course_id"]
    student_id = f"s_{os.urandom(3).hex()}"

    # POST 批量 upsert
    r = await client.post(
        f"/api/teacher/courses/{course_id}/grades",
        headers=admin_headers,
        json={"grades": [
            {"student_id": student_id, "item_name": "期中", "score": 80, "full_score": 100, "comment": "粗心"},
            {"student_id": student_id, "item_name": "作业1", "score": 95, "full_score": 100},
        ]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["upserted"] == 2

    # GET
    r = await client.get(f"/api/teacher/courses/{course_id}/grades", headers=admin_headers)
    assert r.status_code == 200
    grades = r.json()
    assert len(grades) == 2

    # 再次 POST 同 (student,course,item)「期中」→ upsert 更新而非新增
    r = await client.post(
        f"/api/teacher/courses/{course_id}/grades",
        headers=admin_headers,
        json={"grades": [{"student_id": student_id, "item_name": "期中", "score": 85, "full_score": 100, "comment": "已复核"}]},
    )
    assert r.status_code == 200
    r = await client.get(f"/api/teacher/courses/{course_id}/grades", headers=admin_headers)
    grades = r.json()
    assert len(grades) == 2  # 仍 2 条（更新非新增）
    midterm = next(g for g in grades if g["item_name"] == "期中")
    assert midterm["score"] == 85
    assert midterm["comment"] == "已复核"

    # DELETE 单条
    grade_id = midterm["id"]
    r = await client.delete(f"/api/teacher/courses/{course_id}/grades/{grade_id}", headers=admin_headers)
    assert r.status_code == 200
    r = await client.get(f"/api/teacher/courses/{course_id}/grades", headers=admin_headers)
    assert len(r.json()) == 1

    # DELETE 不存在的 id → 404
    r = await client.delete(f"/api/teacher/courses/{course_id}/grades/nope", headers=admin_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_schedule_owner_forbidden(client, admin_headers, course_with_code):
    """非 owner 教师操作他人课程 → 403。"""
    course_id = course_with_code["course_id"]
    other_teacher = await _make_teacher(client)

    r = await client.put(
        f"/api/teacher/courses/{course_id}/schedule",
        headers=other_teacher,
        json={"items": [{"weekday": 1}]},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_grades_owner_forbidden(client, admin_headers, course_with_code):
    course_id = course_with_code["course_id"]
    other_teacher = await _make_teacher(client)

    r = await client.post(
        f"/api/teacher/courses/{course_id}/grades",
        headers=other_teacher,
        json={"grades": [{"student_id": "x", "item_name": "y"}]},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_schedule_nonexistent_course_404(client, admin_headers):
    r = await client.get(
        "/api/teacher/courses/no_such_course_xyz/schedule", headers=admin_headers
    )
    assert r.status_code == 404
