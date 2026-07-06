"""Courses API: list + join happy-path tests."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_courses_unauthenticated(client: AsyncClient):
    """未登录访问课程列表 → 401。"""
    r = await client.get("/api/courses")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_list_courses_empty_for_new_student(client: AsyncClient, auth_headers: dict):
    """新注册学生未加入任何课程，列表应为空。"""
    r = await client.get("/api/courses", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["courses"] == []


@pytest.mark.asyncio
async def test_join_course_success(
    client: AsyncClient, auth_headers: dict, course_with_code: dict
):
    """学生凭有效课程码入课 → 200，返回 course_id 且 already_enrolled=false。"""
    r = await client.post(
        "/api/courses/join",
        headers=auth_headers,
        json={"join_code": course_with_code["join_code"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["course_id"] == course_with_code["course_id"]
    assert data["already_enrolled"] is False


@pytest.mark.asyncio
async def test_join_course_duplicate(
    client: AsyncClient, auth_headers: dict, course_with_code: dict
):
    """重复加入同一课程 → 200，already_enrolled=true。"""
    code = course_with_code["join_code"]
    await client.post(
        "/api/courses/join",
        headers=auth_headers,
        json={"join_code": code},
    )
    r = await client.post(
        "/api/courses/join",
        headers=auth_headers,
        json={"join_code": code},
    )
    assert r.status_code == 200
    assert r.json()["already_enrolled"] is True


@pytest.mark.asyncio
async def test_join_accepts_dashed_code(
    client: AsyncClient, auth_headers: dict, course_with_code: dict
):
    """带 4-4 分隔符的课程码仍能入课（normalize 去连字符）。"""
    raw = course_with_code["join_code"]
    dashed = f"{raw[:4]}-{raw[4:]}"
    r = await client.post(
        "/api/courses/join",
        headers=auth_headers,
        json={"join_code": dashed},
    )
    assert r.status_code == 200, r.text
    assert r.json()["name"] == course_with_code["name"]


@pytest.mark.asyncio
async def test_join_accepts_lowercase_and_spaces(
    client: AsyncClient, auth_headers: dict, course_with_code: dict
):
    """小写 + 前后空格的课程码仍能入课（normalize 转大写、去空白）。"""
    raw = course_with_code["join_code"]
    messy = f"  {raw.lower()}  "
    r = await client.post(
        "/api/courses/join",
        headers=auth_headers,
        json={"join_code": messy},
    )
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_join_invalid_code_404(client: AsyncClient, auth_headers: dict):
    """不存在的课程码 → 404。"""
    r = await client.post(
        "/api/courses/join",
        headers=auth_headers,
        json={"join_code": "NOPE-NOPE"},
    )
    assert r.status_code == 404
