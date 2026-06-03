"""Pytest fixtures: in-memory SQLite app + httpx AsyncClient.

Import order matters: env vars must be set before any backend module is imported.
"""
from __future__ import annotations

import os

# Set all env vars before importing backend modules
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("REDIS_URL", "memory://")
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("ALLOWED_ORIGINS", "*")
os.environ.setdefault("TESTING", "1")

import pytest
from httpx import ASGITransport, AsyncClient

from core.db.database import close_db, engine, init_db


@pytest.fixture
async def client():
    from main import app

    await init_db()

    # SQLite in-memory: every engine connect uses the same connection for schema
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await close_db()


@pytest.fixture
async def auth_headers(client: AsyncClient):
    username = f"u{os.urandom(4).hex()}"
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "testpass123", "display_name": "T"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def admin_headers(client: AsyncClient):
    """Register or login as the built-in admin user."""
    r = await client.post(
        "/api/auth/register",
        json={"username": "admin", "password": "adminpass123", "display_name": "Admin"},
    )
    if r.status_code == 409:
        r = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "adminpass123"},
        )
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def course_with_code(client: AsyncClient, admin_headers: dict):
    """管理员创建测试课程并生成课程码，供 courses/sessions 相关测试使用。"""
    course_id = f"tc_{os.urandom(3).hex()}"
    r = await client.post(
        "/api/admin/kb",
        headers=admin_headers,
        json={"course_id": course_id, "name": "测试课程", "is_visible": True},
    )
    assert r.status_code == 201, r.text
    r2 = await client.post(
        f"/api/teacher/courses/{course_id}/join-code",
        headers=admin_headers,
    )
    assert r2.status_code == 200, r2.text
    return {"course_id": course_id, "join_code": r2.json()["join_code"], "name": "测试课程"}


@pytest.fixture
async def enrolled_user_headers(client: AsyncClient, course_with_code: dict):
    """注册学生并凭课程码入课，返回该学生的 auth headers。"""
    username = f"stu_{os.urandom(3).hex()}"
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "testpass123", "display_name": "Student"},
    )
    assert r.status_code == 200, r.text
    headers = {"Authorization": f"Bearer {r.json()['token']}"}
    r2 = await client.post(
        "/api/courses/join",
        headers=headers,
        json={"join_code": course_with_code["join_code"]},
    )
    assert r2.status_code == 200, r2.text
    return headers
