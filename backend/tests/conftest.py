"""Pytest fixtures: in-memory SQLite app + httpx AsyncClient.

Import order matters: env vars must be set before any backend module is imported.
"""
from __future__ import annotations

import os

# Prevent transformers (pulled in by langsmith pytest plugin) from loading TF/Keras
os.environ.setdefault("USE_TF", "0")

# Set all env vars before importing backend modules（嵌套式 env 名，对齐 settings 组合式重构）
os.environ.setdefault("DB__URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DB__REDIS_URL", "memory://")
os.environ.setdefault("LLM__API_KEY", "sk-test")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")
os.environ.setdefault("TESTING", "1")

import pytest
from httpx import ASGITransport, AsyncClient

from core.db.database import close_db, init_db


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
    """注册一名测试管理员并升级为 admin 角色。

    H-18：注册接口不再因用户名自动提权（消除「撞名 admin 即 admin」攻击面），
    故测试不能靠注册 ``username="admin"`` 拿到管理员——那样得到的只是 student。
    这里注册一个普通用户后，直接在 DB 把 role 升为 admin（模拟 DBA / 初始 bootstrap
    通道，对齐生产「admin 仅由安全通道授予」的语义），再走登录拿 token。
    """
    from sqlalchemy import update

    from core.db.database import AsyncSessionLocal, User

    username = f"admin_{os.urandom(3).hex()}"
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "adminpass123", "display_name": "Admin"},
    )
    assert r.status_code == 200, r.text
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(User).where(User.username == username).values(role="admin", is_admin=True)
        )
        await db.commit()
    r = await client.post(
        "/api/auth/login",
        json={"username": username, "password": "adminpass123"},
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
