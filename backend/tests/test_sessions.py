"""Session API: course access + ownership isolation."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_session_requires_course_access(client: AsyncClient, auth_headers: dict):
    r = await client.post(
        "/api/sessions",
        headers=auth_headers,
        json={"course_id": "nonexistent_course", "title": "t"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_session_ownership_isolation(client: AsyncClient, admin_headers: dict):
    """admin 创建的 session，普通用户无权访问（403）。

    H-18 连带：admin 不再靠「注册 username=admin 自动提权」获得，改用 conftest 的
    admin_headers fixture（注册后经安全通道授予 admin 角色），消除对废弃行为的依赖。
    """
    u1 = f"sess_u1_{__import__('os').urandom(3).hex()}"

    r1 = await client.post(
        "/api/auth/register",
        json={"username": u1, "password": "pass1234"},
    )
    h1 = {"Authorization": f"Bearer {r1.json()['token']}"}

    created = await client.post(
        "/api/sessions",
        headers=admin_headers,
        json={"course_id": "stamp", "title": "admin session"},
    )
    assert created.status_code == 200
    session_id = created.json()["id"]

    forbidden = await client.get(f"/api/sessions/{session_id}", headers=h1)
    assert forbidden.status_code == 403


# ── Happy-path: admin 创建/查询/修改/删除 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_create_session_returns_id(client: AsyncClient, admin_headers: dict):
    """管理员可为任意 course_id 创建会话（admin 绕过课程权限），响应含 id 字段。"""
    r = await client.post(
        "/api/sessions",
        headers=admin_headers,
        json={"course_id": "tc_happy", "title": "Happy Session"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["title"] == "Happy Session"


@pytest.mark.asyncio
async def test_list_sessions_after_create(client: AsyncClient, admin_headers: dict):
    """创建会话后，GET /api/sessions?course_id=... 列表中可见该会话。"""
    await client.post(
        "/api/sessions",
        headers=admin_headers,
        json={"course_id": "tc_list", "title": "Listed"},
    )
    r = await client.get("/api/sessions", headers=admin_headers, params={"course_id": "tc_list"})
    assert r.status_code == 200
    sessions = r.json()["sessions"]
    assert any(s["title"] == "Listed" for s in sessions)


@pytest.mark.asyncio
async def test_get_session_by_id(client: AsyncClient, admin_headers: dict):
    """GET /api/sessions/{id} 返回对应会话详情。"""
    cr = await client.post(
        "/api/sessions",
        headers=admin_headers,
        json={"course_id": "tc_get", "title": "GetMe"},
    )
    sid = cr.json()["id"]
    r = await client.get(f"/api/sessions/{sid}", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["id"] == sid


@pytest.mark.asyncio
async def test_patch_session_title(client: AsyncClient, admin_headers: dict):
    """PATCH /api/sessions/{id} 修改标题后返回 ok=true。"""
    cr = await client.post(
        "/api/sessions",
        headers=admin_headers,
        json={"course_id": "tc_patch", "title": "OldTitle"},
    )
    sid = cr.json()["id"]
    r = await client.patch(
        f"/api/sessions/{sid}",
        headers=admin_headers,
        json={"title": "NewTitle"},
    )
    assert r.status_code == 200
    assert r.json().get("ok") is True


@pytest.mark.asyncio
async def test_delete_session(client: AsyncClient, admin_headers: dict):
    """DELETE 会话后再 GET 返回 404。"""
    cr = await client.post(
        "/api/sessions",
        headers=admin_headers,
        json={"course_id": "tc_del", "title": "ToDelete"},
    )
    sid = cr.json()["id"]
    dr = await client.delete(f"/api/sessions/{sid}", headers=admin_headers)
    assert dr.status_code == 200
    assert dr.json().get("ok") is True
    gr = await client.get(f"/api/sessions/{sid}", headers=admin_headers)
    assert gr.status_code == 404


@pytest.mark.asyncio
async def test_message_attachments_persisted_and_echoed(client: AsyncClient, admin_headers: dict):
    """POST /messages 带 attachments → GET 回显 metadata.attachments 且 base64 被清。

    验证附件随消息持久化（attachments_json），且持久化层剥掉
    base64（省 DB 空间 + 不泄露原文字节）。
    """
    cr = await client.post(
        "/api/sessions",
        headers=admin_headers,
        json={"course_id": "tc_att", "title": "Attachments"},
    )
    sid = cr.json()["id"]

    await client.post(
        f"/api/sessions/{sid}/messages",
        headers=admin_headers,
        json={
            "role": "user",
            "content": "看这张图",
            "attachments": [
                {
                    "type": "image",
                    "url": "/api/uploads/1_abc.png",
                    "filename": "abc.png",
                    "base64": "SHOULD_BE_STRIPPED",
                },
            ],
        },
    )

    r = await client.get(f"/api/sessions/{sid}/messages", headers=admin_headers)
    msgs = r.json()["messages"]
    assert len(msgs) >= 1
    atts = msgs[-1].get("metadata", {}).get("attachments")
    assert atts and len(atts) == 1
    assert atts[0]["url"] == "/api/uploads/1_abc.png"
    assert "base64" not in atts[0]  # 铁律：持久化清 base64
