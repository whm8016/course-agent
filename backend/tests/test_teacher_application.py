"""教师申请-审批流端到端测试。

覆盖：申请提交、防重复、状态机守卫、审批事务原子性（role+通知）、
rejected 可重申、权限隔离、邀请码快速通道回归。
"""
from __future__ import annotations

import pytest


async def _apply(client, headers, reason="任教高等数学，需教师权限管理课程"):
    return await client.post(
        "/api/auth/apply-teacher", headers=headers, json={"reason": reason}
    )


async def _register_student(client, username="stu"):
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "testpass123", "display_name": "S"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}, r.json()["user"]["id"]


@pytest.mark.asyncio
async def test_student_apply_success(client, auth_headers):
    r = await _apply(client, auth_headers)
    assert r.status_code == 201
    assert r.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_apply_duplicate_pending_409(client, auth_headers):
    await _apply(client, auth_headers)
    r = await _apply(client, auth_headers)
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_admin_cannot_apply_409(client, admin_headers):
    """已是 admin/teacher 的用户无需申请。"""
    r = await _apply(client, admin_headers)
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_my_application_status(client, auth_headers):
    """申请后 /me 能查到 pending 状态。"""
    await _apply(client, auth_headers)
    r = await client.get("/api/auth/teacher-applications/me", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_admin_approve_promotes_role_and_notifies(client, admin_headers):
    """通过申请 → 用户角色升 teacher + 写入站内通知（事务原子）。"""
    stu_headers, _ = await _register_student(client, "stu_approve")
    app = (await _apply(client, stu_headers)).json()

    r = await client.post(
        f"/api/admin/teacher-applications/{app['id']}/approve",
        headers=admin_headers,
    )
    assert r.status_code == 200
    assert r.json()["role"] == "teacher"

    # 角色确实变更（/auth/me 读 DB 真实状态）
    me = (await client.get("/api/auth/me", headers=stu_headers)).json()
    assert me["user"]["role"] == "teacher"

    # 站内通知已写入（同一事务）
    notifs = (
        await client.get("/api/bot/notifications", headers=stu_headers)
    ).json()
    contents = " ".join(n["content"] for n in notifs["notifications"])
    assert "通过" in contents


@pytest.mark.asyncio
async def test_approve_non_pending_409(client, admin_headers, auth_headers):
    """已通过的申请再次审批 → 409（状态机守卫，终态不可逆）。"""
    app = (await _apply(client, auth_headers)).json()
    await client.post(
        f"/api/admin/teacher-applications/{app['id']}/approve",
        headers=admin_headers,
    )
    r = await client.post(
        f"/api/admin/teacher-applications/{app['id']}/approve",
        headers=admin_headers,
    )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_reject_keeps_student_and_allows_reapply(client, admin_headers):
    """拒绝后角色仍 student，且能重新申请（rejected 不阻塞）。"""
    stu_headers, _ = await _register_student(client, "stu_reject")
    app = (await _apply(client, stu_headers)).json()

    r = await client.post(
        f"/api/admin/teacher-applications/{app['id']}/reject",
        headers=admin_headers,
        json={"note": "材料不足"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"

    me = (await client.get("/api/auth/me", headers=stu_headers)).json()
    assert me["user"]["role"] == "student"

    # 可重新申请（部分唯一索引只锁 pending，不锁 rejected）
    r2 = await _apply(client, stu_headers)
    assert r2.status_code == 201


@pytest.mark.asyncio
async def test_reject_writes_note_notification(client, admin_headers):
    """拒绝通知带理由。"""
    stu_headers, _ = await _register_student(client, "stu_note")
    app = (await _apply(client, stu_headers)).json()
    await client.post(
        f"/api/admin/teacher-applications/{app['id']}/reject",
        headers=admin_headers,
        json={"note": "请补充教师资格证明"},
    )
    notifs = (
        await client.get("/api/bot/notifications", headers=stu_headers)
    ).json()
    contents = " ".join(n["content"] for n in notifs["notifications"])
    assert "未通过" in contents
    assert "教师资格证明" in contents


@pytest.mark.asyncio
async def test_list_applications_filter(client, admin_headers, auth_headers):
    await _apply(client, auth_headers)
    r = await client.get(
        "/api/admin/teacher-applications?status=pending", headers=admin_headers
    )
    assert r.status_code == 200
    assert all(a["status"] == "pending" for a in r.json())


@pytest.mark.asyncio
async def test_student_cannot_access_admin_applications(client, auth_headers):
    """学生访问 admin 审批接口 → 403。"""
    r = await client.get("/api/admin/teacher-applications", headers=auth_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_invite_code_fast_track_still_works(client, admin_headers):
    """邀请码快速通道回归：admin 发码 → 学生填码注册秒升 teacher（与申请-审批并存）。"""
    inv = await client.post(
        "/api/admin/invite-codes", headers=admin_headers, json={"count": 1}
    )
    assert inv.status_code == 201
    code = inv.json()["codes"][0]

    r = await client.post(
        "/api/auth/register",
        json={
            "username": "stu_invite",
            "password": "testpass123",
            "invite_code": code,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["user"]["role"] == "teacher"
