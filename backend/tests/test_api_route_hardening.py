"""M-50 / M-51 API 路由层加固回归测试（攻击者视角）。

每个测试模拟恶意输入，证明被正确拒绝（403/400），而非 500 异常或 200 放行：
- M-50：教师越权向他人课程广播 / 向非自己课程的学生单发通知（IDOR）。
- M-51：知识库上传携带 ``../`` 路径分隔符的文件名（path traversal）。

这是安全修复的"门禁"：防止后续重构把校验改回漏洞形态。
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


async def _teacher_token(client: AsyncClient) -> dict:
    """注册一名普通教师（非任何课程 owner）并返回其 auth headers。

    课程由 admin 通过 course_with_code fixture 拥有，本教师与该课程无任何归属关系，
    用来模拟"教师 B 试图操作 admin/教师 A 的课程/学生"的越权攻击者。
    """
    from sqlalchemy import update

    from core.db.database import AsyncSessionLocal, User

    username = f"tch_{os.urandom(3).hex()}"
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "teachpass123", "display_name": "T"},
    )
    assert r.status_code == 200, r.text
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(User).where(User.username == username).values(role="teacher")
        )
        await db.commit()
    r2 = await client.post(
        "/api/auth/login", json={"username": username, "password": "teachpass123"}
    )
    assert r2.status_code == 200, r2.text
    return {"Authorization": f"Bearer {r2.json()['token']}"}


# ---------------------------------------------------------------------------
# M-50 /bot/notify 越权
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m50_broadcast_to_other_course_denied(
    client: AsyncClient, course_with_code: dict
):
    """教师 B 向 admin 拥有的课程群发通知 → 403，broadcast 不被调用。

    攻击者视角：登录一个无关教师，传他人 course_id 调 /bot/notify。
    修复前：只校验角色（teacher），直接放行 → broadcast 会把陌生教师的推送灌进
    该课程全部学生的 IM 绑定（IDOR + 骚扰/钓鱼面）。
    """
    headers = await _teacher_token(client)
    cid = course_with_code["course_id"]

    with patch("api.bot.NotificationService") as svc_cls:
        svc_cls.return_value.broadcast = AsyncMock(return_value=0)
        r = await client.post(
            "/api/bot/notify",
            headers=headers,
            json={"course_id": cid, "content": "恶意广播"},
        )
    assert r.status_code == 403, r.text
    svc_cls.return_value.broadcast.assert_not_awaited()


@pytest.mark.asyncio
async def test_m50_push_to_stranger_student_denied(client: AsyncClient):
    """教师 B 向不在自己任何课程里的用户单发通知 → 403，push_to_student 不被调用。

    攻击者视角：教师传任意 user_id 给陌生用户发私信。
    修复前：只校验角色，直接 push_to_student → 任意教师可私信任意学生。
    """
    # 先注册一个"受害者"学生（不属于任何教师的课程）
    r_v = await client.post(
        "/api/auth/register",
        json={"username": f"vic_{os.urandom(3).hex()}", "password": "pass1234", "display_name": "V"},
    )
    victim_token = r_v.json()["token"]
    victim_id = (await client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {victim_token}"}
    )).json()["user"]["id"]

    headers = await _teacher_token(client)
    with patch("api.bot.NotificationService") as svc_cls:
        svc_cls.return_value.push_to_student = AsyncMock(return_value=[])
        r = await client.post(
            "/api/bot/notify",
            headers=headers,
            json={"user_id": victim_id, "content": "私信"},
        )
    assert r.status_code == 403, r.text
    svc_cls.return_value.push_to_student.assert_not_awaited()


@pytest.mark.asyncio
async def test_m50_admin_can_broadcast_any_course(
    client: AsyncClient, course_with_code: dict, admin_headers: dict
):
    """正向用例：admin 对任意课程广播 → 通过归属校验（admin 放行，check_course_access）。

    确保修复没有误伤合法路径：admin 是课程 owner，broadcast 正常被调用。
    """
    cid = course_with_code["course_id"]
    with patch("api.bot.NotificationService") as svc_cls:
        svc_cls.return_value.broadcast = AsyncMock(return_value=0)
        r = await client.post(
            "/api/bot/notify",
            headers=admin_headers,
            json={"course_id": cid, "content": "正常广播"},
        )
    assert r.status_code == 200, r.text
    svc_cls.return_value.broadcast.assert_awaited_once()


# ---------------------------------------------------------------------------
# M-51 upload 文件名 path traversal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m51_admin_upload_rejects_traversal_filename(
    client: AsyncClient, course_with_code: dict, admin_headers: dict
):
    """知识库上传文件名含 ``../`` → 400（不落盘到 raw_dir 之外）。

    攻击者视角：admin（被盗/共用账号）上传 ``filename="../../../evil.pdf"``。
    修复前：safe_name = f"{uuid}_{filename}" → 写到 raw_dir/../../../evil.pdf（穿越）。
    """
    cid = course_with_code["course_id"]
    files = {"files": ("../../../../evil.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = await client.post(
        f"/api/admin/kb/{cid}/upload", headers=admin_headers, files=files
    )
    assert r.status_code == 400, r.text
    assert "无效" in r.json()["detail"]


@pytest.mark.asyncio
async def test_m51_admin_upload_rejects_pathsep_filename(
    client: AsyncClient, course_with_code: dict, admin_headers: dict
):
    """文件名含路径分隔符 ``/`` → 400（防子目录写入）。"""
    cid = course_with_code["course_id"]
    files = {"files": ("sub/dir/x.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = await client.post(
        f"/api/admin/kb/{cid}/upload", headers=admin_headers, files=files
    )
    assert r.status_code == 400, r.text


@pytest.mark.asyncio
async def test_m51_teacher_upload_rejects_traversal_filename(
    client: AsyncClient, course_with_code: dict, admin_headers: dict
):
    """teacher 端点同样拒绝 ``../`` 文件名（M-51 同根因，两端点共用 _safe_upload_name）。

    admin 同时满足 get_current_teacher 且 _get_owned_kb 对 admin 放行，故用 admin_headers
    直接打 teacher upload 路由即可覆盖 teacher 端点的清洗路径。
    """
    cid = course_with_code["course_id"]
    files = {"files": ("../evil.pdf", b"%PDF-1.4 fake", "application/pdf")}
    r = await client.post(
        f"/api/teacher/courses/{cid}/upload", headers=admin_headers, files=files
    )
    assert r.status_code == 400, r.text


def test_m51_safe_upload_name_helper_rejects_control_chars():
    """单元层攻击者测试：_safe_upload_name 直接拒控制字符 / 空名 / ``.``。

    把 helper 当作门禁单独验证，避免端点测试被文件 IO 干扰。
    """
    from fastapi import HTTPException

    from api.admin import _safe_upload_name

    # 合法名放行
    assert _safe_upload_name("教材.pdf") == "教材.pdf"
    assert _safe_upload_name("a normal file.pdf") == "a normal file.pdf"
    # 拒绝穿越 / 分隔符 / 空名 / 控制字符
    for bad in ("../evil.pdf", "a/b.pdf", "x\\y.pdf", "", None, ".", "..", "a\x00b.pdf"):
        with pytest.raises(HTTPException):
            _safe_upload_name(bad)


@pytest.mark.asyncio
async def test_m51_normal_upload_still_works(
    client: AsyncClient, course_with_code: dict, admin_headers: dict
):
    """正向用例：合法文件名（纯文件名 + 合法扩展名）正常上传 → 200。

    确保清洗没有误伤合法文件名。
    """
    cid = course_with_code["course_id"]
    files = {"files": ("教材.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    r = await client.post(
        f"/api/admin/kb/{cid}/upload", headers=admin_headers, files=files
    )
    assert r.status_code == 200, r.text
    assert r.json()["uploaded"] == ["教材.pdf"]
