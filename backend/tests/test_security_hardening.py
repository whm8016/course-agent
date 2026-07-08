"""H-1~H-6 安全加固回归测试（攻击者视角）。

每个测试模拟恶意输入，证明被正确拒绝（403/404/静默丢弃），而非 500 异常或 200 放行。
这是安全修复的"门禁"：防止后续重构把校验改回漏洞形态。
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# H-1 Turn IDOR：trm 因归属不符返回 False → answer_now 端点 404（非 200/500）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h1_answer_now_idor_returns_404(client: AsyncClient, auth_headers: dict):
    """B 用户拿 A 的 turn_id 调 answer_now：trm 归属校验返回 False → 端点 404。"""
    fake_trm = SimpleNamespace(request_answer_now=AsyncMock(return_value=False))
    with patch("api.chat.get_turn_runtime_manager", return_value=fake_trm):
        r = await client.post(
            "/api/chat/answer_now",
            headers=auth_headers,
            json={"turn_id": "victim-turn-id"},
        )
    assert r.status_code == 404
    # trm 收到的是当前登录用户的 id（归属校验由 trm 内部完成，见 test_agent_loop）
    fake_trm.request_answer_now.assert_awaited_once()
    assert "user_id" in fake_trm.request_answer_now.call_args.kwargs


# ---------------------------------------------------------------------------
# H-2 LightRAG 课程绕过：非 owner 调 index → check_course_access 抛 403，索引不触发
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h2_lightrag_index_denies_non_owner(client: AsyncClient, admin_headers: dict):
    """teacher 对非自己 course_id 触发索引 → 403，get_indexer 绝不被调用。"""
    from fastapi import HTTPException

    async def _deny(*a, **kw):
        raise HTTPException(status_code=403, detail="无权访问此课程")

    with patch("api.lightrag.is_lightrag_available", return_value=(True, "")), \
         patch("api.lightrag.check_course_access", new=AsyncMock(side_effect=_deny)), \
         patch("api.lightrag.get_indexer") as gi:
        r = await client.post(
            "/api/chat/lightrag/index",
            headers=admin_headers,
            json={"course_id": "not-mine"},
        )
    assert r.status_code == 403
    gi.assert_not_called()  # 归属拒绝发生在索引之前


# ---------------------------------------------------------------------------
# H-3 exam_mimic 路径遍历：paper_path 越界 → _validate_paper_path 返回 None
# ---------------------------------------------------------------------------

def test_h3_paper_path_traversal_rejected():
    from api.question import _validate_paper_path
    from core.question.path import get_question_dir

    # 绝对路径越界
    assert _validate_paper_path("/etc/passwd") is None
    # 相对路径穿越
    assert _validate_paper_path("../../etc/passwd") is None
    assert _validate_paper_path("../../../windows/system32/config/sam") is None
    # 合法：question 目录内
    legit = get_question_dir() / "mimic_papers" / "x.pdf"
    assert _validate_paper_path(str(legit)) is not None


# ---------------------------------------------------------------------------
# H-4 file_path 任意文件读取：越界路径不读盘，图片被丢弃
# ---------------------------------------------------------------------------

def test_h4_image_path_resolver_rejects_out_of_root():
    from core.llm.multimodal import _resolve_image_within_allowed_roots

    assert _resolve_image_within_allowed_roots("/etc/passwd") is None
    assert _resolve_image_within_allowed_roots("../../etc/passwd") is None
    assert _resolve_image_within_allowed_roots("C:\\Windows\\System32\\drivers\\etc\\hosts") is None
    # 合法：系统临时目录内（pytest tmp_path 同根，保纯函数测试通过）
    import tempfile
    legit = Path(tempfile.gettempdir()) / "ok.png"
    assert _resolve_image_within_allowed_roots(str(legit)) is not None


def test_h4_inject_drops_out_of_root_image():
    """越界 file_path 的图片被丢弃：不读盘、不注入 image part、不抛异常。"""
    from core.attachment import from_image_path
    from core.llm.multimodal import prepare_multimodal_messages

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "看图"}]
    atts = [from_image_path("/etc/passwd")]
    out = prepare_multimodal_messages(msgs, atts, "dashscope")

    content = out[-1]["content"]
    assert isinstance(content, list)
    image_parts = [c for c in content if c.get("type") in ("image_url", "image")]
    assert image_parts == []  # 攻击者的越界图片被丢弃，无 image part


# ---------------------------------------------------------------------------
# H-5 Notebook IDOR：A 删 B 的 category_id → False，B 的 category 与连接表都不受影响
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h5_delete_category_idor_leaves_victim_intact(client: AsyncClient):
    """攻击者删别人的 category：返回 False，且 victim 的 category 与连接表均完好。

    旧实现（先无条件删连接表）会让此处连接表断言失败（len==0）。"""
    from sqlalchemy import select
    from core.db import notebook_store as nb
    from core.db.database import (
        AsyncSessionLocal, NotebookCategory, NotebookEntry, NotebookEntryCategory,
    )

    OWNER = "owner-1"
    ATTACKER = "attacker-1"
    async with AsyncSessionLocal() as db:
        db.add(NotebookCategory(id=100, user_id=OWNER, name="owner的分类"))
        db.add(NotebookEntry(
            id=200, user_id=OWNER, session_id="s1", question_id="q1", question="Q",
        ))
        db.add(NotebookEntryCategory(entry_id=200, category_id=100))
        await db.commit()

    # 攻击者尝试删 owner 的 category
    async with AsyncSessionLocal() as db:
        deleted = await nb.delete_category(db, ATTACKER, 100)
        await db.commit()
    assert deleted is False

    # owner 的 category 与连接表都完好（关键：连接表未被无条件删除）
    async with AsyncSessionLocal() as db:
        cat = (await db.execute(
            select(NotebookCategory).where(NotebookCategory.id == 100)
        )).scalar_one_or_none()
        assert cat is not None
        links = (await db.execute(
            select(NotebookEntryCategory).where(NotebookEntryCategory.category_id == 100)
        )).all()
        assert len(links) == 1  # 旧实现此处会是 0

    # owner 自己能删（正向路径不受影响）
    async with AsyncSessionLocal() as db:
        deleted = await nb.delete_category(db, OWNER, 100)
        await db.commit()
    assert deleted is True
    async with AsyncSessionLocal() as db:
        cat = (await db.execute(
            select(NotebookCategory).where(NotebookCategory.id == 100)
        )).scalar_one_or_none()
        assert cat is None


# ---------------------------------------------------------------------------
# H-6 课程访问缓存失效：remove_student 删 enrollment 后调用 course_access_invalidate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h6_remove_student_invalidates_access_cache(
    client: AsyncClient, admin_headers: dict, course_with_code: dict,
):
    """教师踢学生 → 删除 enrollment 后必须失效该学生的课程访问缓存。"""
    cid = course_with_code["course_id"]

    # 注册学生 + 教师加学生，拿到 student_id
    username = f"stu_{os.urandom(3).hex()}"
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "testpass123", "display_name": "S"},
    )
    assert r.status_code == 200, r.text
    r2 = await client.post(
        f"/api/teacher/courses/{cid}/students",
        headers=admin_headers,
        json={"username": username},
    )
    assert r2.status_code in (200, 201), r2.text
    sid = r2.json()["student_id"]

    # spy course_access_invalidate
    with patch("api.teacher.course_access_invalidate", new=AsyncMock()) as inv:
        r3 = await client.delete(
            f"/api/teacher/courses/{cid}/students/{sid}", headers=admin_headers,
        )
    assert r3.status_code == 200, r3.text
    inv.assert_awaited_once_with(sid, cid)
