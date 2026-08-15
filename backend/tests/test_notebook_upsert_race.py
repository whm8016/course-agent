"""notebook upsert 并发加固测试。

upsert_notebook_entry 是 SELECT-then-INSERT：两个并发请求同键同时未命中 → 双 INSERT，
第二个撞 uq_notebook_entry_natural。修复：IntegrityError → 回滚重查 → 改走 UPDATE
（对齐 create_category 的既有处理）。前端"提交答案"逐题 upsert 接通后该路径真实可达。
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy import func, select


async def _setup_user_session(prefix: str) -> tuple[str, str]:
    """注册测试用户 + 建 Session 行，返回 (user_id, session_id)。"""
    from core.db.database import AsyncSessionLocal, Session, User

    username = f"{prefix}_{os.urandom(3).hex()}"
    async with AsyncSessionLocal() as db:
        u = User(username=username, password_hash="x")
        db.add(u)
        await db.flush()
        uid = u.id
        sid = f"sess_{prefix}_{os.urandom(3).hex()}"
        db.add(Session(id=sid, course_id=f"c_{prefix}", user_id=uid))
        await db.commit()
    return uid, sid


@pytest.mark.asyncio
async def test_upsert_integrity_error_falls_back_to_update(client, monkeypatch):
    """并发同键双 INSERT：第二个请求撞唯一约束 → 回退 UPDATE，不抛 500、不产生第二行。"""
    import core.db.notebook_store as nb
    from core.db.database import AsyncSessionLocal, NotebookEntry

    uid, sid = await _setup_user_session("race")

    # "另一个并发请求"已抢先落库同键行
    async with AsyncSessionLocal() as db:
        await nb.upsert_notebook_entry(db, uid, {
            "session_id": sid, "question_id": "q1", "question": "1+1=?",
            "user_answer": "2", "is_correct": True,
        })
        await db.commit()

    # 模拟竞态：本请求的存在性检查未命中（抢先者尚未提交可见），
    # 但 INSERT 撞约束后的重查走真实查询（能看到已提交行）
    real_find = nb._find_entry_row
    state = {"first": True}

    async def racing_find(db, user_id, session_id, question_id):
        if state["first"]:
            state["first"] = False
            return None
        return await real_find(db, user_id, session_id, question_id)

    monkeypatch.setattr(nb, "_find_entry_row", racing_find)

    async with AsyncSessionLocal() as db:
        result = await nb.upsert_notebook_entry(db, uid, {
            "session_id": sid, "question_id": "q1", "question": "1+1=?",
            "user_answer": "3", "is_correct": False,
        })
        await db.commit()

    assert result["user_answer"] == "3"
    assert result["is_correct"] is False
    async with AsyncSessionLocal() as db:
        count, row = (await db.execute(
            select(func.count(NotebookEntry.id), NotebookEntry).where(
                NotebookEntry.user_id == uid, NotebookEntry.session_id == sid
            )
        )).one()
        assert count == 1  # 无重复行
        assert row.user_answer == "3"  # 落的是本次（最后到达）的值


@pytest.mark.asyncio
async def test_upsert_second_call_updates_in_place(client):
    """顺序两次 upsert 同键（重答场景）→ 行数 1、字段为最新值（last-write-wins 回归保护）。"""
    import core.db.notebook_store as nb
    from core.db.database import AsyncSessionLocal, NotebookEntry

    uid, sid = await _setup_user_session("seq")

    async with AsyncSessionLocal() as db:
        first = await nb.upsert_notebook_entry(db, uid, {
            "session_id": sid, "question_id": "q2", "question": "2+2=?",
            "user_answer": "5", "is_correct": False,
        })
        await db.commit()
    async with AsyncSessionLocal() as db:
        second = await nb.upsert_notebook_entry(db, uid, {
            "session_id": sid, "question_id": "q2", "question": "2+2=?",
            "user_answer": "4", "is_correct": True,
        })
        await db.commit()

    assert first["id"] == second["id"]  # 同一行
    assert second["user_answer"] == "4"
    assert second["is_correct"] is True
    assert second["created_at"] == first["created_at"]  # created 不随重答漂移
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(NotebookEntry).where(
                NotebookEntry.user_id == uid, NotebookEntry.session_id == sid
            )
        )).scalars().all()
    assert len(rows) == 1
