"""读侧游标尾部载入回归（L2 返修步骤 2：load_uncompressed_tail）。

目标不变式：任一条会话消息，要么在摘要卡里（游标之前），要么在本轮原文里（游标
之后）——本函数兜住后半段，与压缩何时跑无关。取代读侧按条数砍 history[-10:] 的
旧口径（写侧游标推进与读侧条数截断不同源，滑出历史与写进卡之间存在 1-2 轮真空）。

真实 in-memory SQLite（client fixture 触发 init_db 建表）。
"""
from __future__ import annotations

import pytest

from core.db.database import AsyncSessionLocal, Message, Session
from core.memory.session_summary import load_uncompressed_tail


def _msgs(n: int):
    """n 条消息 (id m00..m{n-1}，created_at=1000+i)，role 交替。"""
    return [(f"m{i:02d}", "user" if i % 2 == 0 else "assistant", f"c{i}", 1000.0 + i) for i in range(n)]


async def _seed(sid, msg_count, *, user_id="u1", up_to_msg_id=None, up_to_ca=None):
    db = AsyncSessionLocal()
    db.add(Session(
        id=sid, course_id="c1", user_id=user_id, title="t",
        summary_up_to_msg_id=up_to_msg_id, summary_up_to_created_at=up_to_ca,
    ))
    for mid, role, content, ca in _msgs(msg_count):
        db.add(Message(id=mid, session_id=sid, role=role, content=content, created_at=ca))
    await db.commit()
    await db.close()


@pytest.mark.asyncio
async def test_tail_returns_all_post_cursor_ascending(client):
    """游标 m03 -> 返回 (m03, m09] 全部 6 条、升序、{role, content} 形状。"""
    await _seed("s-tail-1", 10, up_to_msg_id="m03", up_to_ca=1003.0)
    async with AsyncSessionLocal() as db:
        out = await load_uncompressed_tail(db, "s-tail-1")
    assert [m["content"] for m in out] == [f"c{i}" for i in range(4, 10)]
    assert out[0] == {"role": "user", "content": "c4"}  # m04: i=4 偶数=user
    assert out[-1] == {"role": "assistant", "content": "c9"}


@pytest.mark.asyncio
async def test_tail_no_cursor_returns_everything(client):
    """无游标（从未压缩）-> 全量返回——旧 [-10:] 口径会砍掉前 2 条（空窗来源）。"""
    await _seed("s-tail-2", 12)
    async with AsyncSessionLocal() as db:
        out = await load_uncompressed_tail(db, "s-tail-2")
    assert len(out) == 12


@pytest.mark.asyncio
async def test_tail_legacy_null_created_at_cursor(client):
    """存量行 summary_up_to_created_at=NULL -> 走 msg_id 反查兼容路径，等价区间。"""
    await _seed("s-tail-3", 10, up_to_msg_id="m01", up_to_ca=None)
    async with AsyncSessionLocal() as db:
        out = await load_uncompressed_tail(db, "s-tail-3")
    assert [m["content"] for m in out] == [f"c{i}" for i in range(2, 10)]


@pytest.mark.asyncio
async def test_tail_deleted_cursor_message_falls_back_full(client):
    """游标消息已被删（CASCADE）-> 无下界全量返回（不丢消息优先于精确区间）。"""
    await _seed("s-tail-4", 8, up_to_msg_id="gone", up_to_ca=None)
    async with AsyncSessionLocal() as db:
        out = await load_uncompressed_tail(db, "s-tail-4")
    assert len(out) == 8


@pytest.mark.asyncio
async def test_tail_ownership_filter_blocks_foreign_user(client):
    """归属过滤：session 不属于该 user_id -> []（防 B 带 A 的 session_id 越权读原文）。"""
    await _seed("s-tail-5", 6, user_id="owner-u")
    async with AsyncSessionLocal() as db:
        assert await load_uncompressed_tail(db, "s-tail-5", user_id="owner-u") != []
        assert await load_uncompressed_tail(db, "s-tail-5", user_id="attacker-u") == []


@pytest.mark.asyncio
async def test_tail_missing_session_returns_empty(client):
    async with AsyncSessionLocal() as db:
        assert await load_uncompressed_tail(db, "no-such") == []
