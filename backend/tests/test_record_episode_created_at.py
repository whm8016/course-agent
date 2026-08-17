"""record_episode 的 created_at 注入口径（表E live 拨钟缝 1）。

评测拨钟：暴露轮的 episode.created_at = 剧本钟（对话发生时刻），巩固写观测
按 episode.created_at 取时间（consolidation._write_mastery_observations）——
拨的是「对话发生时刻」，不是写入时刻。线上不传 created_at → 列默认墙钟，零行为变化。
"""
import time

import pytest


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def _ensure_user(user_id: str) -> None:
    from core.db.database import AsyncSessionLocal, User

    async with AsyncSessionLocal() as s:
        s.add(User(id=user_id, username=user_id, password_hash="x", role="student"))
        await s.commit()


async def _fetch_episode(turn_id: str):
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        return (
            await s.execute(select(MemoryEpisode).where(MemoryEpisode.turn_id == turn_id))
        ).scalars().one()


async def test_created_at_persisted(db):
    """传 created_at → 原样落库（28 天前的对话就是 28 天前，不写成现在）。"""
    from core.memory.episodic import record_episode

    await _ensure_user("u-ca1")
    t = time.time() - 28 * 86400
    ok = await record_episode(
        user_id="u-ca1", course_id="c", session_id="s", turn_id="t-inject",
        user_msg="问", assistant_msg="答", created_at=t,
    )
    assert ok is True
    row = await _fetch_episode("t-inject")
    assert row.created_at == t


async def test_default_wall_clock_when_omitted(db):
    """不传 created_at → 列默认墙钟（线上行为不变，缝是可选的）。"""
    from core.memory.episodic import record_episode

    await _ensure_user("u-ca2")
    before = time.time()
    ok = await record_episode(
        user_id="u-ca2", course_id="c", session_id="s", turn_id="t-default",
        user_msg="问", assistant_msg="答",
    )
    after = time.time()
    assert ok is True
    row = await _fetch_episode("t-default")
    assert before <= row.created_at <= after


async def test_injected_created_at_idempotent_on_replay(db):
    """(session_id, turn_id) 幂等对拨钟路径同样成立：重放返回 False、不写第二行。"""
    from core.memory.episodic import record_episode
    from core.db.database import MemoryEpisode
    from sqlalchemy import func, select

    await _ensure_user("u-ca3")
    t = time.time() - 3 * 86400
    kwargs = dict(
        user_id="u-ca3", course_id="c", session_id="s", turn_id="t-replay",
        user_msg="问", assistant_msg="答", created_at=t,
    )
    assert await record_episode(**kwargs) is True
    assert await record_episode(**kwargs) is False

    from core.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as s:
        n = (
            await s.execute(
                select(func.count()).select_from(MemoryEpisode).where(
                    MemoryEpisode.turn_id == "t-replay"
                )
            )
        ).scalar_one()
    assert n == 1
