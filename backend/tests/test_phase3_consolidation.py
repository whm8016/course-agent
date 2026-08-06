"""Phase 3: L3 巩固——claim pending → 按 session 分组 → mem0 升格 → 标 done/retry。

覆盖：分组、claim 并发安全、promote 成功标 done / 失败回 pending、importance 累计触发。
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory import consolidation


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def _seed(episodes) -> None:
    from core.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as s:
        for ep in episodes:
            s.add(ep)
        await s.commit()


# ── group_episodes_by_session（纯函数）──────────────────────────────────────


def test_group_by_session_splits_by_session_id():
    eps = [
        SimpleNamespace(session_id="s1", id="1"),
        SimpleNamespace(session_id="s1", id="2"),
        SimpleNamespace(session_id="s2", id="3"),
        SimpleNamespace(session_id="", id="4"),
    ]
    groups = consolidation.group_episodes_by_session(eps)
    assert set(groups) == {"s1", "s2", "_no_session"}
    assert [e.id for e in groups["s1"]] == ["1", "2"]


# ── claim_pending（DB）──────────────────────────────────────────────────────


async def test_claim_pending_marks_processing_and_scopes_user(db):
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    await _seed([
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
                      user_msg="q1", assistant_msg="a1", status="pending"),
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t2",
                      user_msg="q2", assistant_msg="a2", status="pending"),
        MemoryEpisode(user_id="u2", course_id="c1", session_id="s1", turn_id="t3",
                      user_msg="q3", assistant_msg="a3", status="pending"),
    ])
    async with AsyncSessionLocal() as s:
        claimed = await consolidation.claim_pending(s, "u1", "c1")
        assert len(claimed) == 2  # 只领 u1
        processing = (
            await s.execute(
                select(MemoryEpisode).where(
                    MemoryEpisode.user_id == "u1", MemoryEpisode.status == "processing"
                )
            )
        ).scalars().all()
    assert len(processing) == 2


async def test_claim_pending_concurrent_safe(db):
    """连续 claim 两次：第二次领不到（已被标 processing，条件 UPDATE rowcount=0）。"""
    from core.db.database import AsyncSessionLocal, MemoryEpisode

    await _seed([
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
                      user_msg="q", assistant_msg="a", status="pending"),
    ])
    async with AsyncSessionLocal() as s:
        first = await consolidation.claim_pending(s, "u1", "c1")
        second = await consolidation.claim_pending(s, "u1", "c1")
    assert len(first) == 1
    assert second == []


async def test_claim_pending_course_scoped(db):
    from core.db.database import AsyncSessionLocal, MemoryEpisode

    await _seed([
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s", turn_id="t1",
                      user_msg="q", assistant_msg="a", status="pending"),
        MemoryEpisode(user_id="u1", course_id="c2", session_id="s", turn_id="t2",
                      user_msg="q", assistant_msg="a", status="pending"),
    ])
    async with AsyncSessionLocal() as s:
        claimed = await consolidation.claim_pending(s, "u1", "c1")
    assert len(claimed) == 1 and claimed[0].course_id == "c1"


# ── consolidate（DB + mocked mem0）──────────────────────────────────────────


async def test_consolidate_promotes_and_marks_done(db):
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    await _seed([
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
                      user_msg="什么是导数", assistant_msg="变化率", status="pending"),
    ])
    with patch("core.memory.mem0_client.get_memory") as gm:
        gm.return_value.add = AsyncMock(return_value={"results": []})
        async with AsyncSessionLocal() as s:
            result = await consolidation.consolidate(s, "u1", "c1")

    assert result == {"claimed": 1, "promoted": 1}
    gm.return_value.add.assert_awaited_once()  # 一个 session → 一次 mem0.add
    async with AsyncSessionLocal() as s:
        rows = (
            (await s.execute(select(MemoryEpisode).where(MemoryEpisode.user_id == "u1")))
            .scalars()
            .all()
        )
    assert all(r.status == "done" for r in rows)
    assert all(r.consolidated_at is not None for r in rows)


async def test_consolidate_groups_multiple_sessions(db):
    """两个 session → 两次 mem0.add（每 session 一批）。"""
    from core.db.database import AsyncSessionLocal, MemoryEpisode

    await _seed([
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
                      user_msg="q", assistant_msg="a", status="pending"),
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s2", turn_id="t2",
                      user_msg="q", assistant_msg="a", status="pending"),
    ])
    with patch("core.memory.mem0_client.get_memory") as gm:
        gm.return_value.add = AsyncMock(return_value={"results": []})
        async with AsyncSessionLocal() as s:
            result = await consolidation.consolidate(s, "u1", "c1")
    assert result["promoted"] == 2
    assert gm.return_value.add.await_count == 2


async def test_consolidate_retry_on_mem0_failure(db):
    """mem0.add 抛异常 → episode 回 pending（不丢，等重试），promoted=0。"""
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    await _seed([
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
                      user_msg="q", assistant_msg="a", status="pending"),
    ])
    with patch("core.memory.mem0_client.get_memory") as gm:
        gm.return_value.add = AsyncMock(side_effect=RuntimeError("mem0 down"))
        async with AsyncSessionLocal() as s:
            result = await consolidation.consolidate(s, "u1", "c1")
    assert result == {"claimed": 1, "promoted": 0}
    async with AsyncSessionLocal() as s:
        rows = (
            (await s.execute(select(MemoryEpisode).where(MemoryEpisode.user_id == "u1")))
            .scalars()
            .all()
        )
    assert all(r.status == "pending" for r in rows)  # 回退重试


async def test_consolidate_no_pending_noop(db):
    from core.db.database import AsyncSessionLocal

    with patch("core.memory.mem0_client.get_memory") as gm:
        gm.return_value.add = AsyncMock()
        async with AsyncSessionLocal() as s:
            result = await consolidation.consolidate(s, "u1", "c1")
    assert result == {"claimed": 0, "promoted": 0}
    gm.return_value.add.assert_not_called()


# ── importance 累计（mocked redis）───────────────────────────────────────────


async def test_add_importance_accumulates_via_redis():
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[0.9, True])
    fake_redis = MagicMock()
    fake_redis.pipeline.return_value = pipe

    with patch("core.memory.flush_manager._get_redis", return_value=fake_redis):
        val = await consolidation.add_importance("u1", "c1", 0.4)

    assert val == 0.9
    pipe.incrbyfloat.assert_called_once_with("mem_importance:u1:c1", 0.4)
    pipe.expire.assert_called_once_with("mem_importance:u1:c1", 86400)
