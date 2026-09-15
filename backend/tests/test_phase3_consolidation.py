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


async def test_claim_pending_returns_only_returning_rows_under_race(db):
    """并发修复回归：返回集 = UPDATE RETURNING 命中的行，而非 SELECT 快照。

    SQLite in-memory 单连接（StaticPool）无法真并发，故拦截 UPDATE 语句伪造 RETURNING
    只命中 t1（模拟 t2 在 SELECT 后、UPDATE 前被别的 worker 抢走、status='pending' 不再
    命中）。修复前 ``return rows`` 会返回 SELECT 快照 [t1,t2]；修复后只返回 RETURNING
    命中的 [t1]。
    """
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.dml import Update
    from unittest.mock import MagicMock

    await _seed([
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
                      user_msg="q1", assistant_msg="a1", status="pending"),
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t2",
                      user_msg="q2", assistant_msg="a2", status="pending"),
    ])
    async with AsyncSessionLocal() as s:
        all_rows = (await s.execute(
            select(MemoryEpisode).where(MemoryEpisode.user_id == "u1")
        )).scalars().all()
    t1_id = next(r.id for r in all_rows if r.turn_id == "t1")

    real_execute = AsyncSession.execute

    async def fake_execute(self, stmt, *a, **kw):
        if isinstance(stmt, Update):
            # 伪造 RETURNING 只命中 t1（t2 被并发抢走）
            result = MagicMock()
            result.scalars.return_value.all.return_value = [t1_id]
            return result
        return await real_execute(self, stmt, *a, **kw)

    with patch.object(AsyncSession, "execute", fake_execute):
        async with AsyncSessionLocal() as s:
            claimed = await consolidation.claim_pending(s, "u1", "c1")

    assert sorted(ep.turn_id for ep in claimed) == ["t1"], \
        "应只返回 RETURNING 命中的 t1，而非 SELECT 快照 [t1,t2]"


# ── consolidate（DB + mocked mem0）──────────────────────────────────────────


async def test_consolidate_promotes_and_marks_done(db):
    from core.db.database import AsyncSessionLocal, CourseTopic, MemoryEpisode
    from sqlalchemy import select

    await _seed([
        CourseTopic(course_id="c1", topic_id="deriv", label="导数"),
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
                      user_msg="什么是导数", assistant_msg="变化率", status="pending"),
    ])
    with (
        patch("core.memory.mem0_client.get_memory") as gm,
        # 段级切分后 add 次数取决于主题段：mock 无命中 → 不写 event，只验 done 标记
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[])),
    ):
        gm.return_value.add = AsyncMock(return_value={"results": []})
        async with AsyncSessionLocal() as s:
            result = await consolidation.consolidate(s, "u1", "c1")

    assert result == {"claimed": 1, "promoted": 1}
    async with AsyncSessionLocal() as s:
        rows = (
            (await s.execute(select(MemoryEpisode).where(MemoryEpisode.user_id == "u1")))
            .scalars()
            .all()
        )
    assert all(r.status == "done" for r in rows)
    assert all(r.consolidated_at is not None for r in rows)
    async with AsyncSessionLocal() as s:
        rows = (
            (await s.execute(select(MemoryEpisode).where(MemoryEpisode.user_id == "u1")))
            .scalars()
            .all()
        )
    assert all(r.status == "done" for r in rows)
    assert all(r.consolidated_at is not None for r in rows)


async def test_consolidate_groups_multiple_sessions(db):
    """两个 session → 两条 event（每 session 各一个主题段一条）。

    掌握度判定 mock 为 None（判定不可用 → episode 回 pending 等下轮）：本用例只验
    mem0 分组口径，promoted 数按判定可用计 0（阶段C：promoted=判定可用段）。
    """
    from core.db.database import AsyncSessionLocal, CourseTopic, MemoryEpisode

    await _seed([
        CourseTopic(course_id="c1", topic_id="ohm", label="欧姆定律"),
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
                      user_msg="q", assistant_msg="a", status="pending"),
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s2", turn_id="t2",
                      user_msg="q", assistant_msg="a", status="pending"),
    ])
    with (
        patch("core.memory.mem0_client.get_memory") as gm,
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=None)),
    ):
        gm.return_value.add = AsyncMock(return_value={"results": []})
        async with AsyncSessionLocal() as s:
            result = await consolidation.consolidate(s, "u1", "c1")
    assert result["claimed"] == 2
    assert result["promoted"] == 0  # 判定不可用 → 回 pending；mem0 分组口径见 events
    events = [
        c for c in gm.return_value.add.call_args_list
        if (c.kwargs.get("metadata") or {}).get("kind") == "event"
    ]
    assert len(events) == 2  # 每 session 一段 → 一条 event（mem0 照常升格）


async def test_consolidate_mem0_failure_degrades_not_retry(db):
    """mem0.add 抛异常 → 降级 warning（阶段2）：episode 照 done、不回 pending。

    旧行为（整段回 pending）已废止：mem0 失败会连带丢掉本可写入的掌握度观测。"""
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    await _seed([
        MemoryEpisode(user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
                      user_msg="q", assistant_msg="a", status="pending"),
    ])
    with (
        patch("core.memory.mem0_client.get_memory") as gm,
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[])),
    ):
        gm.return_value.add = AsyncMock(side_effect=RuntimeError("mem0 down"))
        async with AsyncSessionLocal() as s:
            result = await consolidation.consolidate(s, "u1", "c1")
    assert result == {"claimed": 1, "promoted": 1}
    async with AsyncSessionLocal() as s:
        rows = (
            (await s.execute(select(MemoryEpisode).where(MemoryEpisode.user_id == "u1")))
            .scalars()
            .all()
        )
    assert all(r.status == "done" for r in rows)  # 降级不回 pending（mastery 照写）


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

    with patch("core.memory.redis_pool.get_redis", return_value=fake_redis):
        val = await consolidation.add_importance("u1", "c1", 0.4)

    assert val == 0.9
    pipe.incrbyfloat.assert_called_once_with("mem_importance:u1:c1", 0.4)
    pipe.expire.assert_called_once_with("mem_importance:u1:c1", 86400)


# ── cron 兜底窗（worker.cron_consolidate_memory）────────────────────────────


async def test_cron_consolidate_stale_window(db):
    """兜底窗回归：pending 挂 5min（<1800s）不捞；超 30min 才兜底 enqueue。

    旧窗 300s 让任何 pending 5min 内必被捞走，阈值形同虚设（每轮问答都进 mem0）。
    另验崩溃遗留：processing 超 _PROCESSING_TIMEOUT_SECONDS（600s）回 pending。
    """
    import time

    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select
    from worker import cron_consolidate_memory

    now = time.time()
    await _seed([
        # 5min 前的 pending：阈值路径负责，cron 不捞
        MemoryEpisode(user_id="u_fresh", course_id="c1", session_id="s", turn_id="t1",
                      user_msg="q", assistant_msg="a", status="pending", created_at=now - 300),
        # 31min 前的 pending：超兜底窗，cron 捞走
        MemoryEpisode(user_id="u_stale", course_id="c1", session_id="s", turn_id="t2",
                      user_msg="q", assistant_msg="a", status="pending", created_at=now - 1900),
        # 崩溃遗留 processing（11min 前 > 600s 超时）：回 pending；本轮不 enqueue
        #（回 pending 后 created_at 仍只有 11min，未到 30min 窗）
        MemoryEpisode(user_id="u_orphan", course_id="c1", session_id="s", turn_id="t3",
                      user_msg="q", assistant_msg="a", status="processing", created_at=now - 700),
    ])
    pool = MagicMock()
    pool.enqueue_job = AsyncMock()
    with patch("core.arq_pool.get_arq_pool", AsyncMock(return_value=pool)):
        await cron_consolidate_memory(ctx=None)

    pool.enqueue_job.assert_awaited_once_with(
        "consolidate_memory", user_id="u_stale", course_id="c1"
    )
    async with AsyncSessionLocal() as s:
        orphan = (await s.execute(
            select(MemoryEpisode).where(MemoryEpisode.user_id == "u_orphan")
        )).scalar_one()
        fresh = (await s.execute(
            select(MemoryEpisode).where(MemoryEpisode.user_id == "u_fresh")
        )).scalar_one()
    assert orphan.status == "pending"   # 崩溃遗留被翻回
    assert fresh.status == "pending"    # 窗内 pending 原样不动
