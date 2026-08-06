"""L2 摘要架构加固回归：COUNT 短路 / keyset 增量 / 游标兼容 / OCC 冲突 / 跨进程锁。

对应 session_summary.py 的三个缺陷修复：
  1) 读放大——每轮全量拉消息 → COUNT 短路（test_count_short_circuit_skips_compress）；
  2) 游标不可范围查询 → keyset (created_at, id) 增量（test_keyset_incremental_*）+
     存量 NULL 游标兼容路径（test_legacy_cursor_compat_*）；
  3) 跨进程竞态 + lost update → Redis per-session 锁（test_cross_process_redis_lock_*）
     + OCC 条件 UPDATE（test_occ_conflict_*）。

COUNT/keyset/兼容用真实 in-memory SQLite（client fixture 触发 init_db 建表），
_maybe_compress_locked 跑真 SQL；OCC 冲突与跨进程锁用 mock（确定性）。
不依赖真实 LLM：_do_compress 全程 patch。
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.db.database import AsyncSessionLocal, Message, Session  # noqa: E402
from core.memory import session_summary as ss  # noqa: E402

# window_size=3 → window_msg_count=6；10 条消息时 boundary=m03、压缩区间 m00..m03
_WINDOW = 3
_BUFFER = 0


def _msgs(n: int):
    """n 条消息 (id m00..m{n-1}，created_at=1000+i)，role 交替。"""
    return [(f"m{i:02d}", "user" if i % 2 == 0 else "assistant", f"c{i}", 1000.0 + i) for i in range(n)]


async def _seed(sid, msg_count, *, summary="", up_to_msg_id=None, up_to_ca=None, version=0):
    """插入 session（含 L2 状态）+ msg_count 条消息，返回 db。调用方负责关闭。"""
    db = AsyncSessionLocal()
    db.add(Session(
        id=sid, course_id="c1", user_id="u1", title="t", summary=summary,
        summary_up_to_msg_id=up_to_msg_id, summary_up_to_created_at=up_to_ca,
        summary_version=version,
    ))
    for mid, role, content, ca in _msgs(msg_count):
        db.add(Message(id=mid, session_id=sid, role=role, content=content, created_at=ca))
    await db.commit()
    return db


async def _read_l2(db, sid):
    """读回 session 当前的 L2 列（Core UPDATE 不改 ORM 对象，须重查）。"""
    return (await db.execute(
        select(Session.summary_version, Session.summary_up_to_msg_id, Session.summary_up_to_created_at)
        .where(Session.id == sid)
    )).first()


# ── COUNT 短路：消息不足阈值时不压缩、不取消息 ──────────────────────────────

@pytest.mark.asyncio
async def test_count_short_circuit_skips_compress(client):
    """消息数 ≤ window+buffer → COUNT 守卫短路，_do_compress 不被调用。"""
    sid = "sess-count"
    db = await _seed(sid, msg_count=3)  # 3 ≤ window+buffer(3) → 不压
    try:
        mgr = ss.SessionSummaryManager(window_size=_WINDOW, buffer_size=_BUFFER, compress_interval=1)
        with patch.object(mgr, "_do_compress", new=AsyncMock(return_value="x")) as mocked:
            ret = await mgr.maybe_compress(db, sid)
        assert ret is False
        mocked.assert_not_called()  # 没走到 LLM
    finally:
        await db.close()


# ── keyset 增量：只把游标之后的消息喂给 LLM，游标前移到 boundary ─────────────

@pytest.mark.asyncio
async def test_keyset_incremental_only_feeds_post_cursor(client):
    """有 (created_at, id) 游标 → 只压缩游标之后到 boundary 的增量，游标前移、版本+1。"""
    sid = "sess-keyset"
    # 游标停在 m01（新游标路径：summary_up_to_created_at 已回填）
    db = await _seed(sid, msg_count=10, summary="已有摘要",
                     up_to_msg_id="m01", up_to_ca=1001.0, version=0)
    try:
        mgr = ss.SessionSummaryManager(window_size=_WINDOW, buffer_size=_BUFFER, compress_interval=1)
        seen: list[str] = []

        async def record(existing, msgs):
            seen.extend(m.id for m in msgs)
            return '{"topics":["t"]}'

        with patch.object(mgr, "_do_compress", side_effect=record):
            ret = await mgr.maybe_compress(db, sid)

        assert ret is True
        assert seen == ["m02", "m03"]  # (m01, m03]：游标后到 boundary 的增量
        row = await _read_l2(db, sid)
        assert row.summary_up_to_msg_id == "m03"          # 游标前移到 boundary
        assert row.summary_up_to_created_at == 1003.0     # 新游标时间分量落库
        assert row.summary_version == 1                    # OCC 版本号 +1
    finally:
        await db.close()


# ── 游标兼容：NULL created_at 走 msg_id 兼容路径，成功后写回新游标 ───────────

@pytest.mark.asyncio
async def test_legacy_cursor_compat_writes_new_cursor(client):
    """存量行 summary_up_to_created_at=NULL → 按 msg_id 查 created_at 走兼容路径，
    压缩后写回新的 (msg_id, created_at) 游标，此后走 SQL 路径。"""
    sid = "sess-compat"
    db = await _seed(sid, msg_count=10, summary="已有摘要",
                     up_to_msg_id="m01", up_to_ca=None, version=0)  # NULL：legacy
    try:
        mgr = ss.SessionSummaryManager(window_size=_WINDOW, buffer_size=_BUFFER, compress_interval=1)
        seen: list[str] = []

        async def record(existing, msgs):
            seen.extend(m.id for m in msgs)
            return '{"topics":["t"]}'

        with patch.object(mgr, "_do_compress", side_effect=record):
            ret = await mgr.maybe_compress(db, sid)

        assert ret is True
        assert seen == ["m02", "m03"]  # 兼容路径解析 m01.created_at 后等价的增量区间
        row = await _read_l2(db, sid)
        assert row.summary_up_to_created_at == 1003.0  # 兼容一次性回填新游标
        assert row.summary_up_to_msg_id == "m03"
        assert row.summary_version == 1
    finally:
        await db.close()


# ── OCC 冲突：UPDATE rowcount=0 → 放弃本轮、rollback、不覆盖 ────────────────

@pytest.mark.asyncio
async def test_occ_conflict_returns_false_and_rollback():
    """条件 UPDATE 命中 0 行（别的 worker 已先写、版本号不匹配）→ 返回 False、
    rollback、不 commit、不覆盖 summary。"""
    mgr = ss.SessionSummaryManager(window_size=_WINDOW, buffer_size=_BUFFER, compress_interval=1)

    session = MagicMock()
    session.summary = "旧摘要"
    session.summary_up_to_msg_id = None
    session.summary_up_to_created_at = None
    session.summary_version = 0

    db = AsyncMock()
    db.get = AsyncMock(return_value=session)
    db.scalar = AsyncMock(return_value=22)  # COUNT 过阈值
    result_mock = MagicMock()
    boundary = MagicMock()
    boundary.id = "m03"
    boundary.created_at = 1003.0
    result_mock.scalars.return_value.first.return_value = boundary
    result_mock.scalars.return_value.all.return_value = [MagicMock(id="m02"), MagicMock(id="m03")]
    result_mock.rowcount = 0  # OCC 冲突：UPDATE 命中 0 行
    db.execute = AsyncMock(return_value=result_mock)

    with patch.object(mgr, "_do_compress", return_value="新摘要"):
        ret = await mgr.maybe_compress(db, "sess-occ")

    assert ret is False
    db.rollback.assert_awaited_once()  # 冲突放弃 → rollback
    db.commit.assert_not_awaited()      # 不覆盖、不提交


# ── 跨进程锁：两 worker（独立 asyncio.Lock）+ 共享 Redis → 只一个进入压缩 ────

class _FakeRedis:
    """最小 async Redis mock：支持 SET NX EX + DELETE，记录抢锁次数。"""
    def __init__(self):
        self.store: dict[str, str] = {}
        self.set_calls = 0

    async def set(self, key, val, ex=None, nx=False):
        self.set_calls += 1
        if nx and key in self.store:
            return False
        self.store[key] = val
        return True

    async def delete(self, key):
        self.store.pop(key, None)
        return 1


@pytest.mark.asyncio
async def test_cross_process_redis_lock_only_one_proceeds():
    """两个 manager（各自独立 asyncio.Lock，模拟两 worker）+ 共享 fake Redis，
    同 session 并发 → 只有一个进入 _maybe_compress_locked，另一个抢锁失败让路。"""
    fake = _FakeRedis()
    mgr_a = ss.SessionSummaryManager(window_size=_WINDOW, buffer_size=_BUFFER, compress_interval=1)
    mgr_b = ss.SessionSummaryManager(window_size=_WINDOW, buffer_size=_BUFFER, compress_interval=1)
    entered: list[str] = []

    async def slow_locked(self, db, sid):
        entered.append(sid)
        await asyncio.sleep(0.1)  # 让另一个 worker 有窗口抢 Redis 锁
        return True

    # 每 worker 独立 asyncio.Lock（模拟两进程，L1 不互串），Redis 锁是唯一仲裁点
    def fresh_lock(sid):
        return asyncio.Lock()

    with patch.object(ss, "_get_compress_lock", side_effect=fresh_lock), \
         patch.object(ss, "_get_l2_redis", return_value=fake), \
         patch.object(ss.SessionSummaryManager, "_maybe_compress_locked", slow_locked):
        results = await asyncio.gather(
            mgr_a.maybe_compress(AsyncMock(), "sess-xproc"),
            mgr_b.maybe_compress(AsyncMock(), "sess-xproc"),
        )

    assert len(entered) == 1              # 只一个 worker 进入压缩
    assert sorted(results) == [False, True]
    assert fake.set_calls == 2            # 两个都尝试抢锁
