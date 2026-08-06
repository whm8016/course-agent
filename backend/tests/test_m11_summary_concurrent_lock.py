"""M-11：Session summary 并发压缩无锁的回归测试。

根因（AUDIT M-11，main.py + session_summary.py）：每次 CAPABILITY_COMPLETE 都
asyncio.create_task(_maybe_compress_summary(session_id))，无 per-session 锁。同一
session 连续多轮 turn → 多个压缩 task 并发：都读到同一份旧 session.summary、各自调
LLM、各自 commit 写 summary_up_to_msg_id → 后者覆盖前者、增量丢失、游标错乱。

修法：session_summary.py 加 per-session asyncio.Lock（L1）。maybe_compress 入口非阻塞
探测（lock.locked()），已有压缩在进行则本轮让路（返回 False，下次 turn 再压）；否则
持锁执行。本文件聚焦 L1 进程内锁的语义——SQL keyset / OCC / 跨进程 Redis 锁的正确性
由 test_l2_summary_hardening.py 用真实 SQLite 覆盖；这里用 mock db 驱动新版
_maybe_compress_locked（COUNT 短路 + boundary + keyset 增量 + OCC 写回）走到 _do_compress。

interleaving（同 session 两个并发 maybe_compress，单事件循环）：
  - 竞态（本测试）：A 拿锁进入 _do_compress（LLM 耗时）→ B 探测 lock.locked()=True →
    B return False（让路，不并发覆盖）；A 完成 commit。最终 _do_compress 仅一次。
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory import session_summary as ss  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_locks():
    """每个测试前后清空 per-session 锁字典，隔离。"""
    ss._compress_locks.clear()
    yield
    ss._compress_locks.clear()


def _make_session_with_messages(msg_count: int, window_size: int = 10):
    """构造一个 mock Session + msg_count 条 Message，使阈值判断进入压缩分支。"""
    session = MagicMock()
    session.summary = ""  # 首次全量
    session.summary_up_to_msg_id = None
    session.summary_up_to_created_at = None  # 新 keyset 游标（首次为 None → 走全量）
    session.summary_updated_at = None
    session.summary_version = 0  # OCC 版本号

    messages = []
    for i in range(msg_count):
        m = MagicMock()
        m.id = f"msg-{i}"
        m.role = "user" if i % 2 == 0 else "assistant"
        m.content = f"content-{i}"
        m.created_at = float(i)
        messages.append(m)

    # window_msg_count = window_size * 2；需 msg_count > window_size*2 且 > window+buffer，
    # 给 22 条（window=10）满足。
    return session, messages


def _make_db(session, messages, scalar_side_effect=None):
    """构造 mock db，驱动新版 _maybe_compress_locked 走完整流程到 _do_compress。

    新实现：db.scalar 做 COUNT 短路；db.execute 取 boundary（.scalars().first()）、
    keyset 增量（.scalars().all()）、OCC 写回（.rowcount）。同一 result_mock 同时装载
    这三种访问方式，使三次 db.execute 都取到正确值。
    """
    db = AsyncMock()
    db.get = AsyncMock(return_value=session)
    if scalar_side_effect is not None:
        db.scalar = AsyncMock(side_effect=scalar_side_effect)
    else:
        db.scalar = AsyncMock(return_value=len(messages))
    result_mock = MagicMock()
    # boundary 取窗口外最后一条；测试只关心非 None + 有 .id/.created_at，取任一消息即可
    result_mock.scalars.return_value.first.return_value = messages[1]
    result_mock.scalars.return_value.all.return_value = messages  # keyset 增量区间
    result_mock.rowcount = 1  # OCC 写回成功
    db.execute = AsyncMock(return_value=result_mock)
    return db


async def test_concurrent_compress_same_session_only_one_runs():
    """同 session 并发 maybe_compress：只有持锁者执行 _do_compress + commit，另一让路。"""
    mgr = ss.SessionSummaryManager(window_size=10, buffer_size=2, compress_interval=1)
    session, messages = _make_session_with_messages(22, window_size=10)
    db = _make_db(session, messages)

    compress_call_count = 0
    compress_started = asyncio.Event()

    async def slow_do_compress(existing, msgs):
        nonlocal compress_call_count
        compress_call_count += 1
        compress_started.set()
        await asyncio.sleep(0.1)  # 让第二个 maybe_compress 探测到锁
        return "compressed-summary"

    with patch.object(mgr, "_do_compress", side_effect=slow_do_compress):
        results = await asyncio.gather(
            mgr.maybe_compress(db, "sess-A"),
            mgr.maybe_compress(db, "sess-A"),
        )

    # M-11 核心：_do_compress 仅被调用一次（第二个让路，未并发覆盖）
    assert compress_call_count == 1, f"expected 1 compress, got {compress_call_count}"
    # 一个 True（执行了），一个 False（让路）
    assert sorted(results) == [False, True]
    # commit 仅一次（在持锁者的临界区内）
    assert db.commit.await_count == 1


async def test_different_sessions_compress_concurrently():
    """不同 session 的锁互相隔离，可并发压缩（不误伤吞吐）。"""
    mgr = ss.SessionSummaryManager(window_size=10, buffer_size=2, compress_interval=1)
    session_a, msgs_a = _make_session_with_messages(22)
    session_b, msgs_b = _make_session_with_messages(22)
    db_a, db_b = _make_db(session_a, msgs_a), _make_db(session_b, msgs_b)

    barrier = asyncio.Event()

    async def gated_do_compress(existing, msgs):
        await barrier.wait()  # 让两个并发压缩同时卡在 _do_compress
        return "summary"

    with patch.object(mgr, "_do_compress", side_effect=gated_do_compress):
        task_a = asyncio.create_task(mgr.maybe_compress(db_a, "sess-A"))
        task_b = asyncio.create_task(mgr.maybe_compress(db_b, "sess-B"))
        await asyncio.sleep(0.05)  # 让两者都进入各自的锁
        barrier.set()  # 放行
        r_a, r_b = await asyncio.gather(task_a, task_b)

    # 不同 session → 各自压缩成功（锁隔离，不互相阻塞）
    assert r_a is True and r_b is True


async def test_compress_after_release_succeeds():
    """前一次压缩完成后，下一次同 session 可正常进入（锁已释放，非永久阻塞）。

    第二次走 COUNT 短路返回 False（scalar 低于阈值），证明锁未卡死、流程可重入。
    新实现写回走 Core UPDATE（不原地改写 mock session 游标），故用 COUNT 短路模拟
    「无需再压」——游标推进 / 无增量的语义由 test_l2_summary_hardening 真实库覆盖。
    """
    mgr = ss.SessionSummaryManager(window_size=10, buffer_size=2, compress_interval=1)
    session, messages = _make_session_with_messages(22)
    # 第一次 scalar=22（进入压缩）；第二次 scalar=5（低于阈值 12 → COUNT 短路）
    db = _make_db(session, messages, scalar_side_effect=[22, 5])

    with patch.object(mgr, "_do_compress", return_value="summary"):
        r1 = await mgr.maybe_compress(db, "sess-C")
        r2 = await mgr.maybe_compress(db, "sess-C")

    assert r1 is True
    assert r2 is False  # COUNT 短路，非锁阻塞
    # 锁在两次调用间已释放
    assert not ss._get_compress_lock("sess-C").locked()
