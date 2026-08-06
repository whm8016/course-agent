"""M-13: 并发 flush 同一 key 必须有 per-key 互斥锁，避免多 worker 重复落盘/竞态。

根因：原 flush 循环无 per-key 锁。两个 worker 的 cron 同时扫到同一 buffer key，
会并发各自 lrange 拿到同一批 turns、各自 flush（重复写 mem0/graph）、各自 delete
（第二个 delete 时 key 可能已被第一个删/或两者交替造成状态混乱）。

修复：_flush_one 用 Redis `SET NX EX` 抢 per-key 锁（lock_key = key:lock），
抢不到即跳过（说明别的 worker 正在处理），flush 完 finally 释放。
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.memory import flush_manager


def _make_lockable_redis(store: dict, lock_state: dict):
    """构造能真实模拟 SET NX 语义的 mock redis。

    `store`: 业务数据（key / :ts / :meta）
    `lock_state`: 模拟分布式锁状态 dict（key -> True）；SET NX 语义基于它
    """
    # lrange/llen 读业务数据
    async def _lrange(k, a, b):
        data = store.get(k, [])
        return list(data)[a : (b + 1 if b != -1 else None)]

    async def _llen(k):
        return len(store.get(k, []))

    async def _get(k):
        return store.get(k)

    async def _scan(cursor=0, match=None, count=100):  # noqa: ARG001
        import fnmatch
        ks = [k for k in store.keys() if match is None or fnmatch.fnmatch(k, match)]
        return (0, ks)

    async def _set(k, v, ex=None, nx=False):  # noqa: ARG002
        # 真实模拟 SET NX：已存在则返回 None/False
        if nx and k in lock_state:
            return False
        lock_state[k] = v
        return True

    async def _delete(*keys):
        n = 0
        for k in keys:
            removed = False
            if k in store:
                del store[k]
                removed = True
            if k in lock_state:
                del lock_state[k]
                removed = True
            if removed:
                n += 1
        return n

    return SimpleNamespace(
        scan=AsyncMock(side_effect=_scan), set=AsyncMock(side_effect=_set),
        get=AsyncMock(side_effect=_get), delete=AsyncMock(side_effect=_delete),
        lrange=AsyncMock(side_effect=_lrange), llen=AsyncMock(side_effect=_llen),
    )


async def test_second_concurrent_flush_skipped_by_lock():
    """M-13: 两个 worker 并发 flush 同一 key，第二个被锁挡住（不重复 flush）。"""
    key = "mem_flush:u1:s1"
    store = {
        key: ['{"u": "q", "a": "ans"}'] * 3,
        f"{key}:ts": "0",
        f"{key}:meta": '{"user_id": "u1", "course_id": "c1"}',
    }
    lock_state: dict = {}
    r = _make_lockable_redis(store, lock_state)

    flush_calls = 0

    async def _slow_flush(*a, **kw):  # noqa: ARG001
        nonlocal flush_calls
        flush_calls += 1
        # 模拟 flush 耗时，让并发窗口重叠
        await asyncio.sleep(0.05)

    with patch.object(flush_manager, "_flush_turns", AsyncMock(side_effect=_slow_flush)):
        # 两个 worker 同时 flush 同一 key
        meta = {"user_id": "u1", "course_id": "c1"}
        turns = [{"u": "q", "a": "ans"}] * 3
        await asyncio.gather(
            flush_manager._flush_one(r, key, turns, meta),
            flush_manager._flush_one(r, key, turns, meta),
        )

    # 关键断言：_flush_turns 只应被调用一次（第二个被锁挡住）
    assert flush_calls == 1, (
        f"并发 flush 同一 key 应互斥，_flush_turns 只调 1 次，实际 {flush_calls} 次 → 重复落盘/竞态"
    )
    # 锁在 _flush_one 的 finally 里被释放（不残留）
    assert f"{key}:lock" not in lock_state, "锁未释放 → 残留会阻塞后续正常 flush"


async def test_lock_released_after_flush_so_next_can_acquire():
    """M-13: flush 完释放锁后，下一次对同一 key 的 flush 能正常抢到锁。"""
    key = "mem_flush:u1:s1"
    store = {
        key: ['{"u": "q", "a": "ans"}'],
        f"{key}:ts": "0",
        f"{key}:meta": '{"user_id": "u1", "course_id": "c1"}',
    }
    lock_state: dict = {}
    r = _make_lockable_redis(store, lock_state)

    # Phase 1：_flush_turns 真实调用会触 mem0（测试环境无 mem0 模块 → 返回 False），
    # 但本用例只验「锁释放后可再次抢锁」，故隔离 _flush_turns 为成功（True）。
    with patch.object(flush_manager, "_flush_turns", AsyncMock(return_value=True)):
        done1 = await flush_manager._flush_one(
            r, key, [{"u": "q", "a": "ans"}], {"user_id": "u1", "course_id": "c1"}
        )
        assert done1 is True

        # 第一次 flush 已删 key 且释放锁；重新塞回数据模拟"又有新对话进来"
        store[key] = ['{"u": "q2", "a": "a2"}']
        store[f"{key}:ts"] = "0"
        store[f"{key}:meta"] = '{"user_id": "u1", "course_id": "c1"}'

        done2 = await flush_manager._flush_one(
            r, key, [{"u": "q2", "a": "a2"}], {"user_id": "u1", "course_id": "c1"}
        )
        assert done2 is True, "锁已释放，第二次应能正常抢锁 flush（不能被残留锁永久阻塞）"
        assert f"{key}:lock" not in lock_state
