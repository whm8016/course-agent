"""H-7: flush 先写后删（原子性）—— flush 失败时 Redis key 必须保留，数据不丢。

根因：原 scan_and_flush 第 203 行 `await r.delete(...)` 在第 206 行
`await _flush_turns(...)` **之前**。当 mem0/PG 写入失败时，数据已从 Redis 删掉，
对话永久丢失。

修复：抽 _flush_one，顺序为「抢锁 → flush 成功 → 才删 key」，flush 抛异常
则 key 保留，等下次 cron 重试。
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.memory import flush_manager


def _make_redis(store: dict) -> SimpleNamespace:
    """构造内存版 mock redis，状态存在 `store` dict。

    只实现 _flush_one/scan_and_flush 用到的子集：scan/set/get/delete/lrange/llen。
    """

    async def _scan(cursor=0, match=None, count=100):  # noqa: ARG001
        # 单页返回所有匹配 key（SCAN 分页由 H-8 单测覆盖）
        ks = [k for k in store.keys() if match is None or _match(match, k)]
        return (0, ks)

    async def _set(k, v, ex=None, nx=False):  # noqa: ARG002
        if nx and k in store:
            return False
        store[k] = v
        return True

    async def _get(k):
        return store.get(k)

    async def _delete(*keys):
        n = 0
        for k in keys:
            if k in store:
                del store[k]
                n += 1
        return n

    async def _lrange(k, a, b):
        return list(store.get(k, []))[a : (b + 1 if b != -1 else None)]

    async def _llen(k):
        return len(store.get(k, []))

    async def _ltrim(k, start, stop):
        data = list(store.get(k, []))
        if stop == -1:
            stop = len(data) - 1
        store[k] = data[start:stop + 1]
        if not store[k]:  # Redis 删空 list key
            del store[k]

    return SimpleNamespace(scan=AsyncMock(side_effect=_scan), set=AsyncMock(side_effect=_set),
                           get=AsyncMock(side_effect=_get), delete=AsyncMock(side_effect=_delete),
                           lrange=AsyncMock(side_effect=_lrange), llen=AsyncMock(side_effect=_llen),
                           ltrim=AsyncMock(side_effect=_ltrim))


def _match(pattern: str, key: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(key, pattern)


@pytest.fixture
def buffer_store():
    """单个 buffer key 的初始状态：3 轮对话 + meta + ts。"""
    key = "mem_flush:u1:s1"
    turn = '{"u": "hi", "a": "hello"}'
    return {
        key: [turn, turn, turn],          # 3 轮
        f"{key}:ts": "0",                 # idle 极大，触发 flush
        f"{key}:meta": '{"user_id": "u1", "course_id": "c1"}',
    }


async def test_flush_failure_keeps_redis_key(buffer_store):
    """H-7: flush 抛异常时，数据 key 必须仍在 Redis（不丢）。"""
    r = _make_redis(buffer_store)
    data_key = "mem_flush:u1:s1"

    # flush 抛异常（模拟 mem0/PG 写失败）
    async def _boom(*a, **kw):  # noqa: ARG001
        raise RuntimeError("mem0 down")

    with patch.object(flush_manager, "_flush_turns", AsyncMock(side_effect=_boom)):
        n = await flush_manager.scan_and_flush(r, max_turns=1, idle_timeout=1.0)

    # flush 失败：不计入成功数
    assert n == 0
    # 数据 key 仍在（含 :ts / :meta）——数据完整保留待下次重试
    assert data_key in buffer_store, "数据 key 被删 → 永久丢数据（H-7 回归）"
    assert len(buffer_store[data_key]) == 3, "3 轮对话必须原样保留"
    assert f"{data_key}:ts" in buffer_store
    assert f"{data_key}:meta" in buffer_store


async def test_flush_success_deletes_redis_key(buffer_store):
    """H-7: flush 成功时，数据 key 才被删除（正常路径）。"""
    r = _make_redis(buffer_store)
    data_key = "mem_flush:u1:s1"

    # Phase 1：_flush_turns 返回 bool——True 表示 mem0 关键写成功（可删 key）。
    with patch.object(flush_manager, "_flush_turns", AsyncMock(return_value=True)):
        n = await flush_manager.scan_and_flush(r, max_turns=1, idle_timeout=1.0)

    assert n == 1
    # 成功后数据 key 及其 :ts/:meta 全部清除
    assert data_key not in buffer_store
    assert f"{data_key}:ts" not in buffer_store
    assert f"{data_key}:meta" not in buffer_store


async def test_flush_preserves_turns_enqueued_during_flush(buffer_store):
    """P1 数据丢失修复：flush 期间新 enqueue 的 turn 不被连带删除。

    原代码锁外 lrange 读快照，flush 后 ``delete(key)`` 把 flush 期间 RPUSH 的新 turn 一起
    删 -> 永久丢数据（单 worker 也会发生）。修复后锁内读快照、LTRIM 只裁已 flush 前缀，
    新 turn 保留待下次 flush。
    """
    r = _make_redis(buffer_store)
    data_key = "mem_flush:u1:s1"
    new_turn = '{"u": "q2", "a": "a2"}'

    async def _flush_and_enqueue(*a, **kw):  # noqa: ARG001
        # 模拟 flush 期间用户又发一轮 -> RPUSH 进 list
        buffer_store[data_key].append(new_turn)
        return True

    with patch.object(flush_manager, "_flush_turns", AsyncMock(side_effect=_flush_and_enqueue)):
        flushed = await flush_manager._flush_one(r, data_key)

    assert flushed == 3  # 快照是初始 3 轮
    # 关键：flush 期间新增的 turn 保留在 list（旧 delete 会连同删掉 -> 丢数据）
    assert buffer_store[data_key] == [new_turn], \
        "flush 期间新 enqueue 的 turn 被连带删除 -> 数据丢失回归"
    # list 非空 -> :ts/:meta 保留
    assert f"{data_key}:ts" in buffer_store
    assert f"{data_key}:meta" in buffer_store
