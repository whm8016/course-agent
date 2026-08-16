"""索引分布式锁（acquire_index_dlock）短租约改造的回归测试。

覆盖：
- 空锁直接拿到，release 摘牌 + 取消续约
- 被占（活人持有）→ 返回 (None, None) 跳过，绝不动别人的锁（无偷锁设计）
- 默认租约 60s，acquire 即写满 TTL
- 续约走 reacquire()（重置回 ttl，防 extend 累加滚雪球），节奏 ttl/3=20s
- 丢锁（LockNotOwnedError）→ ERROR 留痕后续约循环退出，不再空转

FakeRedis 手写 redis-py asyncio.Lock 的最小镜像（对齐 test_leader.py 手写 mock 的做法）。
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from redis.exceptions import LockNotOwnedError

from core.rag.lightrag.instance_pool import (
    _INDEX_DLOCK_PREFIX,
    _INDEX_DLOCK_TTL,
    acquire_index_dlock,
    release_index_dlock,
)

_KEY = f"{_INDEX_DLOCK_PREFIX}c1:lightrag"


class FakeLock:
    """redis-py asyncio.Lock 的最小镜像：token 所有权 + reacquire 重置 TTL。"""

    def __init__(self, store: dict, name: str, timeout: float):
        self._store = store
        self.name = name
        self.timeout = timeout
        self.token = uuid.uuid4().hex
        self.reacquire_calls = 0
        self.extend_calls = 0

    async def acquire(self, blocking: bool = False) -> bool:  # noqa: ARG002
        if self.name not in self._store:
            self._store[self.name] = {"value": self.token, "pttl": self.timeout * 1000}
            return True
        return False

    async def reacquire(self) -> bool:
        self.reacquire_calls += 1
        ent = self._store.get(self.name)
        if not ent or ent["value"] != self.token:
            raise LockNotOwnedError("not owned")
        ent["pttl"] = self.timeout * 1000  # 重置回 timeout（不是累加）
        return True

    async def extend(self, additional_time: float) -> bool:  # noqa: ARG002
        # 只当 spy 用：断言生产代码不会退回 extend 累加续约（会滚雪球）
        self.extend_calls += 1
        return True

    async def release(self) -> bool:
        ent = self._store.get(self.name)
        if ent and ent["value"] == self.token:
            del self._store[self.name]
            return True
        return False


class FakeRedis:
    """最小 Redis 模拟：lock() 工厂 + 共享 store。"""

    def __init__(self):
        self.store: dict[str, dict] = {}

    def lock(self, name: str, timeout: float = None) -> FakeLock:  # noqa: ARG002
        return FakeLock(self.store, name, timeout)


@pytest.fixture
def fake_redis(monkeypatch):
    """monkeypatch core.db.cache._get_pool 返回 FakeRedis（锁实现的延迟导入处）。"""
    fake = FakeRedis()
    monkeypatch.setattr("core.db.cache._get_pool", lambda: fake)
    return fake


async def test_free_lock_acquired_with_full_default_lease(fake_redis):
    """空锁直接拿到：默认租约 60s，acquire 即写满 TTL；release 摘牌并取消续约。"""
    assert _INDEX_DLOCK_TTL == 60  # 钉死租约数字：崩溃后别人最多等 60s

    lock, renew = await acquire_index_dlock("c1", "lightrag")
    assert lock is not None and renew is not None
    assert fake_redis.store[_KEY]["value"] == lock.token
    assert fake_redis.store[_KEY]["pttl"] == 60_000

    await release_index_dlock(lock, renew)
    assert _KEY not in fake_redis.store
    assert renew.cancelled() or renew.done()


async def test_held_lock_returns_none_and_never_touches_it(fake_redis):
    """被占（无论活人还是刚死残留）→ (None, None) 跳过；短租约下死锁 ≤60s 自灭，无需偷。"""
    fake_redis.store[_KEY] = {"value": "other-worker-token", "pttl": 55_000}
    lock, renew = await acquire_index_dlock("c1", "lightrag")
    assert lock is None and renew is None
    assert fake_redis.store[_KEY] == {"value": "other-worker-token", "pttl": 55_000}  # 纹丝不动


async def test_renew_uses_reacquire_at_ttl_third_interval(fake_redis, monkeypatch):
    """续约走 reacquire()（TTL 重置不滚雪球）且不碰 extend；节奏 = ttl/3（60s 租约 → 20s）。"""
    real_sleep = asyncio.sleep
    intervals: list[float] = []
    calls = 0

    async def fast_sleep(t):
        nonlocal calls
        calls += 1
        intervals.append(t)
        if calls >= 3:
            await real_sleep(30)  # 跑满 3 轮后停车等 release 的 cancel，消除忙转
        await real_sleep(0)  # 立即让出控制权，推进 renew 循环

    monkeypatch.setattr("core.rag.lightrag.instance_pool.asyncio.sleep", fast_sleep)

    lock, renew = await acquire_index_dlock("c1", "lightrag")
    await real_sleep(0.02)  # 让 renew 循环跑若干轮（快进，不真等 20s）
    assert lock.reacquire_calls >= 1
    assert lock.extend_calls == 0
    assert fake_redis.store[_KEY]["pttl"] == 60_000  # 重置回租约值，不是累加后的 80s/100s/...
    await release_index_dlock(lock, renew)
    assert intervals and all(t == 20 for t in intervals)


async def test_renew_exits_when_lock_lost(fake_redis, monkeypatch):
    """丢锁（Redis 断超租约被别人抢走）→ LockNotOwnedError → ERROR 留痕并退出循环。"""
    real_sleep = asyncio.sleep

    async def fast_sleep(t):  # noqa: ARG001
        await real_sleep(0)

    monkeypatch.setattr("core.rag.lightrag.instance_pool.asyncio.sleep", fast_sleep)

    lock, renew = await acquire_index_dlock("c1", "lightrag")
    fake_redis.store[_KEY]["value"] = "someone-else"  # 锁被换走（租约过期被抢）
    await real_sleep(0.02)
    assert renew.done()  # 续约循环退出，不再空转
    assert lock.reacquire_calls == 1
    await release_index_dlock(lock, renew)  # release 幂等：锁不归自己时静默
