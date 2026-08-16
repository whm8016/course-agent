"""索引分布式锁（acquire_index_dlock）租约化改造的回归测试。

覆盖四态偷锁判定 + 续约行为：
- 活着（PTTL≈ttl）→ 不偷，跳过语义照旧
- 刚死（PTTL 仍高于阈值，如 SIGKILL 后残余 50min）→ 不偷（索引锁是正确性锁，只等自然过期）
- 死透（PTTL ≤ min(120s, ttl/4)）→ Lua 双条件原子偷走 → 重试 acquire 成功
- 持有者续约插在 GET 与 EVAL 之间 → 双条件拦下，不偷
- 续约用 reacquire()（重置 TTL，防 extend 累加滚雪球），节奏 min(ttl/3, 30s)
- 丢锁（LockNotOwnedError）→ ERROR 留痕后续约循环退出，不再空转

FakeRedis 手写实现 _STEAL_SCRIPT 的双条件语义（对齐 test_leader.py 手写 mock 的做法）。
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from redis.exceptions import LockNotOwnedError

from core.rag.lightrag.instance_pool import (
    _INDEX_DLOCK_PREFIX,
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

    async def extend(self, additional_time: float) -> bool:
        # 真实语义是往剩余 TTL 上加（会滚雪球）；测试只断言它不该再被调用
        self.extend_calls += 1
        ent = self._store.get(self.name)
        if not ent or ent["value"] != self.token:
            raise LockNotOwnedError("not owned")
        ent["pttl"] += additional_time * 1000
        return True

    async def release(self) -> bool:
        ent = self._store.get(self.name)
        if ent and ent["value"] == self.token:
            del self._store[self.name]
            return True
        return False


class FakeRedis:
    """最小 Redis 模拟：lock()/get/pttl + eval（Python 镜像 _STEAL_SCRIPT 双条件）。"""

    def __init__(self):
        self.store: dict[str, dict] = {}
        self.eval_calls: list[tuple] = []
        self.on_eval = None  # 测试钩子：eval 执行前触发（模拟 GET 与 EVAL 之间持有者续约）

    def lock(self, name: str, timeout: float = None) -> FakeLock:  # noqa: ARG002
        return FakeLock(self.store, name, timeout)

    async def get(self, key: str):
        ent = self.store.get(key)
        return None if ent is None else ent["value"]

    async def pttl(self, key: str) -> int:
        ent = self.store.get(key)
        return -2 if ent is None else ent["pttl"]

    async def eval(self, script: str, numkeys: int, key: str, value: str, threshold_ms: int):  # noqa: ARG002
        self.eval_calls.append((key, value, threshold_ms))
        if self.on_eval:
            self.on_eval()
        # _STEAL_SCRIPT 的 Python 镜像：值还是观察值 且 PTTL 仍低于阈值 才 DEL
        ent = self.store.get(key)
        if ent and ent["value"] == value and 0 < ent["pttl"] <= threshold_ms:
            del self.store[key]
            return 1
        return 0


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("core.db.cache._get_pool", lambda: fake)
    return fake


async def test_free_lock_acquired_and_released(fake_redis):
    """空锁直接拿到；release 删 key 且取消续约任务。"""
    lock, renew = await acquire_index_dlock("c1", "lightrag", ttl=3600)
    assert lock is not None and renew is not None
    assert fake_redis.store[_KEY]["value"] == lock.token

    await release_index_dlock(lock, renew)
    assert _KEY not in fake_redis.store
    assert renew.cancelled() or renew.done()


async def test_alive_holder_high_pttl_no_steal(fake_redis):
    """活着（PTTL≈ttl，刚续过约）→ 不偷，返回 (None, None)。"""
    fake_redis.store[_KEY] = {"value": "other-worker-token", "pttl": 3_570_000}
    lock, renew = await acquire_index_dlock("c1", "lightrag", ttl=3600)
    assert lock is None and renew is None
    assert fake_redis.store[_KEY]["value"] == "other-worker-token"  # 没动别人的锁
    assert fake_redis.eval_calls == []  # 根本没走到偷锁脚本


async def test_freshly_dead_high_pttl_no_steal(fake_redis):
    """刚死（SIGKILL 后残余仍 >120s 阈值）→ 同样不偷，只能等自然过期。"""
    fake_redis.store[_KEY] = {"value": "dead-worker-token", "pttl": 3_000_000}  # 残余 50min
    lock, renew = await acquire_index_dlock("c1", "lightrag", ttl=3600)
    assert lock is None and renew is None
    assert fake_redis.eval_calls == []
    assert fake_redis.store[_KEY]["value"] == "dead-worker-token"


async def test_dead_stale_low_pttl_stolen_then_acquired(fake_redis):
    """死透（PTTL 低于阈值）→ 双条件偷锁成功 → 重试 acquire 拿到新锁。"""
    fake_redis.store[_KEY] = {"value": "dead-worker-token", "pttl": 30_000}  # 30s < 120s
    lock, renew = await acquire_index_dlock("c1", "lightrag", ttl=3600)
    assert lock is not None
    assert len(fake_redis.eval_calls) == 1
    assert fake_redis.eval_calls[0][1] == "dead-worker-token"  # 用观察值做 CAS，不是自己的新 token
    assert fake_redis.store[_KEY]["value"] == lock.token  # 新持有者
    await release_index_dlock(lock, renew)


async def test_renew_race_between_get_and_eval_no_steal(fake_redis):
    """持有者续约插在 GET 与 EVAL 之间（PTTL 被拨回满）→ 双条件拦下，不偷。"""
    fake_redis.store[_KEY] = {"value": "alive-worker-token", "pttl": 30_000}

    def holder_renews_mid_race():
        fake_redis.store[_KEY]["pttl"] = 3_600_000  # 恰在脚本执行前续上

    fake_redis.on_eval = holder_renews_mid_race
    lock, renew = await acquire_index_dlock("c1", "lightrag", ttl=3600)
    assert lock is None and renew is None  # 偷锁被拦，照旧跳过
    assert len(fake_redis.eval_calls) == 1  # 脚本确实跑了，但双条件判定不删
    assert fake_redis.store[_KEY]["value"] == "alive-worker-token"
    assert fake_redis.store[_KEY]["pttl"] == 3_600_000


async def test_renew_uses_reacquire_with_capped_interval(fake_redis, monkeypatch):
    """续约走 reacquire()（TTL 重置，不滚雪球）且不碰 extend；节奏 min(ttl/3, 30s)。"""
    real_sleep = asyncio.sleep
    intervals: list[float] = []

    async def fast_sleep(t):
        intervals.append(t)
        await real_sleep(0)  # 立即让出控制权，推进 renew 循环

    monkeypatch.setattr("core.rag.lightrag.instance_pool.asyncio.sleep", fast_sleep)

    # 长租约：间隔封顶 30s
    lock, renew = await acquire_index_dlock("c1", "lightrag", ttl=3600)
    await real_sleep(0.02)  # 让 renew 循环跑若干轮（快进，不真等 30s）
    assert lock.reacquire_calls >= 1
    assert lock.extend_calls == 0
    assert fake_redis.store[_KEY]["pttl"] == 3_600_000  # 重置回 ttl，不是累加后的 7200s+
    await release_index_dlock(lock, renew)
    assert all(t == 30 for t in intervals)

    # 短租约：间隔 = ttl/3
    intervals.clear()
    fake_redis.store.clear()
    lock, renew = await acquire_index_dlock("c1", "lightrag", ttl=60)
    await real_sleep(0.02)
    await release_index_dlock(lock, renew)
    assert all(t == 20 for t in intervals)


async def test_renew_exits_when_lock_lost(fake_redis, monkeypatch):
    """丢锁（续约期间过期被抢/被偷）→ LockNotOwnedError → ERROR 留痕并退出循环。"""
    real_sleep = asyncio.sleep

    async def fast_sleep(t):  # noqa: ARG001
        await real_sleep(0)

    monkeypatch.setattr("core.rag.lightrag.instance_pool.asyncio.sleep", fast_sleep)

    lock, renew = await acquire_index_dlock("c1", "lightrag", ttl=3600)
    fake_redis.store[_KEY]["value"] = "someone-else"  # 锁被换走（过期被抢）
    await real_sleep(0.02)
    assert renew.done()  # 续约循环退出，不再空转
    assert lock.reacquire_calls == 1
    await release_index_dlock(lock, renew)  # release 幂等：锁不归自己时静默
