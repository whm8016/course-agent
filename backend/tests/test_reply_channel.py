"""reply_channel 跨 worker 投递回归（plan 阶段 2A）。

验证：
1. Redis 路径：push 后 wait 能收到；wait 先阻塞、另一任务 push 后被唤醒（真 BLPOP 阻塞语义）。
   用 _FakeRedis 共享一个实例 = 共享 Redis，模拟「A 进程 wait / B 进程 push」。
2. Redis 不可用（None）→ 回退进程内 asyncio.Queue，同进程 push/wait 仍通。
3. IDOR：跨 worker submit_user_reply 本地 _executions miss，按 owner key 比对；他人 user_id / 未知 turn 被拒。
"""
from __future__ import annotations

import asyncio

import pytest

import services.session.reply_channel as rc


class _FakeRedis:
    """最小 Redis 模拟：blpop/rpush（asyncio.Queue 驱动，race-free）+ set/get/expire。

    共享同一个 _FakeRedis 实例 = 共享 Redis（模拟跨进程）。
    """

    def __init__(self) -> None:
        self._qs: dict[str, asyncio.Queue] = {}
        self._kv: dict[str, str] = {}

    def _q(self, key: str) -> asyncio.Queue:
        return self._qs.setdefault(key, asyncio.Queue())

    async def rpush(self, key: str, val: str) -> int:
        await self._q(key).put(val)
        return 1

    async def blpop(self, key: str, timeout: float = 0):
        try:
            if timeout and timeout > 0:
                v = await asyncio.wait_for(self._q(key).get(), timeout=timeout)
            else:
                v = await self._q(key).get()
            return (key, v)
        except asyncio.TimeoutError:
            return None

    async def expire(self, key: str, ttl: int) -> bool:  # noqa: ARG002
        return True

    async def set(self, key: str, val: str, ex: int | None = None) -> bool:  # noqa: ARG002
        self._kv[key] = val
        return True

    async def get(self, key: str) -> str | None:
        return self._kv.get(key)


def _use(monkeypatch, redis_value) -> None:
    """强制 reply_channel._redis_client() 返回 redis_value（None=走进程内回退）。"""
    monkeypatch.setattr(rc, "_redis_checked", True)
    monkeypatch.setattr(rc, "_redis", redis_value)


@pytest.mark.asyncio
async def test_redis_path_push_then_wait(monkeypatch):
    """Redis 路径：push 后 wait 收到同一 payload（_FakeRedis 共享态=跨进程投递）。"""
    _use(monkeypatch, _FakeRedis())
    await rc.push_reply("t_redis", {"text": "你好", "answers": None})
    got = await rc.wait_reply("t_redis", timeout=1)
    assert got == {"text": "你好", "answers": None}


@pytest.mark.asyncio
async def test_redis_path_wait_blocks_until_push(monkeypatch):
    """waiter 先阻塞，另一任务 push 后被唤醒（真 BLPOP 阻塞→唤醒语义）。"""
    _use(monkeypatch, _FakeRedis())
    out: dict = {}

    async def _wait():
        out["v"] = await rc.wait_reply("t_block", timeout=2)

    async def _push():
        await asyncio.sleep(0.05)
        await rc.push_reply("t_block", {"text": "late"})

    await asyncio.gather(_wait(), _push())
    assert out["v"] == {"text": "late"}


@pytest.mark.asyncio
async def test_fallback_when_redis_unavailable(monkeypatch):
    """Redis 不可用（None）→ 回退进程内队列，同进程 push/wait 仍通。"""
    _use(monkeypatch, None)
    monkeypatch.setattr(rc, "_fallback_queues", {})  # 隔离本测试
    await rc.push_reply("t_fb", {"text": "x"})
    got = await rc.wait_reply("t_fb", timeout=1)
    assert got == {"text": "x"}


@pytest.mark.asyncio
async def test_owner_key_idor(monkeypatch):
    """跨 worker submit_user_reply：本地无 _executions，按 owner key 比对；他人/未知被拒。"""
    from services.session.turn_runtime import TurnRuntimeManager

    _use(monkeypatch, None)  # owner/reply 都走进程内回退
    monkeypatch.setattr(rc, "_fallback_owners", {})
    monkeypatch.setattr(rc, "_fallback_queues", {})

    await rc.set_turn_owner("t_idor", "userA")
    assert await rc.get_turn_owner("t_idor") == "userA"

    mgr = TurnRuntimeManager()  # _executions 空 → 模拟 turn 在另一 worker
    # 归属相符 → 投递成功，且回复能被 waiter 取到
    ok = await mgr.submit_user_reply("t_idor", text="回答", user_id="userA")
    assert ok is True
    assert await rc.wait_reply("t_idor", timeout=1) == {"text": "回答", "answers": None, "outline": None}
    # 他人 user_id → 拒绝（IDOR）
    bad = await mgr.submit_user_reply("t_idor", text="投毒", user_id="userB")
    assert bad is False
    # 未知 turn（无 owner key）→ 拒绝
    assert await mgr.submit_user_reply("t_unknown", user_id="userA") is False
