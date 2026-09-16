"""reply_channel 跨 worker 投递回归（plan 阶段 2A）。

验证：
1. Redis 路径：push 后 wait 能收到；wait 先阻塞、另一任务 push 后被唤醒（真 BLPOP 阻塞语义）。
   用 _FakeRedis 共享一个实例 = 共享 Redis，模拟「A 进程 wait / B 进程 push」。
2. Redis 不可用（None）→ 回退进程内 asyncio.Queue，同进程 push/wait 仍通。
3. IDOR：跨 worker submit_user_reply 本地 _executions miss，按 owner key 比对；他人 user_id / 未知 turn 被拒。
4. answer_now 跨 worker 标志位（ca:answer_now:{turn_id}）：set/check roundtrip、无 Redis 如实
   False（无进程内回退）、request_answer_now 三级路径（同 worker 快路径不碰 Redis / 死 turn
   短路在 Redis 之前 / 跨 worker owner 校验后置 flag）、watcher 轮询置本地 ev 与退出三途。
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
        self._ex: dict[str, int | None] = {}  # set() 收到的 ex（TTL），供断言不被丢弃

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
        self._ex[key] = ex
        return True

    async def get(self, key: str) -> str | None:
        return self._kv.get(key)


@pytest.mark.asyncio
async def test_redis_path_push_then_wait(use_redis):
    """Redis 路径：push 后 wait 收到同一 payload（_FakeRedis 共享态=跨进程投递）。"""
    use_redis(_FakeRedis())
    await rc.push_reply("t_redis", {"text": "你好", "answers": None})
    got = await rc.wait_reply("t_redis", timeout=1)
    assert got == {"text": "你好", "answers": None}


@pytest.mark.asyncio
async def test_redis_path_wait_blocks_until_push(use_redis):
    """waiter 先阻塞，另一任务 push 后被唤醒（真 BLPOP 阻塞→唤醒语义）。"""
    use_redis(_FakeRedis())
    out: dict = {}

    async def _wait():
        out["v"] = await rc.wait_reply("t_block", timeout=2)

    async def _push():
        await asyncio.sleep(0.05)
        await rc.push_reply("t_block", {"text": "late"})

    await asyncio.gather(_wait(), _push())
    assert out["v"] == {"text": "late"}


@pytest.mark.asyncio
async def test_fallback_when_redis_unavailable(monkeypatch, use_redis):
    """Redis 不可用（None）→ 回退进程内队列，同进程 push/wait 仍通。"""
    use_redis(None)
    monkeypatch.setattr(rc, "_fallback_queues", {})  # 隔离本测试
    await rc.push_reply("t_fb", {"text": "x"})
    got = await rc.wait_reply("t_fb", timeout=1)
    assert got == {"text": "x"}


@pytest.mark.asyncio
async def test_owner_key_idor(monkeypatch, use_redis):
    """跨 worker submit_user_reply：本地无 _executions，按 owner key 比对；他人/未知被拒。"""
    from services.session.turn_runtime import TurnRuntimeManager

    use_redis(None)  # owner/reply 都走进程内回退
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


@pytest.mark.asyncio
async def test_answer_now_flag_roundtrip(use_redis):
    """answer_now 标志位：set→True、check→True、未知 id→False；键名/值/TTL 符合约定。"""
    fake = _FakeRedis()
    use_redis(fake)
    assert await rc.set_answer_now("t_ans") is True
    assert await rc.check_answer_now("t_ans") is True
    assert await rc.check_answer_now("t_ans_other") is False
    assert fake._kv["ca:answer_now:t_ans"] == "1"
    assert fake._ex["ca:answer_now:t_ans"] == 600  # TTL 不可丢（死 turn 残窗宽度由它决定）


@pytest.mark.asyncio
async def test_answer_now_flag_no_redis_returns_false(use_redis):
    """无 Redis（memory://）→ set/check 均 False，无进程内回退（同进程路径已由本地 ev.set 覆盖）。"""
    use_redis(None)
    assert await rc.set_answer_now("t_noredis") is False
    assert await rc.check_answer_now("t_noredis") is False


@pytest.mark.asyncio
async def test_redis_available_predicate(use_redis):
    """redis_available()：配置了真实 Redis → True；None → False。"""
    use_redis(_FakeRedis())
    assert rc.redis_available() is True
    use_redis(None)
    assert rc.redis_available() is False


@pytest.mark.asyncio
async def test_answer_now_cross_worker_owner_three_states(use_redis):
    """跨 worker request_answer_now：本地无 _executions，按 owner key 比对；他人被拒且拒绝发生在 SET 之前。"""
    from services.session.turn_runtime import TurnRuntimeManager

    use_redis(_FakeRedis())
    await rc.set_turn_owner("t_x", "userA")

    mgr = TurnRuntimeManager()  # _executions 空 → 模拟 turn 在另一 worker
    # 本人 → True 且 flag 真的置了
    assert await mgr.request_answer_now("t_x", user_id="userA") is True
    assert await rc.check_answer_now("t_x") is True
    # 他人 → False（IDOR），且 flag 未泄漏（拒绝发生在 SET 之前）
    await rc.set_turn_owner("t_x2", "userA")
    assert await mgr.request_answer_now("t_x2", user_id="userB") is False
    assert await rc.check_answer_now("t_x2") is False
    # 未知 turn（无 owner key）→ 拒绝
    assert await mgr.request_answer_now("t_unknown", user_id="userA") is False
    # user_id=None 内部调用不设门槛（与 submit_user_reply 的 user_id=None 对称，parity）
    assert await mgr.request_answer_now("t_unknown2") is True


@pytest.mark.asyncio
async def test_answer_now_dead_turn_execution_window(use_redis):
    """死 turn 短路：execution 在（300s 回放窗内）而 ev 已清 → 即使归属本人、即使 Redis 可用也 False，
    且不得落入 Redis 路径（不置 flag）——否则跨 worker 会比修复前同 worker 更谎报 200。"""
    from types import SimpleNamespace

    from services.session.turn_runtime import TurnRuntimeManager

    use_redis(_FakeRedis())
    mgr = TurnRuntimeManager()
    mgr._executions["t_dead"] = SimpleNamespace(context=SimpleNamespace(user_id="uA"))
    assert await mgr.request_answer_now("t_dead", user_id="uA") is False
    assert await rc.check_answer_now("t_dead") is False
    # 分支与 user_id 无关（内部调用同样拦）
    assert await mgr.request_answer_now("t_dead") is False


@pytest.mark.asyncio
async def test_answer_now_same_worker_fast_path_and_idor(use_redis):
    """同 worker：① 快路径——ev 命中直接 set，不写 Redis 标志；② IDOR——归属不符
    则拒绝且 ev 不被置，钉住「归属校验在 ev 检查之前」的分支序（防 B 用户直连
    set A 的本地 ev）。"""
    from types import SimpleNamespace

    from services.session.turn_runtime import TurnRuntimeManager

    use_redis(_FakeRedis())
    mgr = TurnRuntimeManager()

    # ① 快路径：本人触发
    mgr._executions["t_fast"] = SimpleNamespace(context=SimpleNamespace(user_id="uA"))
    ev = asyncio.Event()
    mgr._answer_now_events["t_fast"] = ev
    assert await mgr.request_answer_now("t_fast", user_id="uA") is True
    assert ev.is_set() is True
    assert await rc.check_answer_now("t_fast") is False

    # ② IDOR：B 用户对 A 的 turn
    mgr._executions["t_idor2"] = SimpleNamespace(context=SimpleNamespace(user_id="uA"))
    ev2 = asyncio.Event()
    mgr._answer_now_events["t_idor2"] = ev2
    assert await mgr.request_answer_now("t_idor2", user_id="uB") is False
    assert ev2.is_set() is False
    assert await rc.check_answer_now("t_idor2") is False


@pytest.mark.asyncio
async def test_answer_now_watcher_sets_local_event(monkeypatch, use_redis):
    """watcher：Redis flag 置位后置本地 ev，随后自退（退出途①）。"""
    import services.session.turn_runtime as trt

    monkeypatch.setattr(trt, "_ANSWER_NOW_POLL_S", 0.05)
    use_redis(_FakeRedis())
    mgr = trt.TurnRuntimeManager()
    ev = asyncio.Event()
    watcher = asyncio.create_task(mgr._watch_answer_now_flag("t_w", ev))
    await asyncio.sleep(0.1)  # 让 watcher 先空转一轮
    await rc.set_answer_now("t_w")
    await asyncio.wait_for(ev.wait(), 1)
    await asyncio.wait_for(watcher, 1)  # 置位后自退


@pytest.mark.asyncio
async def test_answer_now_watcher_exit_and_cancel(monkeypatch, use_redis):
    """watcher 退出：本地 ev 已置 → while 不成立自退（退出途②）；sleep 中 cancel → 吞
    CancelledError 正常返回（退出途③，无未检索异常）。"""
    import services.session.turn_runtime as trt

    monkeypatch.setattr(trt, "_ANSWER_NOW_POLL_S", 0.05)
    use_redis(_FakeRedis())
    mgr = trt.TurnRuntimeManager()

    # ① 直连路径置 ev → watcher 自退
    ev1 = asyncio.Event()
    w1 = asyncio.create_task(mgr._watch_answer_now_flag("t_w1", ev1))
    ev1.set()
    await asyncio.wait_for(w1, 1)
    assert w1.cancelled() is False

    # ② sleep 中 cancel → 正常返回（CancelledError 被吞）
    ev2 = asyncio.Event()
    w2 = asyncio.create_task(mgr._watch_answer_now_flag("t_w2", ev2))
    await asyncio.sleep(0.02)  # 确保已进到 sleep
    w2.cancel()
    await asyncio.wait_for(w2, 1)
    assert w2.cancelled() is False
