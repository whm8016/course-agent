"""
Leader Worker Election — 多 worker 部署下的单例服务收敛。

效率型锁（最坏情况是 Cron/Bot/MCP 多跑或停摆一次，不损坏数据），单实例 Redis
足够，无需 Redlock。状态翻转通过回调通知 main 拉起/停止单例服务。

选举模型（对标 Redisson / K8s lease 选主）：
  - 启动时尝试获取 leader 锁（``worker:leader`` key，TTL 30s）。
  - 成功 → 进入续约 loop（每 15s **CAS 原子续约**）。
  - 失败 → 进入**竞选者循环**（每 10s 重试 SETNX），不再「启动抢一次就放弃」。
  - 续约发现锁被别人持（CAS 返回 0）→ ``on_lose`` 停单例 → 重回竞选 loop。
  - 竞选抢到 → ``on_gain`` 拉单例 → 进续约 loop。形成闭环。

为什么需要竞选者循环：leader 进程**卡死但未被进程管理器杀死**（事件循环阻塞 /
GC 长暂停）时，它的续约 loop 一起卡死 → TTL 过期。若非 leader 永不重试，则要
等到 Gunicorn ``--timeout`` 杀掉卡死 worker、拉起新进程才能接管（分钟级空窗）。
竞选者循环让现存 worker 持续竞争，把接管空窗压到「TTL + 竞选间隔 ≈ 40s」。

四道防线：
  1. 竞选原子（``SET ... NX``）—— 任一时刻只有一个 leader，无脑裂。
  2. CAS 原子续约（Lua）—— 消除 ``GET+EXPIRE`` 竞态：否则续约者可能误续别人
     （竞选接管后）刚抢到的锁，导致双 leader、单例服务跑两份。
  3. worker_id 全局唯一（``pid-uuid``）—— pid 跨重启会复用，CAS 续约会误判。
  4. 竞选成功二次确认 —— ``SET NX`` 后 ``GET`` 校验 value==自己再 ``on_gain``。

降级：Redis 不可用时竞选/续约 loop 捕获异常后继续重试（恢复即接管），不崩溃。
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Awaitable, Callable

import redis.asyncio as aioredis

from core.observability.metrics import set_leader_status
from settings import get_settings

logger = logging.getLogger(__name__)

_LEADER_KEY = "worker:leader"
_LEASE_TTL = 30  # seconds
_RENEW_INTERVAL = _LEASE_TTL // 2  # 15 seconds
_CAMPAIGN_INTERVAL = 10  # 非 leader 竞选节奏（比续约略短，加快接管）

# CAS 原子续约：仅当锁 value==自己（worker_id）才续期；否则返回 0（锁已被别人持/过期）。
# 消除 GET+EXPIRE 的竞态：leader A 的 GET 与 EXPIRE 之间锁过期、竞选者 B 抢到，
# A 的 EXPIRE 会续 B 的锁 → A、B 都以为自己是 leader、单例服务跑两份（脑裂）。
_RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
else
    return 0
end
"""

# CAS 原子删锁：仅当锁 value==自己（worker_id）才 del，否则返回 0。shutdown 时若用无条件
# delete 会误删别人的锁——follower 正常重启会把真 leader 的锁删掉（→ leader 续约 CAS 返回 0
# 误判丢锁、触发空窗）；leader 自身 shutdown 时若锁已被新 leader 抢走（_is_leader 仍 True 的
# 极端窗口）也会删错人的锁。CAS 保证"只有锁是自己的才删"。
_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

# worker 身份：pid 跨重启/容器会复用，加 uuid 保证全局唯一，CAS 续约才不会误判。
_worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

_is_leader: bool = False
_redis_client: aioredis.Redis | None = None
_active_task: asyncio.Task | None = None  # 当前活动 loop（renew 或 campaign）
_on_gain: Callable[[], Awaitable[None]] | None = None
_on_lose: Callable[[], Awaitable[None]] | None = None

# 关停标志（C-1/C-3）：应用进入 shutdown 流程后置位。lifespan shutdown 第一行经
# mark_shutting_down() 置位，是「即将死亡」的最早信号——比 _is_leader=False、比
# stop_singleton_services 都早。start_singleton_services 在每个 await 后检查它，
# 发现已关停则回滚已启动服务、绝不置位 _singletons_started，避免单例在死亡 worker
# 上被拉起又无人停止。
_shutting_down: bool = False


def mark_shutting_down() -> None:
    """标记应用正在关停（lifespan shutdown 第一行调用）。

    置位后不可逆：本进程必然退出，单例服务绝不应再被启动。幂等。
    """
    global _shutting_down
    _shutting_down = True


def is_shutting_down() -> bool:
    """是否已进入关停流程（start_singleton_services 的检查点读它）。"""
    return _shutting_down


def register_leader_callbacks(
    on_gain: Callable[[], Awaitable[None]],
    on_lose: Callable[[], Awaitable[None]],
) -> None:
    """注册 leader 状态翻转回调（main.py lifespan startup 调用一次）。

    - on_gain：成为 leader 时拉起单例服务（Cron/Bot/MCP）。需幂等。
    - on_lose：丢失 leader 时停止单例服务。需幂等。
    回调异常只记日志，不影响选举 loop。
    """
    global _on_gain, _on_lose
    _on_gain = on_gain
    _on_lose = on_lose


async def try_become_leader() -> bool:
    """尝试成为 leader worker；失败则进入竞选者循环（持续重试，可能后续接管）。

    Returns:
        True  — 首次即获取 leader 锁
        False — 未成为 leader（已起竞选 loop，后续锁过期可接管）
    """
    global _active_task

    _ensure_client()
    if _redis_client is not None:
        try:
            if await _try_acquire():
                await _become_leader()
                return True
        except Exception as exc:
            logger.warning("Leader election: initial acquire failed: %s", exc)

    logger.info(
        "Leader election: worker %s is NOT leader yet; entering campaign loop (every %ds)",
        _worker_id,
        _CAMPAIGN_INTERVAL,
    )
    _active_task = asyncio.create_task(_campaign_loop(), name="leader:campaign")
    return False


async def _try_acquire() -> bool:
    """单次原子抢锁：``SET key worker_id NX EX TTL``，成功后二次确认 value==自己。"""
    if _redis_client is None:
        return False
    acquired = await _redis_client.set(
        _LEADER_KEY,
        _worker_id,
        nx=True,
        ex=_LEASE_TTL,
    )
    if not acquired:
        return False
    # 二次确认（防御极端窗口，如 Redis 主从切换）
    current = await _redis_client.get(_LEADER_KEY)
    return current == _worker_id


async def _become_leader() -> None:
    """切换到 leader 状态：标志 + metrics + 起续约 loop + on_gain 回调。

    先起续约 loop、再 on_gain，保证拉单例期间锁不会因无续约而过期。
    不 cancel 旧 campaign task —— 它会在本函数返回后自然 return（调用方约定）。
    """
    global _is_leader, _active_task
    if _is_leader:
        return
    _is_leader = True
    set_leader_status(_worker_id, True)
    logger.info("Leader election: worker %s became leader", _worker_id)
    _active_task = asyncio.create_task(_renew_lease_loop(), name="leader:renew")
    await _invoke_callback(_on_gain, "on_gain")


async def _lose_leader() -> None:
    """切换到 follower 状态：标志 + metrics + on_lose 回调 + 起竞选 loop。

    先起竞选 loop、再 on_lose。旧续约 task 会在本函数返回后自然 return。
    """
    global _is_leader, _active_task
    if not _is_leader:
        return
    _is_leader = False
    set_leader_status(_worker_id, False)
    logger.warning("Leader election: worker %s lost leadership", _worker_id)
    _active_task = asyncio.create_task(_campaign_loop(), name="leader:campaign")
    await _invoke_callback(_on_lose, "on_lose")


async def _invoke_callback(
    cb: Callable[[], Awaitable[None]] | None,
    name: str,
) -> None:
    if cb is None:
        return
    try:
        await cb()
    except Exception:
        logger.exception("Leader callback %s failed (non-fatal)", name)


def _ensure_client() -> None:
    """惰性创建 redis client（实际连接在首次命令时建立）。"""
    global _redis_client
    if _redis_client is not None:
        return
    try:
        url = get_settings().db.redis_url.get_secret_value()
        _redis_client = aioredis.from_url(
            url,
            socket_connect_timeout=2,
            socket_timeout=3,  # 命令执行超时（略大于一个 RTT）；只有连接超时会让 eval 在网络分区时无限挂起 → 续约 loop 卡死、TTL 过期后幽灵 leader
            decode_responses=True,
        )
    except Exception as exc:
        logger.warning("Leader: redis client init failed: %s", exc)
        _redis_client = None


async def _renew_lease_loop() -> None:
    """leader 续约 loop：每 15s CAS 原子续约；锁被别人持则 _lose_leader 切竞选。

    Redis 连接异常时**短暂不丢锁**（lease 语义保持 leader 身份，待 Redis 恢复）。
    但若中断持续超过一个 lease TTL（M-1）：锁在 Redis 侧早已过期被别人抢，本 worker
    却仍 `_is_leader=True`、继续当 leader → 双 leader 窗口。故引入续约失败时间窗：
    自首次失败起累积 >= TTL 即主动 `_lose_leader`——此时锁必然已不在我们名下，
    主动让位比「装作还是 leader」安全得多。短抖动（< TTL）仍保持身份，不误伤。
    """
    try:
        fail_since: float | None = None  # 续约连续失败起始时间（monotonic）；成功则重置为 None
        while True:
            await asyncio.sleep(_RENEW_INTERVAL)
            try:
                _ensure_client()
                if _redis_client is None:
                    # 无 client 视同失败，纳入时间窗统计（M-1）
                    if fail_since is None:
                        fail_since = asyncio.get_event_loop().time()
                    elif asyncio.get_event_loop().time() - fail_since >= _LEASE_TTL:
                        logger.warning(
                            "Leader renew: redis unavailable > TTL; relinquishing to avoid split brain"
                        )
                        await _lose_leader()
                        return
                    logger.warning("Leader renew: redis unavailable; retry next cycle")
                    continue
                renewed = await _redis_client.eval(
                    _RENEW_SCRIPT,
                    1,
                    _LEADER_KEY,
                    _worker_id,
                    _LEASE_TTL,
                )
                if renewed:
                    logger.debug("Leader lease renewed for worker %s", _worker_id)
                    fail_since = None  # 成功，重置失败窗
                else:
                    # 锁已被别人持或过期 → 丢锁、切竞选
                    await _lose_leader()
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # M-1：连续失败超过 TTL 才丢锁；短抖动仅 log 重试、保持身份
                if fail_since is None:
                    fail_since = asyncio.get_event_loop().time()
                elif asyncio.get_event_loop().time() - fail_since >= _LEASE_TTL:
                    logger.warning(
                        "Leader renew: failures persisted > TTL; relinquishing to avoid split brain: %s",
                        exc,
                    )
                    await _lose_leader()
                    return
                logger.warning("Leader lease renewal failed (will retry): %s", exc)
    except asyncio.CancelledError:
        logger.info("Leader renew task cancelled")
        raise


async def _campaign_loop() -> None:
    """follower 竞选 loop：每 10s SETNX 抢锁；抢到则 _become_leader。

    Redis 异常时继续重试（恢复即接管）。每 ~60s 输出一次状态日志，便于发现空窗。
    """
    round_ = 0
    try:
        while True:
            await asyncio.sleep(_CAMPAIGN_INTERVAL)
            round_ += 1
            try:
                _ensure_client()
                if _redis_client is None:
                    if round_ % 6 == 0:
                        logger.warning(
                            "Leader campaign: redis still unavailable (round %d)",
                            round_,
                        )
                    continue
                if await _try_acquire():
                    logger.info(
                        "Leader campaign: worker %s won leadership after %d round(s)",
                        _worker_id,
                        round_,
                    )
                    await _become_leader()
                    return
                if round_ % 6 == 0:
                    logger.info(
                        "Leader campaign: worker %s still follower (round %d)",
                        _worker_id,
                        round_,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Leader campaign attempt failed: %s", exc)
    except asyncio.CancelledError:
        logger.info("Leader campaign task cancelled")
        raise


def is_leader() -> bool:
    """检查当前 worker 是否是 leader。

    动态反映当前状态（竞选接管 / 丢锁会实时更新），不再是启动时的静态快照。
    """
    return _is_leader


async def shutdown_leader() -> None:
    """清理 leader 资源（应用 shutdown 时调用）。"""
    global _redis_client, _is_leader, _active_task

    # C-1/C-3：置位关停标志（双保险——lifespan shutdown 第一行也会置位）。后续
    # 即便有残留回调试图 start_singleton_services，也会因 is_shutting_down() 早退。
    mark_shutting_down()

    # 先停选举 loop（续约/竞选），避免释放锁期间又续约或抢锁。
    # M-2：旧实现用 ``asyncio.shield(_active_task)`` 包裹——若 cancel 的瞬间 task 正在
    # ``_lose_leader()`` 里（如 CAS 续约返回 0），``_lose_leader`` 会 spawn 一个**新竞选
    # task** 并赋给 ``_active_task``；shield 反而保护了这个本该被取消的新 task，导致
    # shutdown 后竞选继续、可能重新抢锁变 leader。改为：捕获 cancel 前的旧引用，
    # await 其退出后，再把内部 reassign 出来的新 task 一并 cancel，确保无残留 loop。
    task_to_stop = _active_task
    if task_to_stop is not None and not task_to_stop.done():
        task_to_stop.cancel()
        try:
            await asyncio.wait_for(task_to_stop, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
    # cancel 期间 _lose_leader/_become_leader 可能已把 _active_task 重赋为新 loop；
    # shutdown 流程必须把它也停掉，否则本进程已死、竞选 loop 仍在抢锁。
    if _active_task is not None and _active_task is not task_to_stop and not _active_task.done():
        _active_task.cancel()
        try:
            await asyncio.wait_for(_active_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
    _active_task = None

    # 仅 leader 释放锁：follower shutdown 不动锁（旧实现无条件 delete 会删掉真 leader 的锁
    # → 误丢锁空窗）。CAS 原子删锁：锁已被别人抢时返回 0，不误删新 leader 的锁。
    # 顺序——先 CAS 释放锁（其他 worker 立刻能竞选接管），再 on_lose 停单例：把接管空窗
    # 压到最小（原实现先 on_lose 再删锁，停机期间无人能接管）。
    if _is_leader and _redis_client is not None:
        try:
            released = await _redis_client.eval(
                _RELEASE_SCRIPT, 1, _LEADER_KEY, _worker_id
            )
            if released:
                logger.info("Leader lock released during shutdown")
            else:
                logger.warning(
                    "Leader lock already taken by another worker at shutdown"
                )
        except Exception:
            pass
        await _invoke_callback(_on_lose, "on_lose")
        _is_leader = False
        set_leader_status(_worker_id, False)

    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
        _redis_client = None


__all__ = [
    "try_become_leader",
    "is_leader",
    "shutdown_leader",
    "register_leader_callbacks",
    "mark_shutting_down",
    "is_shutting_down",
]
