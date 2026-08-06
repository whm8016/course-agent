"""ask_user 回复跨 worker 投递通道（plan 阶段 2A）。

问题：``TurnRuntimeManager._reply_queues`` 是纯进程内 dict，ask_user 暂停的回复只能在持有
该 turn 的 worker 上投递，靠 Nginx ``ip_hash`` 兜——ip_hash 一变（重连 / 负载均衡）回复就丢。

本模块把回复投到 Redis（``BLPOP`` / ``RPUSH``），任意 worker 的 ``submit_user_reply`` 都能投到
正在 ``wait_reply`` 的那个 worker。Redis 不可用（``memory://`` / 测试环境）时回退进程内
``asyncio.Queue``，行为等价旧实现（同进程投递仍通，只是不跨 worker）。

同时维护 turn 归属 key（``ca:turn:owner:{turn_id} -> user_id``）：跨 worker 时
``submit_user_reply`` 本地 ``_executions`` 不命中，回落该 key 做 IDOR 校验，防 B 用户拿 A 的
turn_id 向 A 的 ask_user 投递回复、操纵 A 的对话。

降级写法参照 ``core/arq_pool.py``：``memory://`` 直接回退，连接异常 best-effort 不阻塞业务。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_REPLY_PREFIX = "ca:askuser:reply:"
_OWNER_PREFIX = "ca:turn:owner:"
_TTL = 600  # 回复 / 归属 key 保留 10 分钟（覆盖一次澄清往返 + 余量），过期自清理

# 懒加载的共享 redis 客户端（仅判定一次：memory:// 或连不上 → None，走进程内回退）
_redis: Any = None
_redis_checked = False
# Redis 不可用时的进程内回退：同进程 wait/push + owner 仍通；跨进程需 Redis
_fallback_queues: dict[str, asyncio.Queue] = {}
_fallback_owners: dict[str, str] = {}


def _redis_client() -> Any:
    """返回共享 redis 客户端；``memory://`` 或连不上返回 None（回退进程内）。仅判定一次。"""
    global _redis, _redis_checked
    if _redis_checked:
        return _redis
    _redis_checked = True
    try:
        from settings import get_settings

        url = get_settings().db.redis_url.get_secret_value()
        if not url or url.startswith("memory://"):
            return None
        import redis.asyncio as aioredis

        _redis = aioredis.from_url(url, decode_responses=True)
        logger.info("reply_channel: Redis 已启用，ask_user 回复支持跨 worker 投递")
        return _redis
    except Exception:
        logger.warning("reply_channel: Redis 不可用，回退进程内队列", exc_info=True)
        return None


def _reply_key(turn_id: str) -> str:
    return f"{_REPLY_PREFIX}{turn_id}"


def _owner_key(turn_id: str) -> str:
    return f"{_OWNER_PREFIX}{turn_id}"


async def wait_reply(turn_id: str, timeout: float) -> dict[str, Any] | None:
    """等待一条 ask_user 回复。``timeout=0`` 表示无限等；超时返回 None。"""
    r = _redis_client()
    if r is not None:
        try:
            res = await r.blpop(_reply_key(turn_id), timeout=int(timeout) if timeout > 0 else 0)
            if res is None:
                return None
            return json.loads(res[1])
        except Exception:
            logger.warning("reply_channel.wait_reply Redis 异常，回退进程内", exc_info=True)
    q = _fallback_queues.setdefault(turn_id, asyncio.Queue())
    if timeout and timeout > 0:
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    return await q.get()


async def push_reply(turn_id: str, payload: dict[str, Any]) -> bool:
    """投递一条 ask_user 回复。best-effort：无消费者时写入也会过期自清理。"""
    r = _redis_client()
    if r is not None:
        try:
            await r.rpush(_reply_key(turn_id), json.dumps(payload, ensure_ascii=False))
            await r.expire(_reply_key(turn_id), _TTL)
            return True
        except Exception:
            logger.warning("reply_channel.push_reply Redis 异常，回退进程内", exc_info=True)
    q = _fallback_queues.setdefault(turn_id, asyncio.Queue())
    await q.put(payload)
    return True


async def set_turn_owner(turn_id: str, user_id: str) -> None:
    """记录 turn 归属（供跨 worker submit_user_reply 做 IDOR 校验）。"""
    r = _redis_client()
    if r is not None:
        try:
            await r.set(_owner_key(turn_id), user_id, ex=_TTL)
            return
        except Exception:
            logger.warning("reply_channel.set_turn_owner Redis 异常，回退进程内", exc_info=True)
    _fallback_owners[turn_id] = user_id


async def get_turn_owner(turn_id: str) -> str | None:
    """读 turn 归属；不存在 / Redis 异常返回 None。"""
    r = _redis_client()
    if r is not None:
        try:
            return await r.get(_owner_key(turn_id))
        except Exception:
            logger.warning("reply_channel.get_turn_owner Redis 异常，回退进程内", exc_info=True)
    return _fallback_owners.get(turn_id)


def cleanup(turn_id: str) -> None:
    """turn 结束清理进程内回退态（Redis 的 key 靠 TTL 自清理）。"""
    _fallback_queues.pop(turn_id, None)
    _fallback_owners.pop(turn_id, None)
