"""
Leader Worker Election — 多 worker 部署下的单例服务收敛。

使用 Redis SETNX 实现 leader 选举:
  - 启动时尝试获取 leader 锁 (worker:leader key)
  - 成功后持续续约 (每 15s)
  - 进程退出自动过期 (TTL 30s)
  - 只有 leader worker 启动 Cron/Bot/MCP 等单例服务

降级策略:
  - Redis 不可用时返回 False (所有 worker 都不启单例服务)
  - 防止重复启动导致的功能错误
"""
from __future__ import annotations

import asyncio
import logging
import os

import redis.asyncio as aioredis

from config import REDIS_URL

logger = logging.getLogger(__name__)

_LEADER_KEY = "worker:leader"
_LEASE_TTL = 30  # seconds
_RENEW_INTERVAL = _LEASE_TTL // 2  # 15 seconds

_is_leader: bool = False
_redis_client: aioredis.Redis | None = None
_renew_task: asyncio.Task | None = None


async def try_become_leader() -> bool:
    """尝试成为 leader worker。

    Returns:
        True  — 成功获取 leader 锁
        False — 未成为 leader 或 Redis 不可用
    """
    global _is_leader, _redis_client, _renew_task

    try:
        _redis_client = aioredis.from_url(
            REDIS_URL,
            socket_connect_timeout=2,
            decode_responses=True,
        )

        worker_id = str(os.getpid())
        acquired = await _redis_client.set(
            _LEADER_KEY,
            worker_id,
            nx=True,
            ex=_LEASE_TTL,
        )

        _is_leader = bool(acquired)

        if _is_leader:
            logger.info(
                "Leader election: worker %s became leader (pid=%s)",
                worker_id,
                os.getpid(),
            )
            # 启动续约 task
            _renew_task = asyncio.create_task(
                _renew_lease_loop(_redis_client, worker_id),
                name="leader:renew",
            )
        else:
            logger.info(
                "Leader election: worker %s is NOT leader (current leader in Redis)",
                os.getpid(),
            )
            await _redis_client.aclose()
            _redis_client = None

        return _is_leader

    except Exception as exc:
        logger.warning(
            "Leader election failed (Redis unavailable or error): %s",
            exc,
        )
        # 降级: Redis 不可用时返回 False,不启单例服务
        _is_leader = False
        if _redis_client:
            try:
                await _redis_client.aclose()
            except Exception:
                pass
            _redis_client = None
        return False


async def _renew_lease_loop(redis_client: aioredis.Redis, worker_id: str) -> None:
    """持续续约 leader 锁。

    每 15s 检查并续约,如果发现锁已被其他 worker 抢占,则退出续约循环。
    """
    try:
        while True:
            await asyncio.sleep(_RENEW_INTERVAL)

            try:
                current_leader = await redis_client.get(_LEADER_KEY)

                if current_leader == worker_id:
                    # 续约成功
                    await redis_client.expire(_LEADER_KEY, _LEASE_TTL)
                    logger.debug("Leader lease renewed for worker %s", worker_id)
                else:
                    # 锁已被抢占,退出续约
                    logger.warning(
                        "Leader lease lost (current leader: %s, expected: %s)",
                        current_leader,
                        worker_id,
                    )
                    global _is_leader
                    _is_leader = False
                    break

            except Exception as exc:
                logger.warning("Leader lease renewal failed: %s", exc)
                # Redis 错误时继续尝试续约 (可能临时故障)

    except asyncio.CancelledError:
        logger.info("Leader renew task cancelled, stopping renewal")
        raise
    except Exception as exc:
        logger.exception("Leader renew loop crashed: %s", exc)


def is_leader() -> bool:
    """检查当前 worker 是否是 leader。

    注意: 这是进程启动时确定的静态状态,不会动态变化。
    """
    return _is_leader


async def shutdown_leader() -> None:
    """清理 leader 资源 (应用 shutdown 时调用)。"""
    global _redis_client, _renew_task

    if _renew_task:
        _renew_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(_renew_task), timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        _renew_task = None

    if _redis_client:
        try:
            # 如果是 leader,主动释放锁 (可选,让其他 worker 立即抢占)
            if _is_leader:
                await _redis_client.delete(_LEADER_KEY)
                logger.info("Leader lock released during shutdown")
            await _redis_client.aclose()
        except Exception:
            pass
        _redis_client = None


__all__ = [
    "try_become_leader",
    "is_leader",
    "shutdown_leader",
]
