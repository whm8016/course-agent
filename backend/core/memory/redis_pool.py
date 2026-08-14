"""core.memory 共享 Redis 连接池（懒加载单例）。

从 flush_manager 抽出——flush cron 停用后仍有三个消费方：consolidation.add_importance /
reset_importance 与 graph_memory 的 turn 计数。集中一处避免各开一个 pool。
"""
from __future__ import annotations

import redis.asyncio as aioredis

_redis_pool: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """返回模块级 Redis 连接池（懒加载，复用同一个 pool，不需要每次 aclose）。"""
    global _redis_pool
    if _redis_pool is None:
        from settings import get_settings
        REDIS_URL = get_settings().db.redis_url.get_secret_value()
        _redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_pool
