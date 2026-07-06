"""ARQ 异步任务队列连接池（全局单例）。

FastAPI 启动时调用 get_arq_pool() 预热；关闭时调用 close_arq_pool()。
Redis 不可用（如测试环境 memory://）时静默返回 None，调用方降级为 BackgroundTasks。
"""
from __future__ import annotations

import logging
import urllib.parse

logger = logging.getLogger(__name__)

_pool = None  # arq.ArqRedis | None


def _arq_settings_from_url(url: str):
    """将 REDIS_URL 转换为 arq.connections.RedisSettings。"""
    from arq.connections import RedisSettings

    parsed = urllib.parse.urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password or None,
        database=int((parsed.path or "/0").lstrip("/") or 0),
    )


async def get_arq_pool():
    """返回共享 ARQ 连接池，Redis 不可用时返回 None。"""
    global _pool
    if _pool is not None:
        return _pool

    from settings import get_settings
    REDIS_URL = get_settings().db.redis_url.get_secret_value()

    # memory:// 是测试占位，ARQ 不支持
    if not REDIS_URL or REDIS_URL.startswith("memory://"):
        return None

    try:
        from arq import create_pool

        settings = _arq_settings_from_url(REDIS_URL)
        _pool = await create_pool(settings)
        logger.info("ARQ pool connected (Redis %s)", REDIS_URL)
        return _pool
    except Exception:
        logger.warning("ARQ pool unavailable – 索引/研究任务将降级为 BackgroundTasks", exc_info=True)
        return None


async def close_arq_pool() -> None:
    """关闭 ARQ 连接池（应用退出时调用）。"""
    global _pool
    if _pool is not None:
        try:
            await _pool.aclose()
        except Exception:
            pass
        _pool = None
        logger.info("ARQ pool closed")
