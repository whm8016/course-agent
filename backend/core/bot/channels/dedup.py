"""Redis-based IM message dedup（H-14）。

替换 in-memory deque / OrderedDict，leader failover 后不丢失去重状态。
使用 Redis SET + TTL；Redis 不可用时降级为"不重复"（安全侧——可能重复处理消息，
但不会丢消息）。

核心 API 是 ``claim_processed``（原子 SET NX）：一次调用同时完成「查 + 标记」，
消除 ``is_processed`` + ``mark_processed`` 两步之间的 TOCTOU 竞态（两步之间 leader
failover / 并发回调都可能导致同一 message_id 被判定未处理两次）。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEDUP_TTL = 3600  # 1 小时，覆盖 leader failover + 重连窗口


def _dedup_key(platform: str, message_id: str) -> str:
    return f"im:dedup:{platform}:{message_id}"


async def claim_processed(platform: str, message_id: str, ttl: int = _DEDUP_TTL) -> bool:
    """原子抢占：``SET key 1 NX EX ttl``。

    Returns:
        True  — 本次抢到（首次处理），调用方应继续处理该消息。
        False — 已被处理过（或 Redis 不可用，降级为「允许处理」见下）。

    Redis 不可用时返回 True（降级为「允许处理」）——与 is_processed 降级语义一致：
    宁可重复处理也不丢消息（最坏在 failover 窗口内重复回一条消息，不损坏数据）。
    """
    try:
        from core.db.cache import _get_pool

        r = _get_pool()
        result = await r.set(_dedup_key(platform, message_id), "1", nx=True, ex=ttl)
        return bool(result)
    except Exception:
        logger.exception("dedup claim failed: %s/%s", platform, message_id)
        return True  # 降级：允许处理（不丢消息）


async def is_processed(platform: str, message_id: str) -> bool:
    """检查 message_id 是否已处理过。Redis 不可用时返回 False（降级为不重复）。"""
    try:
        from core.db.cache import _get_pool

        r = _get_pool()
        return await r.exists(_dedup_key(platform, message_id)) > 0
    except Exception:
        logger.exception("dedup check failed: %s/%s", platform, message_id)
        return False


async def mark_processed(platform: str, message_id: str, ttl: int = _DEDUP_TTL) -> None:
    """标记 message_id 为已处理（SET + EXPIRE）。Redis 不可用时静默降级。"""
    try:
        from core.db.cache import _get_pool

        r = _get_pool()
        await r.set(_dedup_key(platform, message_id), "1", ex=ttl)
    except Exception:
        logger.exception("dedup mark failed: %s/%s", platform, message_id)
