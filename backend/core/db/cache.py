"""Thin async Redis cache helpers.

Provides get/set/invalidate with automatic JSON serialisation and TTL.
Falls back gracefully (returns None / silently skips) when Redis is unavailable,
so the app degrades to "always-miss" rather than crashing.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from settings import get_settings
REDIS_URL = get_settings().db.redis_url.get_secret_value()

logger = logging.getLogger(__name__)

_pool: aioredis.Redis | None = None


def _get_pool() -> aioredis.Redis:
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            # 命令执行超时（对齐 core/leader.py 的选主客户端）：只设连接超时的话，
            # 网络分区时已建立的连接上命令会无限挂死——索引锁续约协程卡住 → TTL
            # 衰减被偷锁 → 双写。本池无阻塞读（blpop 在 reply_channel 自建客户端上），
            # 3s 对本地 Redis 的小命令绰绰有余。
            socket_timeout=3,
        )
    return _pool


async def cache_get(key: str) -> Any | None:
    """Return deserialised value or None on miss / error."""
    try:
        raw = await _get_pool().get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        logger.debug("cache_get failed for key=%s", key, exc_info=True)
        return None


async def cache_set(key: str, value: Any, ttl: int = 60) -> None:
    """Serialise *value* to JSON and store with *ttl* seconds expiry."""
    try:
        await _get_pool().set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
    except Exception:
        logger.debug("cache_set failed for key=%s", key, exc_info=True)


async def cache_delete(key: str) -> None:
    """Remove a single key (best-effort)."""
    try:
        await _get_pool().delete(key)
    except Exception:
        logger.debug("cache_delete failed for key=%s", key, exc_info=True)


async def cache_delete_pattern(pattern: str) -> None:
    """Remove all keys matching *pattern* (e.g. ``sessions:user:abc*``).

    Uses SCAN to avoid blocking Redis on large keyspaces.
    """
    try:
        pool = _get_pool()
        cursor: int | bytes = 0
        while True:
            cursor, keys = await pool.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                await pool.delete(*keys)
            if cursor == 0:
                break
    except Exception:
        logger.debug("cache_delete_pattern failed for pattern=%s", pattern, exc_info=True)


def user_courses_cache_key(user_id: str) -> str:
    """课程列表缓存 key（自 api/courses.py 下沉：与 invalidate_courses_cache 同居，
    key 前缀不再跨模块分裂——改格式只须看这一处）。"""
    return f"courses:list:{user_id}"


async def invalidate_courses_cache() -> None:
    """课程列表缓存批量失效（admin / teacher / indexing_runner 在创建/删除/索引完成时调用）。

    课程列表按用户缓存（key 为 courses:list:{user_id}），课程的可见性/状态变更会影响
    所有学生，因此用 SCAN 匹配前缀批量失效全部用户的缓存，而非删单个不存在的固定 key。
    实现自 api/courses.py 下沉（core/rag/indexing_runner 需要调它——曾因此把索引
    业务逻辑留在 api/admin，见该模块迁移说明）。
    """
    await cache_delete_pattern("courses:list:*")


# ---------------------------------------------------------------------------
# FAQ 高频问题
# ---------------------------------------------------------------------------

import hashlib


def _faq_hash(question: str) -> str:
    """问题文本 → 16 位 MD5，用作 Redis key 的一部分。"""
    return hashlib.md5(question.strip().lower().encode()).hexdigest()[:16]


async def faq_answer_get(course_id: str, question: str) -> str | None:
    """获取高频问题的缓存答案（无则返回 None）。"""
    try:
        key = f"faq:answer:{course_id}:{_faq_hash(question)}"
        return await _get_pool().get(key)
    except Exception:
        logger.debug("faq_answer_get failed course=%s", course_id, exc_info=True)
        return None


async def faq_answer_set(course_id: str, question: str, answer: str) -> None:
    """永久缓存高频问题的答案（不设 TTL）。"""
    try:
        key = f"faq:answer:{course_id}:{_faq_hash(question)}"
        await _get_pool().set(key, answer)
    except Exception:
        logger.debug("faq_answer_set failed course=%s", course_id, exc_info=True)


# ---------------------------------------------------------------------------
# 用户课程访问权限缓存（TTL 5 分钟）
# ---------------------------------------------------------------------------

_ACCESS_TTL = 300  # 秒


async def course_access_get(user_id: str, course_id: str) -> bool | None:
    """返回缓存的权限决策：True=允许, False=拒绝, None=未命中。"""
    try:
        raw = await _get_pool().get(f"access:{user_id}:{course_id}")
        if raw is None:
            return None
        return raw == "1"
    except Exception:
        logger.debug("course_access_get failed user=%s course=%s", user_id, course_id, exc_info=True)
        return None


async def course_access_set(user_id: str, course_id: str, allowed: bool) -> None:
    """缓存权限决策，TTL 5 分钟。"""
    try:
        await _get_pool().set(
            f"access:{user_id}:{course_id}",
            "1" if allowed else "0",
            ex=_ACCESS_TTL,
        )
    except Exception:
        logger.debug("course_access_set failed user=%s course=%s", user_id, course_id, exc_info=True)


async def course_access_invalidate(user_id: str, course_id: str | None = None) -> None:
    """使某用户的权限缓存失效（指定课程或全部）。"""
    try:
        if course_id:
            await _get_pool().delete(f"access:{user_id}:{course_id}")
        else:
            await cache_delete_pattern(f"access:{user_id}:*")
    except Exception:
        logger.debug("course_access_invalidate failed user=%s", user_id, exc_info=True)
