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

from config import REDIS_URL

logger = logging.getLogger(__name__)

_pool: aioredis.Redis | None = None


def _get_pool() -> aioredis.Redis:
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
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


# ---------------------------------------------------------------------------
# FAQ 高频问题
# ---------------------------------------------------------------------------

import hashlib


def _faq_hash(question: str) -> str:
    """问题文本 → 16 位 MD5，用作 Redis key 的一部分。"""
    return hashlib.md5(question.strip().lower().encode()).hexdigest()[:16]


async def faq_record(course_id: str, question: str) -> int:
    """记录一次提问，返回该问题累计被问次数（失败返回 0）。"""
    try:
        key = f"faq:count:{course_id}"
        member = question.strip()
        count = await _get_pool().zincrby(key, 1, member)
        return int(count)
    except Exception:
        logger.debug("faq_record failed course=%s", course_id, exc_info=True)
        return 0


async def faq_top(course_id: str, n: int = 20) -> list[dict]:
    """返回 Top-N 高频问题，格式 [{'question': ..., 'count': ...}]。"""
    try:
        pairs = await _get_pool().zrevrange(f"faq:count:{course_id}", 0, n - 1, withscores=True)
        return [{"question": q, "count": int(s)} for q, s in pairs]
    except Exception:
        logger.debug("faq_top failed course=%s", course_id, exc_info=True)
        return []


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
