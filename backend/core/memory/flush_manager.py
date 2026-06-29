"""Memory Batch Flush Manager — Producer 模式（无状态）。

核心目标：将 per-turn 的记忆更新改为批量处理，减少 LLM 调用次数。

架构：Producer-Consumer 分离
- **Producer**（API worker）：turn 完成后只做 Redis RPUSH + SET timestamp，零逻辑，无锁，无状态
- **Consumer**（ARQ worker cron）：每 30s 扫描一次 Redis，满批或 idle 超时的 key 批量 flush

触发条件（任一满足即 flush）：
1. 累积轮数 >= max_turns（默认 3 轮）
2. 最后一轮后静默 idle_timeout 秒（默认 120s）

Redis key 结构：
- `mem_flush:{user_id}:{session_id}` — List，存储对话 JSON
- `mem_flush:{user_id}:{session_id}:ts` — String，最后更新时间戳
- `mem_flush:{user_id}:{session_id}:meta` — String，元数据（user_id, course_id）
"""
from __future__ import annotations

import json
import logging
import time

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Redis key 前缀和后缀
_PREFIX = "mem_flush:"
_TS_SUFFIX = ":ts"
_META_SUFFIX = ":meta"
_TTL = 600  # key 最长存活 10 分钟（兜底防泄漏）


_redis_pool: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    """返回模块级 Redis 连接池（懒加载，复用同一个 pool，不需要每次 aclose）。"""
    global _redis_pool
    if _redis_pool is None:
        from config import REDIS_URL
        _redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis_pool


async def enqueue(
    user_id: str,
    session_id: str,
    course_id: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """每轮对话结束后调用，将对话数据写入 Redis（Producer 模式）。

    Args:
        user_id: 用户 ID
        session_id: 会话 ID（用于区分不同会话的 buffer）
        course_id: 课程 ID
        user_msg: 用户消息
        assistant_msg: 助手回复
    """
    # 门槛过滤：跳过无意义短消息（与 mem0_client.add_turn 共用同一规则，
    # 保证生产者端就拦截，避免无谓的 LLM 提取）
    from core.memory.mem0_client import should_skip_user_message

    if not user_id or should_skip_user_message(user_msg):
        logger.debug(
            "[flush_manager] enqueue SKIPPED user=%s reason=gate",
            user_id,
        )
        return

    # session_id 为空时使用 user_id 作为 fallback
    key = f"{_PREFIX}{user_id}:{session_id}" if session_id else f"{_PREFIX}{user_id}"

    r = _get_redis()
    pipe = r.pipeline()
    # RPUSH 对话数据
    pipe.rpush(key, json.dumps({"u": user_msg, "a": assistant_msg}))
    # SET 最后更新时间戳
    pipe.set(f"{key}{_TS_SUFFIX}", str(time.time()))
    # SET 元数据
    pipe.set(f"{key}{_META_SUFFIX}", json.dumps({"user_id": user_id, "course_id": course_id}))
    # 设置 TTL
    pipe.expire(key, _TTL)
    pipe.expire(f"{key}{_TS_SUFFIX}", _TTL)
    pipe.expire(f"{key}{_META_SUFFIX}", _TTL)
    await pipe.execute()

    logger.debug(
        "[flush_manager] enqueue key=%s user=%s session=%s",
        key, user_id, session_id
    )


async def _flush_turns(user_id: str, course_id: str, turns: list[dict]) -> None:
    """批量写入 mem0 + graph_memory（Consumer 调用）。

    Args:
        user_id: 用户 ID
        course_id: 课程 ID
        turns: 对话列表，每项 {"u": user_msg, "a": assistant_msg}
    """
    if not turns:
        return

    logger.info(
        "[flush_manager] FLUSH START user=%s turns=%d",
        user_id, len(turns)
    )

    # 1. 拼接 N 轮对话为 messages 格式
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.get("u", "")})
        messages.append({"role": "assistant", "content": turn.get("a", "")})

    # 2. 调用 mem0.add()（内部 1 次 LLM 提取）
    try:
        from core.memory.mem0_client import get_memory

        m = get_memory()
        result = await m.add(messages, user_id=user_id)
        if result:
            items = result if isinstance(result, list) else result.get("results", [])
            logger.info(
                "[flush_manager] mem0.add complete user=%s stored=%d",
                user_id, len(items) if items else 0
            )
        else:
            logger.info("[flush_manager] mem0.add complete user=%s no new facts", user_id)
    except Exception as e:
        logger.warning("[flush_manager] mem0.add failed user=%s error=%s", user_id, e)

    # 3. graph_memory 更新（已有 3 轮节流，但这里我们也批量调用）
    try:
        from core.memory.graph_memory import update_graphs_from_conversation
        from core.db.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            for turn in turns:
                try:
                    await update_graphs_from_conversation(
                        db,
                        user_id=user_id,
                        course_id=course_id,
                        user_message=turn.get("u", ""),
                        assistant_answer=turn.get("a", ""),
                    )
                except Exception as e:
                    logger.warning(
                        "[flush_manager] graph_memory update failed turn error=%s",
                        e
                    )
        logger.info("[flush_manager] graph_memory update complete user=%s", user_id)
    except Exception as e:
        logger.warning("[flush_manager] graph_memory update failed user=%s error=%s", user_id, e)

    logger.info(
        "[flush_manager] FLUSH COMPLETE user=%s turns=%d",
        user_id, len(turns)
    )


async def scan_and_flush(r: aioredis.Redis, max_turns: int, idle_timeout: float) -> int:
    """扫描 Redis 并执行 flush（ARQ cron job 调用）。

    Args:
        r: Redis 连接
        max_turns: 累积 N 轮后 flush
        idle_timeout: 静默 T 秒后 flush

    Returns:
        flush 的 key 数量
    """
    cursor, keys = await r.scan(match=f"{_PREFIX}*", count=200)

    # 过滤出数据 key（排除 :ts 和 :meta 后缀）
    data_keys = [k for k in keys if not k.endswith(_TS_SUFFIX) and not k.endswith(_META_SUFFIX)]

    flushed_count = 0
    now = time.time()

    for key in data_keys:
        try:
            length = await r.llen(key)
            ts_str = await r.get(f"{key}{_TS_SUFFIX}")
            ts = float(ts_str or 0)
            idle = now - ts

            if length >= max_turns or idle >= idle_timeout:
                # 获取所有对话数据
                turns_raw = await r.lrange(key, 0, -1)
                turns = [json.loads(t) for t in turns_raw]

                # 获取元数据
                meta_str = await r.get(f"{key}{_META_SUFFIX}")
                meta = json.loads(meta_str or "{}")

                # 删除 key（包括 :ts 和 :meta）
                await r.delete(key, f"{key}{_TS_SUFFIX}", f"{key}{_META_SUFFIX}")

                # 执行 flush
                await _flush_turns(meta.get("user_id", ""), meta.get("course_id", ""), turns)
                flushed_count += 1

                logger.info(
                    "[flush_manager] cron flush key=%s turns=%d idle=%.1fs",
                    key, len(turns), idle
                )

        except Exception as e:
            logger.warning("[flush_manager] cron flush key=%s error=%s", key, e)

    return flushed_count


async def flush_all_pending(r: aioredis.Redis) -> int:
    """Flush 所有 pending buffer（shutdown 时调用）。

    Args:
        r: Redis 连接

    Returns:
        flush 的 key 数量
    """
    cursor, keys = await r.scan(match=f"{_PREFIX}*", count=500)
    data_keys = [k for k in keys if not k.endswith(_TS_SUFFIX) and not k.endswith(_META_SUFFIX)]

    flushed_count = 0
    for key in data_keys:
        try:
            turns_raw = await r.lrange(key, 0, -1)
            turns = [json.loads(t) for t in turns_raw]

            meta_str = await r.get(f"{key}{_META_SUFFIX}")
            meta = json.loads(meta_str or "{}")

            await r.delete(key, f"{key}{_TS_SUFFIX}", f"{key}{_META_SUFFIX}")

            await _flush_turns(meta.get("user_id", ""), meta.get("course_id", ""), turns)
            flushed_count += 1

        except Exception as e:
            logger.warning("[flush_manager] flush_all key=%s error=%s", key, e)

    logger.info("[flush_manager] flush_all_pending complete flushed=%d", flushed_count)
    return flushed_count


async def stats() -> dict:
    """返回 Redis buffer 状态（用于监控/调试）。"""
    r = _get_redis()
    cursor, keys = await r.scan(match=f"{_PREFIX}*", count=500)
    data_keys = [k for k in keys if not k.endswith(_TS_SUFFIX) and not k.endswith(_META_SUFFIX)]

    buffers = {}
    for key in data_keys:
        length = await r.llen(key)
        ts_str = await r.get(f"{key}{_TS_SUFFIX}")
        ts = float(ts_str or 0)
        idle = time.time() - ts
        buffers[key] = {"turns": length, "idle_seconds": idle}

    return {
        "total_buffers": len(data_keys),
        "buffers": buffers,
    }


# 兼容旧接口的 wrapper
class MemoryFlushManager:
    """兼容旧接口的 wrapper（内部调用 Redis 函数）。"""

    def __init__(self, max_turns: int = 3, idle_timeout: float = 120.0):
        self._max_turns = max_turns
        self._idle_timeout = idle_timeout

    async def enqueue(
        self,
        user_id: str,
        session_id: str,
        course_id: str,
        user_msg: str,
        assistant_msg: str,
    ) -> None:
        """兼容旧接口。"""
        await enqueue(user_id, session_id, course_id, user_msg, assistant_msg)

    async def flush_all(self) -> None:
        """兼容旧接口（shutdown 时调用）。"""
        await flush_all_pending(_get_redis())

    async def flush_user(self, user_id: str) -> None:
        """手动 flush 指定用户的所有 buffer。"""
        r = _get_redis()
        pattern = f"{_PREFIX}{user_id}:*"
        if ":" not in user_id:
            pattern = f"{_PREFIX}{user_id}*"

        cursor, keys = await r.scan(match=pattern, count=100)
        data_keys = [k for k in keys if not k.endswith(_TS_SUFFIX) and not k.endswith(_META_SUFFIX)]

        for key in data_keys:
            try:
                turns_raw = await r.lrange(key, 0, -1)
                turns = [json.loads(t) for t in turns_raw]
                meta_str = await r.get(f"{key}{_META_SUFFIX}")
                meta = json.loads(meta_str or "{}")

                await r.delete(key, f"{key}{_TS_SUFFIX}", f"{key}{_META_SUFFIX}")
                await _flush_turns(meta.get("user_id", ""), meta.get("course_id", ""), turns)
            except Exception as e:
                logger.warning("[flush_manager] flush_user key=%s error=%s", key, e)

    def stats(self) -> dict:
        """同步 wrapper（不推荐，改用 async stats()）。"""
        import asyncio
        return asyncio.run(stats())


# 全局单例（兼容旧代码）
_flush_manager: MemoryFlushManager | None = None


def get_flush_manager() -> MemoryFlushManager:
    """返回全局 MemoryFlushManager 单例。"""
    global _flush_manager
    if _flush_manager is None:
        from settings.base import get_settings

        settings = get_settings()
        _flush_manager = MemoryFlushManager(
            max_turns=settings.mem0_flush_max_turns,
            idle_timeout=settings.mem0_flush_idle_timeout,
        )
        logger.info(
            "[flush_manager] initialized max_turns=%d idle_timeout=%.1fs",
            settings.mem0_flush_max_turns,
            settings.mem0_flush_idle_timeout,
        )
    return _flush_manager


__all__ = [
    "MemoryFlushManager",
    "get_flush_manager",
    "enqueue",
    "scan_and_flush",
    "flush_all_pending",
    "_flush_turns",
    "stats",
]