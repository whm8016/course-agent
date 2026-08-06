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
# H-9: 待落盘的对话数据本身是业务数据，TTL 仅作"防泄漏兜底"。
# 原 600s(10min) 太短——worker 宕机/重启超过 10 分钟即静默丢数据。
# 改为 86400s(24h)：足够覆盖夜间故障窗口；数据量小（每条 ~百字节 JSON），
# 即便积压也不会撑爆 Redis。
_TTL = 86400
# M-13: per-key 分布式锁的后缀与超时。锁只在 flush 期间持有，用于多 worker 互斥。
# 30s 远大于单次 flush 耗时，又能在持有者崩溃后自动释放，避免死锁。
_LOCK_SUFFIX = ":lock"
_LOCK_TTL = 30


_redis_pool: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    """返回模块级 Redis 连接池（懒加载，复用同一个 pool，不需要每次 aclose）。"""
    global _redis_pool
    if _redis_pool is None:
        from settings import get_settings
        REDIS_URL = get_settings().db.redis_url.get_secret_value()
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


async def _flush_turns(user_id: str, course_id: str, turns: list[dict]) -> bool:
    """批量写入 mem0 + graph_memory（Consumer 调用）。

    Args:
        user_id: 用户 ID
        course_id: 课程 ID
        turns: 对话列表，每项 {"u": user_msg, "a": assistant_msg}

    Returns:
        True = 关键持久化（mem0.add）成功，可安全删 key；False = mem0 写失败，调用方
        （_flush_one）须保留 Redis key 等下次重试（Phase 1 止血，避免删 key 丢数据）。
        graph_memory 失败属非致命（派生数据），不影响返回值。
    """
    if not turns:
        return True  # 空批 = 无需写 = 安全删 key（与旧「return 后无条件删」语义一致）

    logger.info(
        "[flush_manager] FLUSH START user=%s turns=%d",
        user_id, len(turns)
    )

    # 1. 拼接 N 轮对话为 messages 格式
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.get("u", "")})
        messages.append({"role": "assistant", "content": turn.get("a", "")})

    # 2. 调用 mem0.add()（内部 1 次 LLM 提取）—— 关键持久化
    mem0_ok = True
    try:
        from core.memory.mem0_client import get_memory

        m = get_memory()
        result = await m.add(messages, user_id=user_id, metadata={"course_id": course_id})
        if result:
            items = result if isinstance(result, list) else result.get("results", [])
            logger.info(
                "[flush_manager] mem0.add complete user=%s stored=%d",
                user_id, len(items) if items else 0
            )
        else:
            logger.info("[flush_manager] mem0.add complete user=%s no new facts", user_id)
    except Exception as e:
        # Phase 1 止血：mem0 写失败不再静默吞——标记 False 让 _flush_one 保留 key 重试。
        # 原实现只 warning 不外抛，_flush_one 据此误判成功并删 key → 永久丢数据（架空 H-7）。
        mem0_ok = False
        logger.warning("[flush_manager] mem0.add failed user=%s error=%s (key retained for retry)", user_id, e)

    # 3. graph_memory 更新（已有 6 轮节流，这里批量调用）—— 仅在 mem0 成功时执行：
    # mem0 失败会整批重试，此时跑 graph 等于白烧一次 LLM 提取（重试时还会再跑一遍）。
    if mem0_ok:
        try:
            from core.memory.graph_memory import update_graphs_from_conversation
            from core.db.database import AsyncSessionLocal

            # M-12: AsyncSessionLocal 退出时默认 rollback（不 commit），
            # 而 update_graphs_from_conversation → save_graphs 内部只 execute(update)，
            # 不自己 commit。原代码漏 commit → graph 更新全部静默丢失。
            # 在整个循环结束后统一 commit 一次（批量语义，单事务）。
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
                try:
                    await db.commit()
                except Exception as e:
                    # commit 失败属可恢复：数据仍在 Redis key 里，下次 cron 会重试。
                    # 这里 rollback 后退出 with（防止脏事务残留）。
                    logger.warning("[flush_manager] graph_memory commit failed user=%s error=%s", user_id, e)
                    await db.rollback()
            logger.info("[flush_manager] graph_memory update complete user=%s", user_id)
        except Exception as e:
            logger.warning("[flush_manager] graph_memory update failed user=%s error=%s", user_id, e)

    logger.info(
        "[flush_manager] FLUSH COMPLETE user=%s turns=%d mem0_ok=%s",
        user_id, len(turns), mem0_ok
    )
    return mem0_ok


async def _scan_all_keys(r: aioredis.Redis, match: str, count: int = 200) -> list[str]:
    """循环 SCAN 直到 cursor 归 0，返回所有匹配 key。

    H-8: 原实现 `cursor, keys = await r.scan(...)` 只调一次，依赖单次返回。
    Redis SCAN 在大 keyspace 下会分批返回（cursor != 0 表示还有），
    单次调用只拿到第一批（~count 个），导致超量的 buffer key 被漏扫、永不 flush。
    这里按 `while cursor:` 持续翻页直到游标归 0。
    """
    all_keys: list[str] = []
    cursor: int | bytes = 0
    while True:
        cursor, batch = await r.scan(cursor=cursor, match=match, count=count)
        # key 可能是 bytes：worker.py 的 cron job 优先复用 ARQ 注入的 ctx["redis"]，
        # 那个连接池（arq.create_pool）不带 decode_responses=True，与本模块自建的
        # _get_redis()（decode_responses=True）不一致。这里统一解码为 str，
        # 否则下游 k.endswith(_TS_SUFFIX) 会因 bytes.endswith(str) 报 TypeError。
        all_keys.extend(k.decode() if isinstance(k, bytes) else k for k in batch)
        # cursor 可能是 bytes（部分客户端）或 int；统一以"是否归 0"判断。
        c = int(cursor) if cursor else 0
        if c == 0:
            break
    return all_keys


async def _flush_one(
    r: aioredis.Redis,
    key: str,
    turns: list[dict],
    meta: dict,
) -> bool:
    """带 per-key 互斥锁的安全 flush。

    H-7 + M-13 的核心修复点。流程（顺序至关重要）：
      1. SET NX 抢 per-key 锁（拿不到说明别的 worker 正在 flush，跳过避免重复落盘）
      2. flush 成功后才删 Redis key（原代码先删后写，flush 失败即永久丢数据）
      3. finally 释放锁（防崩溃残留：锁带 TTL，崩溃也会自动过期）

    Returns:
        True = 已 flush（或数据已被本 worker 处理）；False = 抢锁失败被跳过。
    """
    lock_key = f"{key}{_LOCK_SUFFIX}"

    # 1. 抢 per-key 锁（SET NX + EX，原子）
    got_lock = await r.set(lock_key, "1", ex=_LOCK_TTL, nx=True)
    if not got_lock:
        logger.info("[flush_manager] flush_one SKIP key=%s reason=locked-by-other", key)
        return False

    try:
        # 2. 先 flush 成功，再删 key（H-7：flush 失败则保留 key 等下次重试）
        ok = await _flush_turns(meta.get("user_id", ""), meta.get("course_id", ""), turns)
        if not ok:
            # Phase 1 止血：关键写（mem0）失败 → 保留 key 等下次 cron 重试，绝不删 key 丢数据。
            # 区别于「_flush_turns 抛异常」路径：这里是 mem0 内部失败被捕获、显式返回 False。
            logger.warning(
                "[flush_manager] flush_one KEEP key=%s reason=critical-write-failed-will-retry", key
            )
            return False

        # flush 成功后才删除数据 key（含 :ts 和 :meta），数据安全落盘
        await r.delete(key, f"{key}{_TS_SUFFIX}", f"{key}{_META_SUFFIX}")
        return True
    finally:
        # 3. 释放锁（best-effort；锁有 TTL 兜底，删失败也不致死锁）
        try:
            await r.delete(lock_key)
        except Exception as e:
            logger.warning("[flush_manager] flush_one unlock failed key=%s error=%s", key, e)


async def scan_and_flush(r: aioredis.Redis, max_turns: int, idle_timeout: float) -> int:
    """扫描 Redis 并执行 flush（ARQ cron job 调用）。

    Args:
        r: Redis 连接
        max_turns: 累积 N 轮后 flush
        idle_timeout: 静默 T 秒后 flush

    Returns:
        flush 的 key 数量
    """
    # H-8: 循环 SCAN 翻页，避免单次返回导致漏扫
    keys = await _scan_all_keys(r, match=f"{_PREFIX}*", count=200)

    # 过滤出数据 key（排除 :ts / :meta / :lock 后缀）
    data_keys = [
        k for k in keys
        if not k.endswith(_TS_SUFFIX)
        and not k.endswith(_META_SUFFIX)
        and not k.endswith(_LOCK_SUFFIX)
    ]

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

                # 安全 flush（带 per-key 锁 + 先写后删）
                done = await _flush_one(r, key, turns, meta)
                if done:
                    flushed_count += 1

                logger.info(
                    "[flush_manager] cron flush key=%s turns=%d idle=%.1fs done=%s",
                    key, len(turns), idle, done
                )

        except Exception as e:
            # 可恢复：保留 key 等下次 cron 重试（注意：这里若 _flush_one 已抢锁但
            # flush 抛异常，key 不会被删，数据安全；锁由 finally / TTL 释放）
            logger.warning("[flush_manager] cron flush key=%s error=%s", key, e)

    return flushed_count


async def flush_all_pending(r: aioredis.Redis) -> int:
    """Flush 所有 pending buffer（shutdown 时调用）。

    Args:
        r: Redis 连接

    Returns:
        flush 的 key 数量
    """
    keys = await _scan_all_keys(r, match=f"{_PREFIX}*", count=500)
    data_keys = [
        k for k in keys
        if not k.endswith(_TS_SUFFIX)
        and not k.endswith(_META_SUFFIX)
        and not k.endswith(_LOCK_SUFFIX)
    ]

    flushed_count = 0
    for key in data_keys:
        try:
            turns_raw = await r.lrange(key, 0, -1)
            turns = [json.loads(t) for t in turns_raw]

            meta_str = await r.get(f"{key}{_META_SUFFIX}")
            meta = json.loads(meta_str or "{}")

            # 复用安全 flush（H-7 先写后删 + M-13 per-key 锁）
            done = await _flush_one(r, key, turns, meta)
            if done:
                flushed_count += 1

        except Exception as e:
            logger.warning("[flush_manager] flush_all key=%s error=%s", key, e)

    logger.info("[flush_manager] flush_all_pending complete flushed=%d", flushed_count)
    return flushed_count


async def stats() -> dict:
    """返回 Redis buffer 状态（用于监控/调试）。"""
    r = _get_redis()
    keys = await _scan_all_keys(r, match=f"{_PREFIX}*", count=500)
    data_keys = [
        k for k in keys
        if not k.endswith(_TS_SUFFIX)
        and not k.endswith(_META_SUFFIX)
        and not k.endswith(_LOCK_SUFFIX)
    ]

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

        keys = await _scan_all_keys(r, match=pattern, count=100)
        data_keys = [
            k for k in keys
            if not k.endswith(_TS_SUFFIX)
            and not k.endswith(_META_SUFFIX)
            and not k.endswith(_LOCK_SUFFIX)
        ]

        for key in data_keys:
            try:
                turns_raw = await r.lrange(key, 0, -1)
                turns = [json.loads(t) for t in turns_raw]
                meta_str = await r.get(f"{key}{_META_SUFFIX}")
                meta = json.loads(meta_str or "{}")

                # 复用安全 flush（H-7 先写后删 + M-13 per-key 锁）
                await _flush_one(r, key, turns, meta)
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
            max_turns=settings.mem0.flush_max_turns,
            idle_timeout=settings.mem0.flush_idle_timeout,
        )
        logger.info(
            "[flush_manager] initialized max_turns=%d idle_timeout=%.1fs",
            settings.mem0.flush_max_turns,
            settings.mem0.flush_idle_timeout,
        )
    return _flush_manager


__all__ = [
    "MemoryFlushManager",
    "get_flush_manager",
    "enqueue",
    "scan_and_flush",
    "flush_all_pending",
    "_flush_turns",
    "_scan_all_keys",
    "_flush_one",
    "stats",
]