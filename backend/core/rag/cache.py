"""
RAG 查询缓存模块

提供以下功能：
1. 基于 Redis 的查询结果缓存
2. LRU 淘汰策略
3. 缓存失效机制
4. 缓存统计

原理说明：
- 使用查询的语义哈希作为缓存键
- 设置 TTL 防止缓存无限增长
- 支持手动失效和自动失效
- 记录命中率等统计信息用于监控
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 全局缓存统计
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "errors": 0,
}


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    errors: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": f"{self.hit_rate:.2%}",
            "total_requests": self.total,
        }


def get_cache_stats() -> CacheStats:
    """获取缓存统计"""
    return CacheStats(
        hits=_cache_stats["hits"],
        misses=_cache_stats["misses"],
        errors=_cache_stats["errors"],
    )


def _compute_query_hash(course_id: str, query: str, top_k: int) -> str:
    """
    计算查询的哈希值作为缓存键

    原理：
    - 将 course_id、query 和 top_k 组合
    - 使用 SHA-256 生成固定长度的哈希
    - 确保相同查询得到相同哈希
    """
    raw = json.dumps({
        "course_id": course_id,
        "query": query,
        "top_k": top_k,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class RAGCache:
    """
    RAG 查询结果缓存

    使用 Redis 作为存储后端，支持：
    - 自动过期（TTL）
    - 缓存失效
    - 统计信息
    """

    def __init__(
        self,
        redis_client=None,
        ttl_seconds: int = 3600,
        max_results: int = 1000,
    ):
        """
        初始化缓存

        参数：
            redis_client: Redis 异步客户端，如果为 None 则缓存被禁用
            ttl_seconds: 缓存过期时间（秒），默认 1 小时
            max_results: 单个缓存结果的最大字符数
        """
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._max_results = max_results

    def _prefix(self, course_id: str) -> str:
        """生成缓存键前缀"""
        return f"rag_cache:{course_id}"

    async def get(
        self,
        course_id: str,
        query: str,
        top_k: int,
    ) -> list[dict] | None:
        """
        获取缓存的查询结果

        参数：
            course_id: 课程 ID
            query: 查询文本
            top_k: 返回结果数量

        返回：
            缓存的结果列表，如果未命中返回 None
        """
        if not self._redis:
            return None

        try:
            key = f"{self._prefix(course_id)}:{_compute_query_hash(course_id, query, top_k)}"

            # 尝试从 Redis 获取
            cached = await self._redis.get(key)
            if cached:
                _cache_stats["hits"] += 1
                logger.debug(f"RAG cache HIT: {course_id}/{query[:50]}...")
                return json.loads(cached)
            else:
                _cache_stats["misses"] += 1
                logger.debug(f"RAG cache MISS: {course_id}/{query[:50]}...")
                return None

        except Exception as e:
            _cache_stats["errors"] += 1
            logger.warning(f"RAG cache error: {e}")
            return None

    async def set(
        self,
        course_id: str,
        query: str,
        top_k: int,
        results: list[dict],
    ) -> bool:
        """
        存储查询结果到缓存

        参数：
            course_id: 课程 ID
            query: 查询文本
            top_k: 返回结果数量
            results: 查询结果列表

        返回：
            是否存储成功
        """
        if not self._redis:
            return False

        try:
            key = f"{self._prefix(course_id)}:{_compute_query_hash(course_id, query, top_k)}"

            # 序列化结果，限制大小
            serialized = json.dumps(results, ensure_ascii=False)
            if len(serialized) > self._max_results * 1024:
                # 结果太大，截断
                logger.warning(f"RAG cache: result too large ({len(serialized)} bytes), truncating")
                results = results[:5]  # 只保留前 5 条
                serialized = json.dumps(results, ensure_ascii=False)

            # 存储到 Redis，设置过期时间
            await self._redis.setex(key, self._ttl, serialized)
            logger.debug(f"RAG cache SET: {course_id}/{query[:50]}... (ttl={self._ttl}s)")
            return True

        except Exception as e:
            _cache_stats["errors"] += 1
            logger.warning(f"RAG cache set error: {e}")
            return False

    async def invalidate(self, course_id: str | None = None) -> int:
        """
        失效缓存

        参数：
            course_id: 如果指定，只失效该课程的缓存；如果为 None，失效所有缓存

        返回：
            失效的键数量
        """
        if not self._redis:
            return 0

        try:
            if course_id:
                pattern = f"{self._prefix(course_id)}:*"
            else:
                pattern = "rag_cache:*"

            # 使用 SCAN 查找匹配的键（比 KEYS 更安全）
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100,
                )
                if keys:
                    await self._redis.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break

            logger.info(f"RAG cache invalidated: {deleted} keys (course_id={course_id})")
            return deleted

        except Exception as e:
            logger.warning(f"RAG cache invalidate error: {e}")
            return 0

    async def get_or_set(
        self,
        course_id: str,
        query: str,
        top_k: int,
        fetch_func,
    ) -> list[dict]:
        """
        获取或设置缓存（缓存友好的 fetch）

        原理：
        1. 先检查缓存
        2. 如果命中，返回缓存结果
        3. 如果未命中，调用 fetch_func 获取结果
        4. 将结果存入缓存
        5. 返回结果

        这是一个便捷方法，封装了 get -> fetch -> set 的流程

        参数：
            course_id: 课程 ID
            query: 查询文本
            top_k: 返回结果数量
            fetch_func: 获取结果的异步函数，签名为 async def () -> list[dict]

        返回：
            查询结果列表
        """
        # 尝试从缓存获取
        cached = await self.get(course_id, query, top_k)
        if cached is not None:
            return cached

        # 缓存未命中，获取新结果
        results = await fetch_func()

        # 存入缓存（不阻塞主流程）
        try:
            await self.set(course_id, query, top_k, results)
        except Exception as e:
            logger.warning(f"Failed to cache RAG results: {e}")

        return results


# 全局缓存实例（延迟初始化）
_rag_cache: RAGCache | None = None


def get_rag_cache() -> RAGCache:
    """获取全局 RAG 缓存实例"""
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = RAGCache()
    return _rag_cache


async def init_rag_cache(redis_url: str, ttl_seconds: int = 3600) -> RAGCache:
    """
    初始化 RAG 缓存

    参数：
        redis_url: Redis 连接 URL
        ttl_seconds: 缓存过期时间

    返回：
        RAGCache 实例
    """
    global _rag_cache
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        # 测试连接
        await redis_client.ping()
        _rag_cache = RAGCache(
            redis_client=redis_client,
            ttl_seconds=ttl_seconds,
        )
        logger.info(f"RAG cache initialized with Redis: {redis_url}, TTL={ttl_seconds}s")
        return _rag_cache
    except Exception as e:
        logger.warning(f"Failed to initialize RAG cache with Redis: {e}")
        logger.info("RAG cache disabled, continuing without caching")
        _rag_cache = RAGCache(redis_client=None)
        return _rag_cache


def set_rag_cache(cache: RAGCache | None) -> None:
    """设置全局 RAG 缓存实例。"""
    global _rag_cache
    _rag_cache = cache
