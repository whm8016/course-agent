"""LightRAG 实例池管理（LRU 缓存）。

从 lightrag_engine.py 提取的实例管理逻辑，负责：
- LRU 实例缓存（容量控制、淘汰）
- 实例初始化与销毁
- 索引锁管理（防止并发索引冲突）
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

from settings import get_settings
LIGHTRAG_LRU_CAPACITY = get_settings().lightrag_lru_capacity_scaled
LIGHTRAG_MAX_ASYNC = get_settings().lightrag.max_async
LIGHTRAG_WORKDIR = get_settings().paths.lightrag_workdir

logger = logging.getLogger(__name__)

# ── 全局状态（模块级，与原 lightrag_engine.py 保持一致）──────────────────────────────

_instances: OrderedDict[str, Any] = OrderedDict()
_instances_lock: asyncio.Lock | None = None  # 懒初始化，避免 import 时无 event loop
_index_locks: dict[str, asyncio.Lock] = {}

# 索引签名缓存（用于判断是否需要重新索引）
_index_signatures: dict[str, tuple[str, ...]] = {}
_last_auto_index_at: dict[str, float] = {}
_AUTO_INDEX_STATE_DIR = Path(LIGHTRAG_WORKDIR) / ".auto_index_state"
_AUTO_INDEX_LOCK_DIR = Path(LIGHTRAG_WORKDIR) / ".auto_index_locks"


def _get_instances_lock() -> asyncio.Lock:
    """获取全局实例锁（懒初始化）。"""
    global _instances_lock
    if _instances_lock is None:
        _instances_lock = asyncio.Lock()
    return _instances_lock


def _workspace_name(course_id: str) -> str:
    """生成 LightRAG workspace 名称。"""
    return f"course_{course_id}"


def get_instance_count() -> int:
    """返回当前实例数量。"""
    return len(_instances)


def get_instances() -> OrderedDict[str, Any]:
    """返回实例字典（仅供内部使用）。"""
    return _instances


async def evict_oldest() -> str | None:
    """淘汰最旧的实例。

    Returns:
        被淘汰的 course_id，或 None（无实例可淘汰）
    """
    if not _instances:
        return None
    evicted_id, evicted_rag = _instances.popitem(last=False)
    if hasattr(evicted_rag, "finalize_storages"):
        try:
            await evicted_rag.finalize_storages()
        except Exception:
            pass
    logger.info(
        "LightRAG LRU evict course=%s capacity=%d",
        evicted_id, LIGHTRAG_LRU_CAPACITY,
    )
    return evicted_id


async def _get_instance(course_id: str) -> Any:
    """获取 LightRAG 实例（LRU 缓存）。

    这是从原 lightrag_engine.py 提取的核心逻辑，内部自动引用
    llm_adapter.py 的 llm_func / embedding_func。

    Args:
        course_id: 课程 ID

    Returns:
        LightRAG 实例

    Raises:
        RuntimeError: LightRAG 不可用
    """
    # 延迟导入避免循环依赖
    from core.rag.lightrag.llm_adapter import (
        is_lightrag_available,
        _llm_model_func,
        _embedding_func,
    )

    # 导入 LightRAG（延迟导入避免循环依赖）
    try:
        from lightrag import LightRAG
    except Exception as exc:
        raise RuntimeError(f"LightRAG 依赖不可用: {exc}")

    ok, reason = is_lightrag_available()
    if not ok:
        raise RuntimeError(reason)

    lock = _get_instances_lock()
    async with lock:
        if course_id in _instances:
            _instances.move_to_end(course_id)  # LRU hit
            return _instances[course_id]

        # 淘汰最旧实例直到容量满足
        while len(_instances) >= LIGHTRAG_LRU_CAPACITY:
            await evict_oldest()

        Path(LIGHTRAG_WORKDIR).mkdir(parents=True, exist_ok=True)

        _extra_kwargs: dict[str, Any] = {}
        import inspect
        _sig = inspect.signature(LightRAG.__init__)
        if "llm_model_max_async" in _sig.parameters:
            _extra_kwargs["llm_model_max_async"] = LIGHTRAG_MAX_ASYNC

        # Rerank：DashScope gte-rerank-v2；有 EMBEDDING__API_KEY（或等价凭证）时挂载，否则跳过
        if "rerank_model_func" in _sig.parameters:
            from core.rag.lightrag.rerank_adapter import build_rerank_func
            _rerank_func = build_rerank_func()
            if _rerank_func is not None:
                _extra_kwargs["rerank_model_func"] = _rerank_func

        rag = LightRAG(
            working_dir=LIGHTRAG_WORKDIR,
            workspace=_workspace_name(course_id),
            llm_model_func=_llm_model_func,
            embedding_func=_embedding_func,
            **_extra_kwargs,
        )
        await rag.initialize_storages()
        _instances[course_id] = rag

        logger.info(
            "LightRAG LRU load course=%s slots=%d/%d workspace=%s",
            course_id, len(_instances), LIGHTRAG_LRU_CAPACITY, _workspace_name(course_id),
        )
        return rag


# ── 索引锁管理─────────────────────────────────────────────────────


def get_index_lock(course_id: str) -> asyncio.Lock:
    """获取课程索引锁（防止并发索引冲突）。"""
    if course_id not in _index_locks:
        _index_locks[course_id] = asyncio.Lock()
    return _index_locks[course_id]


def clear_index_lock(course_id: str) -> None:
    """清除索引锁。"""
    if course_id in _index_locks:
        del _index_locks[course_id]


# ── 签名缓存（用于判断是否需要重新索引）───────────────────────────────────────


def _build_signature(file_paths: list[str]) -> tuple[str, ...]:
    """构建文件签名（用于判断是否需要重新索引）。"""
    signature: list[str] = []
    for file_path in sorted(file_paths):
        path = Path(file_path)
        stat = path.stat()
        signature.append(f"{file_path}|{stat.st_mtime_ns}|{stat.st_size}")
    return tuple(signature)


def get_cached_signature(course_id: str) -> tuple[str, ...] | None:
    """获取缓存的签名。"""
    return _index_signatures.get(course_id)


def set_cached_signature(course_id: str, signature: tuple[str, ...]) -> None:
    """设置缓存的签名。"""
    _index_signatures[course_id] = signature


# ── 重新索引前清场 + 跨进程分布式锁 ──────────────────────────────────────────


async def purge_course_workspace(course_id: str) -> None:
    """清空某课程的 LightRAG 工作区（实例池缓存 + 磁盘数据）。

    重新索引前调用，避免旧文档残留导致 ainsert 把整批判为 "Duplicate document"、
    最终堆在 failed entries。仅用于全新索引（resume_from_chunk==0）；续传绝不能
    调用，否则抹掉已索引进度。
    """
    ws_dir = Path(LIGHTRAG_WORKDIR) / _workspace_name(course_id)

    # 先从实例池移除并 finalize storage，释放文件句柄——否则删目录后旧实例仍
    # 持有句柄，后续写操作可能把数据写回刚清空的目录。
    async with _get_instances_lock():
        rag = _instances.pop(course_id, None)
    if rag is not None:
        try:
            await rag.finalize_storages()
        except Exception:
            logger.warning("finalize_storages 失败 course=%s", course_id, exc_info=True)

    if ws_dir.exists():
        shutil.rmtree(ws_dir, ignore_errors=True)
        logger.info("已清空 LightRAG 工作区 course=%s dir=%s", course_id, ws_dir)


# 分布式索引锁：跨 worker 进程互斥（asyncio.Lock 只在单进程内有效，多容器/多进程
# 部署下护不住）。TTL + 续约守护：索引可达数小时，靠 renew_task 反复 extend 防止
# 锁过期被别的 worker 抢走；持有者崩溃则 TTL 到期自动释放，不留死锁。
_INDEX_DLOCK_TTL = 3600  # 单次续约周期 1 小时
_INDEX_DLOCK_PREFIX = "indexing:dlock:"


async def acquire_index_dlock(course_id: str):
    """获取 course 级 Redis 分布式锁。返回 (lock, renew_task)；被占返回 (None, None)。"""
    from core.db.cache import _get_pool
    redis = _get_pool()
    lock = redis.lock(f"{_INDEX_DLOCK_PREFIX}{course_id}", timeout=_INDEX_DLOCK_TTL)
    if not await lock.acquire(blocking=False):
        return None, None

    async def _renew() -> None:
        try:
            while True:
                await asyncio.sleep(_INDEX_DLOCK_TTL / 3)
                try:
                    await lock.extend(_INDEX_DLOCK_TTL)
                except Exception:
                    logger.warning("索引锁续约失败 course=%s", course_id, exc_info=True)
        except asyncio.CancelledError:
            pass

    renew = asyncio.create_task(_renew())
    return lock, renew


async def release_index_dlock(lock, renew) -> None:
    """释放分布式锁并取消续约守护。"""
    if renew is not None:
        renew.cancel()
        try:
            await renew
        except (asyncio.CancelledError, Exception):
            pass
    if lock is not None:
        try:
            await lock.release()
        except Exception:
            # 锁可能已过 TTL 被别人取走，release 找不到自己的 token，属正常
            logger.debug("索引锁释放失败（可能已过期）", exc_info=True)


__all__ = [
    "_get_instance",
    "_get_instances_lock",
    "_workspace_name",
    "get_instance_count",
    "get_instances",
    "evict_oldest",
    "get_index_lock",
    "clear_index_lock",
    "_build_signature",
    "get_cached_signature",
    "set_cached_signature",
    "_instances",
    "_index_locks",
    "_index_signatures",
    "purge_course_workspace",
    "acquire_index_dlock",
    "release_index_dlock",
]