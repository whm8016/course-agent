"""LightRAG 实例池管理（LRU 缓存）。

从 lightrag_engine.py 提取的实例管理逻辑，负责：
- LRU 实例缓存（容量控制、淘汰）
- 实例初始化与销毁
- 索引锁管理（防止并发索引冲突）
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from config import (
    LIGHTRAG_LRU_CAPACITY,
    LIGHTRAG_MAX_ASYNC,
    LIGHTRAG_WORKDIR,
)

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
]