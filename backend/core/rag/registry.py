"""RAG Registry —— 工厂 + 门面。

提供统一的后端获取接口，调用方无需硬编码选择 LightRAG/LlamaIndex/Chroma。
支持：
- register_retriever / register_indexer 注册后端实现
- get_retriever / get_indexer 获取实例
- 根据 settings.RAG_BACKEND 或参数选择后端
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.rag.retriever.base import Retriever
    from core.rag.indexer.base import Indexer

logger = logging.getLogger(__name__)

# ── 注册表 ───────────────────────────────────────────────────────────────────

_retrievers: dict[str, type[Retriever]] = {}
_indexers: dict[str, type[Indexer]] = {}


def register_retriever(name: str, cls: type[Retriever]) -> None:
    """注册 Retriever 实现。

    Args:
        name: 后端名称（如 "lightrag", "llamaindex", "chroma"）
        cls: Retriever 实现类
    """
    _retrievers[name] = cls
    logger.info("RAG registry: registered retriever '%s'", name)


def register_indexer(name: str, cls: type[Indexer]) -> None:
    """注册 Indexer 实现。

    Args:
        name: 后端名称
        cls: Indexer 实现类
    """
    _indexers[name] = cls
    logger.info("RAG registry: registered indexer '%s'", name)


# ── 工厂函数 ─────────────────────────────────────────────────────────────────


def get_retriever(backend: str | None = None) -> Retriever:
    """获取 Retriever 实例。

    Args:
        backend: 后端名称，不传则从 settings.RAG_BACKEND 读取

    Returns:
        Retriever 实例

    Raises:
        ValueError: 后端未注册
    """
    from config import RAG_BACKEND

    name = backend or RAG_BACKEND or "lightrag"

    if name not in _retrievers:
        # 尝试自动导入并注册
        _auto_register(name)

    if name not in _retrievers:
        raise ValueError(f"RAG retriever '{name}' not registered. Available: {list(_retrievers.keys())}")

    cls = _retrievers[name]
    return cls()


def get_indexer(backend: str | None = None) -> Indexer:
    """获取 Indexer 实例。

    Args:
        backend: 后端名称，不传则从 settings.RAG_BACKEND 读取

    Returns:
        Indexer 实例

    Raises:
        ValueError: 后端未注册
    """
    from config import RAG_BACKEND

    name = backend or RAG_BACKEND or "lightrag"

    if name not in _indexers:
        # 尝试自动导入并注册
        _auto_register(name)

    if name not in _indexers:
        raise ValueError(f"RAG indexer '{name}' not registered. Available: {list(_indexers.keys())}")

    cls = _indexers[name]
    return cls()


# ── 自动注册 ─────────────────────────────────────────────────────────────────


def _auto_register(name: str) -> None:
    """自动导入并注册后端实现。"""
    if name == "lightrag":
        try:
            from core.rag.retriever.lightrag import LightRAGREtriever
            register_retriever("lightrag", LightRAGREtriever)
            # Indexer 暂未实现，后续补充
        except ImportError as e:
            logger.warning("Failed to auto-register lightrag retriever: %s", e)

    elif name == "llamaindex":
        try:
            # LlamaIndex Retriever 暂未实现
            pass
        except ImportError as e:
            logger.warning("Failed to auto-register llamaindex retriever: %s", e)

    elif name == "chroma":
        try:
            # Chroma Retriever 暂未实现
            pass
        except ImportError as e:
            logger.warning("Failed to auto-register chroma retriever: %s", e)


# ── 后端可用性检查 ───────────────────────────────────────────────────────────


def is_backend_available(backend: str) -> tuple[bool, str]:
    """检查后端是否可用。

    Args:
        backend: 后端名称

    Returns:
        (is_available, error_message) 元组
    """
    if backend == "lightrag":
        from core.rag.lightrag import is_lightrag_available
        return is_lightrag_available()

    # 其他后端暂未实现
    return False, f"Backend '{backend}' not implemented"


def list_available_backends() -> list[str]:
    """列出所有可用的后端。"""
    available = []
    for name in _retrievers.keys():
        ok, _ = is_backend_available(name)
        if ok:
            available.append(name)
    return available


__all__ = [
    "register_retriever",
    "register_indexer",
    "get_retriever",
    "get_indexer",
    "is_backend_available",
    "list_available_backends",
]