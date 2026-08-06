"""RAG Registry —— 工厂 + 门面。

提供统一的后端获取接口，调用方无需硬编码选择后端。
支持：
- register_retriever / register_indexer 注册后端实现
- get_retriever / get_indexer 获取实例
- 根据参数选择后端（当前仅 lightrag）
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
    """注册 Retriever 实现。"""
    _retrievers[name] = cls
    logger.info("RAG registry: registered retriever '%s'", name)


def register_indexer(name: str, cls: type[Indexer]) -> None:
    """注册 Indexer 实现。"""
    _indexers[name] = cls
    logger.info("RAG registry: registered indexer '%s'", name)


# ── 工厂函数 ─────────────────────────────────────────────────────────────────


def get_retriever(backend: str | None = None) -> Retriever:
    """获取 Retriever 实例。"""
    name = backend or "lightrag"

    if name not in _retrievers:
        _auto_register(name)

    if name not in _retrievers:
        raise ValueError(f"RAG retriever '{name}' not registered. Available: {list(_retrievers.keys())}")

    cls = _retrievers[name]
    return cls()


def get_indexer(backend: str | None = None) -> Indexer:
    """获取 Indexer 实例。"""
    name = backend or "lightrag"

    if name not in _indexers:
        _auto_register(name)

    if name not in _indexers:
        raise ValueError(f"RAG indexer '{name}' not registered. Available: {list(_indexers.keys())}")

    cls = _indexers[name]
    return cls()


# ── 自动注册 ─────────────────────────────────────────────────────────────────


def _auto_register(name: str) -> None:
    """自动导入并注册后端实现。

    各后端实现惰性导入（首次 get_retriever/get_indexer 时触发），core 不在模块加载期
    import 重依赖——llamaindex_pg 缺 llama-index-vector-stores-postgres 时 import 失败，
    仅 warning 不阻断 lightrag 默认链路。
    """
    if name == "lightrag":
        try:
            from core.rag.retriever.lightrag import LightRAGRetriever
            from core.rag.indexer.lightrag import LightRAGIndexer
            register_retriever("lightrag", LightRAGRetriever)
            register_indexer("lightrag", LightRAGIndexer)
        except ImportError as e:
            logger.warning("Failed to auto-register lightrag: %s", e)
    elif name == "llamaindex_pg":
        try:
            from core.rag.retriever.llamaindex_pg import LlamaIndexRetriever
            from core.rag.indexer.llamaindex_pg import LlamaIndexIndexer
            register_retriever("llamaindex_pg", LlamaIndexRetriever)
            register_indexer("llamaindex_pg", LlamaIndexIndexer)
        except ImportError as e:
            logger.warning("Failed to auto-register llamaindex_pg: %s", e)


# ── 后端可用性检查 ───────────────────────────────────────────────────────────


def is_backend_available(backend: str) -> tuple[bool, str]:
    """检查后端是否可用。"""
    if backend == "lightrag":
        from core.rag.lightrag import is_lightrag_available
        return is_lightrag_available()
    if backend == "llamaindex_pg":
        from core.rag.llamaindex.pg_store import is_llamaindex_pg_available
        return is_llamaindex_pg_available()

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