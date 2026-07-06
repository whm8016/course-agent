"""RAG 子系统公共 API。

提供统一的检索和索引接口，调用方通过此模块导入，无需感知具体后端。

使用示例：
    from core.rag import get_retriever, get_indexer

    retriever = get_retriever("lightrag")
    results = await retriever.retrieve(course_id, query)

    indexer = get_indexer("lightrag")
    result = await indexer.index(course_id, file_paths)
"""
from __future__ import annotations

from core.rag.types import (
    RetrievalResult,
    IndexResult,
    ChunkMeta,
    DocumentFragment,
)
from core.rag.registry import (
    get_retriever,
    get_indexer,
    register_retriever,
    register_indexer,
    is_backend_available,
    list_available_backends,
)

# ── 向后兼容：导出原 lightrag_engine 的公共 API（deprecated）───────────────────
# 调用方仍可使用旧导入路径，待 Phase 5 迁移完成后移除

from core.rag.lightrag import (
    is_lightrag_available,
    get_course_entities,
    get_course_relations,
)

__all__ = [
    # Types
    "RetrievalResult",
    "IndexResult",
    "ChunkMeta",
    "DocumentFragment",
    # Registry
    "get_retriever",
    "get_indexer",
    "register_retriever",
    "register_indexer",
    "is_backend_available",
    "list_available_backends",
    # LightRAG internals (deprecated, will remove in Phase 5)
    "is_lightrag_available",
    "get_course_entities",
    "get_course_relations",
]