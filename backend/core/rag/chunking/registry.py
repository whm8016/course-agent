"""切块策略注册表 —— 与检索器后端 registry（core/rag/registry.py）同构的 Strategy Registry。

统一策略签名：(documents, classification, ingest_size) -> (chunks, chunk_sources)。
ingestion._chunk_documents 按 settings.chunking.strategy 查本表分发；新增策略只需
register_chunk_strategy 一行，无需改 _chunk_documents 的分发代码（对齐检索器后端的
register_retriever/get_retriever 风格，消除原先 ingestion 里硬编码的 if/else 分发）。

策略实现因依赖 ingestion 的常量（INGEST_CHUNK_SIZE 等）与 _build_source_prefix，仍留在
ingestion.py 注册；本模块只提供注册机制（dict + register/get + 未注册回退默认策略）。
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.rag.llamaindex.file_routing import FileClassification

logger = logging.getLogger(__name__)

# 统一策略签名：documents + 文件分类 + 切块大小 → (chunks, chunk_sources)
ChunkStrategy = Callable[
    [list, "FileClassification", int], tuple[list[str], list[str]]
]

_chunk_strategies: dict[str, ChunkStrategy] = {}

DEFAULT_STRATEGY = "sentence_splitter"


def register_chunk_strategy(name: str, fn: ChunkStrategy) -> None:
    """注册切块策略实现。"""
    _chunk_strategies[name] = fn
    logger.info("chunking registry: registered strategy '%s'", name)


def get_chunk_strategy(name: str) -> ChunkStrategy:
    """按名取策略；未注册时回退默认 sentence_splitter（保持默认行为）。"""
    if name not in _chunk_strategies:
        logger.warning(
            "chunk strategy '%s' not registered, fallback to '%s'",
            name,
            DEFAULT_STRATEGY,
        )
        name = DEFAULT_STRATEGY
    return _chunk_strategies[name]


__all__ = [
    "ChunkStrategy",
    "DEFAULT_STRATEGY",
    "register_chunk_strategy",
    "get_chunk_strategy",
]
