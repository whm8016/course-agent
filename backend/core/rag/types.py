"""RAG 子系统核心类型定义。

提供 RetrievalResult、IndexResult、ChunkMeta、DocumentFragment 等数据类，
供各后端实现共用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkMeta:
    """文档块元信息。"""
    source_path: str
    start_char: int
    end_char: int
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """检索结果。"""
    content: str
    score: float
    source_chunk: ChunkMeta | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexResult:
    """索引结果摘要。"""
    course_id: str
    files_indexed: int
    chunks_created: int
    status: str = "success"
    error: str | None = None


@dataclass(frozen=True)
class DocumentFragment:
    """文档解析后的文本片段。"""
    text: str
    fragment_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
