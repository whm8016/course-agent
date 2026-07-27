"""RAG 来源标识工具 —— `::chunk-<idx>` 后缀的单一剥离入口。

摄入端（ingestion）为绕过 LightRAG filename 去重，给每个 chunk 的 file_path 加
`::chunk-<node 索引>` 全局唯一后缀；检索端（retriever/lightrag）展示溯源前必须剥掉，
否则用户会看到「xxx.pdf::chunk-5」。两端原各持一份同构实现（rfind 定位最后一个
`::chunk-` 后截断），统一到此单一真相源——日后调整后缀规则只需改一处。
"""
from __future__ import annotations

_CHUNK_SUFFIX_MARKER = "::chunk-"


def strip_chunk_suffix(source: str) -> str:
    """剥离摄入时加的 `::chunk-<idx>` 后缀，还原真实来源文件路径。

    用 rfind 定位最后一个 `::chunk-` 后截断；无后缀（或出现在开头）时原样返回。
    """
    idx = source.rfind(_CHUNK_SUFFIX_MARKER)
    return source[:idx] if idx > 0 else source
