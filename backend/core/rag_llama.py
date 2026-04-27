"""
LlamaIndex 检索封装（对齐 retrieve_context 的返回形状）。
"""
from __future__ import annotations

import logging
from pathlib import Path

from config import LLAMA_INDEX_KB_ROOT, TOP_K
from rag_llama.llamaindex_pipeline import LlamaIndexPipeline

logger = logging.getLogger(__name__)


def llamaindex_index_path(course_id: str) -> Path:
    return Path(LLAMA_INDEX_KB_ROOT) / course_id / "llamaindex_storage"


def llamaindex_has_index(course_id: str) -> bool:
    return (llamaindex_index_path(course_id) / "docstore.json").is_file()


async def retrieve_context_llamaindex(
    course_id: str,
    query: str,
    top_k: int | None = None,
) -> dict[str, str]:
    k = top_k if top_k is not None else TOP_K
    pipeline = LlamaIndexPipeline(kb_base_dir=str(LLAMA_INDEX_KB_ROOT))
    result = await pipeline.search(query=query, kb_name=course_id, top_k=k)
    answer = (result.get("answer") or result.get("content") or "").strip()
    return {"answer": answer, "provider": result.get("provider", "llamaindex")}


async def retrieve_chunks_llamaindex(
    course_id: str,
    query: str,
    top_k: int | None = None,
    max_chars_per_chunk: int = 1200,
    max_total_chars: int = 6000,
) -> str:
    """Return raw text chunks as a formatted string for use as RAG context.

    Each chunk is prefixed with its source file name and chunk index so the
    LLM can cite them.  Total length is capped to avoid over-filling context.
    """
    if not llamaindex_has_index(course_id):
        logger.warning("retrieve_chunks_llamaindex: no index for course=%s", course_id)
        return "（LlamaIndex 索引尚未建立，请先在管理端建库）"

    k = top_k if top_k is not None else TOP_K
    pipeline = LlamaIndexPipeline(kb_base_dir=str(LLAMA_INDEX_KB_ROOT))
    result = await pipeline.search(query=query, kb_name=course_id, top_k=k)

    sources: list[dict] = result.get("sources") or []
    if not sources:
        # fallback: content 一整块当成一个 chunk
        content = (result.get("content") or "").strip()
        return content[:max_total_chars] or "（未检索到相关内容）"

    parts: list[str] = []
    total = 0
    for i, src in enumerate(sources, 1):
        file_name = src.get("title") or src.get("source") or f"片段{i}"
        page = src.get("page", "")
        page_tag = f" p.{page}" if page else ""
        text = (src.get("content") or "").strip()
        if not text:
            continue
        text = text[:max_chars_per_chunk]
        header = f"【片段{i}｜{file_name}{page_tag}】"
        chunk = f"{header}\n{text}"
        if total + len(chunk) > max_total_chars:
            break
        parts.append(chunk)
        total += len(chunk)

    logger.info(
        "retrieve_chunks_llamaindex course=%s result=%s query_chars=%d chunks=%d total_chars=%d",
        course_id, result, len(query), len(parts), total,
    )
    return "\n\n".join(parts) if parts else "（未检索到相关内容）"