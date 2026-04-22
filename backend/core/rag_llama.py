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