"""Contextual Chunking（Anthropic Contextual Retrieval）。

给每个 chunk 注入一段「它在整篇文档里的位置/主题背景」前缀，让 embedding 和 BM25
都能把"脱离上下文就难匹配"的片段（代词指代、章节局部、公式序号）和文档主题关联起来。

Anthropic 实证（2024.09）：仅加 contextual prefix，检索失败率 5.7%→3.7%（-35%）；
配合 BM25 + rerank 可降到 1.9%（-67%）。

成本控制：同一文档的所有 chunk 共享 document_text 前缀 → 天然适配 provider 侧的
prompt caching（相同前缀命中缓存）；用便宜快速的 fast_model（qwen-turbo），不动主力
模型。结果按 chunk 内容 hash 缓存，重复索引不重复调用 LLM。

注意：prompt caching 的实际命中率取决于 provider（dashscope/qwen 与 Anthropic 机制
不同），不要假设固定降本比例——把它当作"每 chunk 一次 fast_model 调用"来评估成本。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document>
{whole_document}
</document>
这是从上述文档中提取的一个片段：
<chunk>
{chunk_content}
</chunk>
请用中文写 1-2 句简短描述，说明这个片段在文档中的位置和主题背景。
只输出描述本身，不要重复片段内容，不要加“这个片段”之类的前缀。"""

# document_text 截断：contextual prompt 里整篇文档作背景，过长爆 token 且稀释重点。
# 8000 字符 ≈ 2000-3000 token，兼顾覆盖与成本。
_DOC_TRUNCATE = 8000
_DEFAULT_BATCH = 10
_DEFAULT_CONCURRENCY = 8


def _batches(seq: list[int], n: int):
    """手动分批（避免 itertools.batched 的 py3.12 版本依赖）。"""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _cache_key(chunk: str) -> str:
    return hashlib.md5(chunk.encode("utf-8")).hexdigest()


async def contextualize_chunks(
    chunks: list[str],
    document_text: str,
    llm_func: Callable[[str], Awaitable[str]],
    *,
    batch_size: int = _DEFAULT_BATCH,
    cache: dict[str, str] | None = None,
    max_concurrency: int = _DEFAULT_CONCURRENCY,
) -> list[str]:
    """批量为 chunks 添加上下文前缀，返回与 chunks 等长、一一对应的 enriched 列表。

    Args:
        chunks: 已切好的原文片段列表。
        document_text: 这些 chunk 所属的整篇文档全文（取前 8000 字符作背景）。
        llm_func: ``async (prompt) -> 描述文本``，由调用方注入（用 fast_model）。
        batch_size: 每批并发请求数（同时控制一次 gather 的任务数）。
        cache: 可选的 ``{chunk_hash: enriched}`` 缓存；命中即跳过 LLM 调用。
        max_concurrency: 跨批次的全局最大并发 LLM 调用数。

    返回 enriched chunks；LLM 失败的 chunk 降级为原文（不丢内容）。长度恒等于入参。
    """
    if not chunks:
        return []

    doc_bg = (document_text or "")[:_DOC_TRUNCATE]
    enriched: list[str | None] = [None] * len(chunks)

    # 第一遍：缓存命中的直接填，未命中的收集待调
    pending_idx: list[int] = []
    for i, chunk in enumerate(chunks):
        if cache is not None:
            hit = cache.get(_cache_key(chunk))
            if hit is not None:
                enriched[i] = hit
                continue
        pending_idx.append(i)

    if not pending_idx:
        return [c for c in enriched if c is not None]  # 全命中

    # 全局并发上限（所有 batch 共享同一个 sem）
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(i: int) -> None:
        chunk = chunks[i]
        prompt = CONTEXT_PROMPT.format(whole_document=doc_bg, chunk_content=chunk)
        async with sem:
            try:
                desc = (await llm_func(prompt)).strip()
            except Exception as exc:  # 单 chunk 失败不拖垮整批，降级原文
                logger.warning("contextualize chunk %d 失败，降级为原文: %s", i, exc)
                desc = ""
        text = f"[背景] {desc}\n\n{chunk}" if desc else chunk
        enriched[i] = text
        if cache is not None:
            cache[_cache_key(chunk)] = text

    for batch in _batches(pending_idx, batch_size):
        await asyncio.gather(*[_one(i) for i in batch])

    # 兜底：任何 None（不该发生）回退原文，保证等长
    return [enriched[i] if enriched[i] is not None else chunks[i] for i in range(len(chunks))]


__all__ = ["contextualize_chunks", "CONTEXT_PROMPT"]
