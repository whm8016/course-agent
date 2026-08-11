"""Contextual Chunking（Anthropic Contextual Retrieval）。

给每个 chunk 注入一段「它在整篇文档里的位置/主题背景」前缀，让 embedding 和 BM25
都能把"脱离上下文就难匹配"的片段（代词指代、章节局部、公式序号）和文档主题关联起来。

Anthropic 实证（2024.09）：仅加 contextual prefix，检索失败率 5.7%→3.7%（-35%）；
配合 BM25 + rerank 可降到 1.9%（-67%）。

背景策略（v2，两层）：
1. 文档级摘要（每篇文档 1 次 LLM，见 ``summarize_document``）——所有 chunk 共享，回答
   "这是篇什么文档"。对标 dsRAG AutoContext，把 Anthropic 原方案 O(N×全文) 的 LLM 成本
   压到 O(1×全文 + N×短窗口)。
2. 位置感知局部窗口（``_WIN_BEFORE``/``_WIN_AFTER``）——按 chunk 在全文中的偏移取其邻近
   文本，回答"这个片段在文档哪个位置"。修掉 v1「所有 chunk 共用前 8000 字符」的错位缺陷：
   靠后的 chunk 不再拿到错误的开头背景，避免 LLM 被误导写出与实际内容无关的定位句。

注意：本项目用 DashScope qwen fast_model，没有 Anthropic 那种显式 cache_control 折扣，
所以"每块重传全文"成本不可控——这正是 v1 退化成截断 8000 字符的根因（截断解决了成本、
代价是正确性）。v2 用「1 次摘要 + N 次短窗口」绕开这个成本结构，且位置正确。

缓存：chunk 结果按 chunk 内容 hash 缓存（key 带 v2 版本前缀，改策略即失效，避免 A/B
评测命中旧结果测出假"无差异"）；文档摘要按 ``md5(document_text)`` 缓存（``docsum:``
命名空间隔离）。重复索引不重复调用 LLM。用便宜快速的 fast_model，不动主力模型。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Awaitable, Callable

from core.rag.source_utils import strip_source_prefix

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document_summary>
{doc_summary}
</document_summary>
<document_excerpt>
{local_context}
</document_excerpt>
这是从上述文档中提取的一个片段：
<chunk>
{chunk_content}
</chunk>
请用中文写 1-2 句简短描述，说明这个片段在文档中的位置和主题背景。
只输出描述本身，不要重复片段内容，不要加“这个片段”之类的前缀。"""

_DOC_SUMMARY_PROMPT = """<document>
{document_text}
</document>
请用中文写一段 200-300 字的概要，说明这篇文档是什么、涵盖哪些主要主题与章节结构。
只输出概要本身，不要加“本文档”之类的前缀。"""

# 局部窗口：chunk 前后各取一段文本作背景（字符数）。窗口随 chunk 位置滑动，替代 v1
# 全局共用前 8000 字符——兼顾"给足定位上下文"与"控制 fast_model 输入成本"。
_WIN_BEFORE = 1500
_WIN_AFTER = 500
# _locate 取 chunk 首多少字符作 needle：长到大概率唯一、短到避开 chunk 间的空白/格式差异。
_NEEDLE_LEN = 60
# 文档摘要输入超过此字符数时取首尾拼接（对齐 LlamaIndex DocumentContextExtractor 的
# warn 策略：宁可记 warning 也不静默丢尾）。
_DOC_SUMMARY_MAX_CHARS = 6000
# 背景策略版本：改策略即 bump，使旧缓存自然失效。
_STRATEGY_VERSION = "v2"
_DEFAULT_BATCH = 10
_DEFAULT_CONCURRENCY = 8

# chunk 文本开头由 ingestion._build_source_prefix 注入的【章节/来源/页码】结构前缀，
# 定位前必须先剥掉（strip_source_prefix），否则 document_text.find(chunk) 必然失配。


def _batches(seq: list[int], n: int):
    """手动分批（避免 itertools.batched 的 py3.12 版本依赖）。"""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _cache_key(chunk: str) -> str:
    return f"{_STRATEGY_VERSION}:{hashlib.md5(chunk.encode('utf-8')).hexdigest()}"


def _doc_summary_cache_key(document_text: str) -> str:
    return f"docsum:{_STRATEGY_VERSION}:{hashlib.md5(document_text.encode('utf-8')).hexdigest()}"


def _locate(document_text: str, chunk: str, cursor: int, ratio_hint: int) -> int:
    """返回 chunk 在全文中的起始偏移；匹配失败退化为按序号比例估算。

    chunk 开头带 ``【章节: xxx】\\n`` 结构前缀（切块阶段注入），直接 ``find`` 会失败，
    先剥前缀（``strip_source_prefix``）再取首 ``_NEEDLE_LEN`` 字符作 needle。同组 chunk
    按文档顺序排列，从 ``cursor`` 起单调前移查找，避免后一个 chunk 误命中前一个更早出现的位置。
    """
    needle = strip_source_prefix(chunk)[:_NEEDLE_LEN].strip()
    if needle:
        pos = document_text.find(needle, cursor)
        if pos >= 0:
            return pos
    return ratio_hint


async def summarize_document(
    document_text: str,
    llm_func: Callable[[str], Awaitable[str]],
    *,
    cache: dict[str, str] | None = None,
) -> str:
    """每篇文档调用一次 LLM 产出 200-300 字概要（"本文档是什么/涵盖哪些主题"）。

    所有 chunk 共享同一段摘要（对标 dsRAG AutoContext 的 document summary header），
    把背景的 O(N) LLM 成本降到 O(1)。输入超长时取首尾拼接 + warning，绝不静默丢尾。
    结果按 ``md5(document_text)`` 缓存，重复索引不重复调用。失败降级为空摘要，不阻断。
    """
    if not document_text or not document_text.strip():
        return ""

    if cache is not None:
        key = _doc_summary_cache_key(document_text)
        if cache.get(key) is not None:
            return cache[key]

    body = document_text
    if len(body) > _DOC_SUMMARY_MAX_CHARS:
        keep_head = _DOC_SUMMARY_MAX_CHARS // 2
        keep_tail = _DOC_SUMMARY_MAX_CHARS - keep_head
        body = body[:keep_head] + "\n…（中段省略）…\n" + body[-keep_tail:]
        logger.warning(
            "summarize_document 输入 %d 字符超过阈值 %d，取首尾拼接（不静默丢尾）",
            len(document_text), _DOC_SUMMARY_MAX_CHARS,
        )

    try:
        desc = (await llm_func(_DOC_SUMMARY_PROMPT.format(document_text=body))).strip()
    except Exception as exc:  # 摘要失败不阻断 enrichment，降级为空摘要
        logger.warning("summarize_document 失败，降级为空摘要: %s", exc)
        desc = ""

    if cache is not None and desc:
        cache[key] = desc
    return desc


async def contextualize_chunks(
    chunks: list[str],
    document_text: str,
    llm_func: Callable[[str], Awaitable[str]],
    *,
    doc_summary: str = "",
    batch_size: int = _DEFAULT_BATCH,
    cache: dict[str, str] | None = None,
    max_concurrency: int = _DEFAULT_CONCURRENCY,
) -> list[str]:
    """批量为 chunks 添加上下文前缀，返回与 chunks 等长、一一对应的 enriched 列表。

    Args:
        chunks: 已切好的原文片段列表（开头可能带【章节】结构前缀）。
        document_text: 这些 chunk 所属的整篇文档全文；用于定位每个 chunk 的偏移并取局部窗口。
        llm_func: ``async (prompt) -> 描述文本``，由调用方注入（用 fast_model）。
        doc_summary: 可选的文档级概要（``summarize_document`` 产出），所有 chunk 共享。
        batch_size: 每批并发请求数（同时控制一次 gather 的任务数）。
        cache: 可选的 ``{cache_key: enriched}`` 缓存；命中即跳过 LLM 调用。
        max_concurrency: 跨批次的全局最大并发 LLM 调用数。

    返回 enriched chunks；LLM 失败的 chunk 降级为原文（不丢内容）。长度恒等于入参。
    """
    if not chunks:
        return []

    doc_full = document_text or ""

    # 先按文档顺序一次性算出每个 chunk 在全文中的偏移，供后续取局部窗口。同组 chunk 本就
    # 按文档顺序排列，游标单调前进；匹配不到按序号比例兜底（定位失败也不抛异常）。
    positions: list[int] = []
    cursor = 0
    n = len(chunks)
    for i, chunk in enumerate(chunks):
        ratio_hint = int(len(doc_full) * i / n)
        pos = _locate(doc_full, chunk, cursor, ratio_hint)
        positions.append(pos)
        cursor = max(cursor, pos + 1)

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
        pos = positions[i]
        window = doc_full[max(0, pos - _WIN_BEFORE) : pos + len(chunk) + _WIN_AFTER]
        prompt = CONTEXT_PROMPT.format(
            doc_summary=doc_summary, local_context=window, chunk_content=chunk,
        )
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


__all__ = ["contextualize_chunks", "summarize_document", "CONTEXT_PROMPT"]
