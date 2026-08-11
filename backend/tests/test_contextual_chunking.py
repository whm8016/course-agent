"""contextual_chunking 单测：覆盖 v2「位置感知局部窗口 + 文档摘要」修复的核心回归点。

回归点（对齐 plan）：
1. 靠后 chunk 的背景含其邻近文本、不含文档开头（v1 的 bug：所有 chunk 共用前 8000 字符）。
2. 带【章节】结构前缀的 chunk 仍能正确定位（直接 find 会失败，必须先剥前缀）。
3. 定位失败走比例兜底且不抛异常。
4. 缓存 key 带 v2 前缀，v1 裸 md5 旧缓存不被命中。
"""
from __future__ import annotations

import hashlib
import re

from core.rag.contextual_chunking import (
    _cache_key,
    _DOC_SUMMARY_MAX_CHARS,
    contextualize_chunks,
    summarize_document,
)

# 测试语料：OPENING 在最前、NEEDLE 在 5000 填充字符之后（足够远，超出 1500 的前窗）。
OPENING = "这是文档的开头总览部分。"
NEEDLE = "这是位于文档很靠后位置的唯一片段标记。"
DOC = OPENING + "甲" * 5000 + NEEDLE + "乙" * 300


class _CapturingLLM:
    """记录每次收到的 prompt，返回固定回复。"""

    def __init__(self, reply: str = "一段背景描述。"):
        self.prompts: list[str] = []
        self.reply = reply

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def _excerpt(prompt: str) -> str:
    """从 prompt 里抽出 <document_excerpt>…</document_excerpt> 的窗口内容。"""
    m = re.search(r"<document_excerpt>\n(.*?)\n</document_excerpt>", prompt, re.S)
    return m.group(1) if m else ""


# ── 回归点 1：靠后 chunk 拿到邻近窗口，不拿文档开头 ─────────────────────────────
async def test_late_chunk_window_excludes_opening():
    llm = _CapturingLLM()
    await contextualize_chunks([NEEDLE], DOC, llm)

    assert len(llm.prompts) == 1
    excerpt = _excerpt(llm.prompts[0])
    # 定位成功：窗口在 NEEDLE 处，包含 chunk 邻近
    assert NEEDLE in excerpt
    # 不含文档开头——这正是 v1「所有 chunk 共用前 8000 字符」错位缺陷的回归断言
    assert OPENING not in excerpt
    assert OPENING not in llm.prompts[0]


# ── 回归点 2：带【章节】前缀的 chunk 仍能定位到靠后位置 ─────────────────────────
async def test_prefixed_chunk_locates_late():
    llm = _CapturingLLM()
    chunk = f"【章节: 实验三 交流电路】\n{NEEDLE}"
    await contextualize_chunks([chunk], DOC, llm)

    excerpt = _excerpt(llm.prompts[0])
    # 剥前缀后命中 NEEDLE：窗口落在 NEEDLE 处，而非比例兜底的文档开头。
    # 若没有剥前缀，find 必然失败 → 兜底 pos=0 → excerpt 会含 OPENING 不含 NEEDLE。
    assert NEEDLE in excerpt
    assert OPENING not in excerpt


async def test_prefix_with_page_marker_still_stripped():
    """前缀里带页码（【章节: x | 第N页】）也要能剥，证明正则不被 | 卡住。"""
    llm = _CapturingLLM()
    chunk = f"【章节: 实验 | 第3页】\n{NEEDLE}"
    await contextualize_chunks([chunk], DOC, llm)

    excerpt = _excerpt(llm.prompts[0])
    assert NEEDLE in excerpt


# ── 回归点 3：定位失败走比例兜底，不抛异常 ─────────────────────────────────────
async def test_locate_miss_falls_back_no_raise():
    llm = _CapturingLLM()
    ghost = "【来源: 不存在】\n这段文字在文档里根本找不到XYZ123"
    out = await contextualize_chunks([ghost], DOC, llm)

    assert len(out) == 1          # 不抛异常，等长返回
    assert ghost in out[0]        # 原文保留
    # 单 chunk i=0,n=1 → ratio_hint=0 → 兜底窗口取自文档开头
    excerpt = _excerpt(llm.prompts[0])
    assert OPENING in excerpt


async def test_cursor_advances_past_earlier_match():
    """同一文本在全文出现两次：第二个 chunk 的游标必须跳过早期出现、命中晚期出现。

    cache=None：相同 chunk 文本不会互相缓存命中，保证两个 chunk 都进 LLM、都能捕获 prompt。
    """
    dup = "重复出现的片段标记XYZ。"
    early_flag = "早期独有EE。"
    late_flag = "晚期独有LL。"
    doc = OPENING + "甲" * 2000 + dup + early_flag + "乙" * 2000 + dup + late_flag + "丁" * 100
    llm = _CapturingLLM()
    await contextualize_chunks([dup, dup], doc, llm, cache=None)

    assert len(llm.prompts) == 2
    # 第一个 dup 命中早期出现：紧随其后的 early_flag 落在 500 后窗内
    assert early_flag in _excerpt(llm.prompts[0])
    # 第二个 dup 游标已越过早期出现、命中晚期：窗口含 late_flag，不含 early_flag
    assert late_flag in _excerpt(llm.prompts[1])
    assert early_flag not in _excerpt(llm.prompts[1])


# ── 回归点 4：缓存 key v2 隔离 ───────────────────────────────────────────────
def test_cache_key_has_v2_prefix():
    assert _cache_key("anything").startswith("v2:")


async def test_v1_raw_md5_cache_not_hit():
    """v1 风格的裸 md5 key（无 v2: 前缀）不被识别 → 仍调 LLM。"""
    chunk = "阿尔法片段"
    llm = _CapturingLLM()
    v1_cache = {hashlib.md5(chunk.encode("utf-8")).hexdigest(): "[背景] 旧结果\n\n" + chunk}
    await contextualize_chunks([chunk], "文档正文", llm, cache=v1_cache)

    assert len(llm.prompts) == 1   # v1 key 未命中，仍调了 LLM


async def test_v2_cache_hit_skips_llm():
    chunk = "贝塔片段"
    llm = _CapturingLLM()
    cached = "[背景] 已缓存\n\n" + chunk
    v2_cache = {_cache_key(chunk): cached}
    out = await contextualize_chunks([chunk], "文档正文", llm, cache=v2_cache)

    assert llm.prompts == []        # 命中，不调 LLM
    assert out[0] == cached


# ── 文档摘要：缓存、超长首尾拼接、注入 prompt ─────────────────────────────────
async def test_summarize_document_caches_by_content():
    llm = _CapturingLLM(reply="本文档讲解电路实验。")
    cache: dict[str, str] = {}
    text = "一段足够长的文档内容。" * 10
    s1 = await summarize_document(text, llm, cache=cache)
    s2 = await summarize_document(text, llm, cache=cache)

    assert s1 == s2 == "本文档讲解电路实验。"
    assert len(llm.prompts) == 1    # 第二次命中缓存


async def test_summarize_document_empty_returns_empty():
    llm = _CapturingLLM()
    assert await summarize_document("", llm) == ""
    assert await summarize_document("   \n  ", llm) == ""
    assert llm.prompts == []


async def test_summarize_document_overlong_uses_head_tail():
    """超长输入取首尾拼接（含中段省略标记），不静默丢尾。"""
    llm = _CapturingLLM(reply="概要")
    long_doc = "头标记内容。" + "内" * (_DOC_SUMMARY_MAX_CHARS + 500) + "尾标记内容。"
    await summarize_document(long_doc, llm)

    assert len(llm.prompts) == 1
    body = llm.prompts[0]
    assert "（中段省略）" in body
    assert "头标记内容" in body
    assert "尾标记内容" in body       # 静默截断会丢尾，这里必须还在


async def test_doc_summary_injected_into_prompt():
    llm = _CapturingLLM()
    await contextualize_chunks(["某个片段"], "文档正文", llm, doc_summary="这是自定义文档摘要。")

    assert "这是自定义文档摘要。" in llm.prompts[0]


# ── 兜底行为 ─────────────────────────────────────────────────────────────────
async def test_empty_chunks_returns_empty():
    llm = _CapturingLLM()
    assert await contextualize_chunks([], "文档", llm) == []
    assert llm.prompts == []


async def test_llm_failure_degrades_to_original():
    async def raising(prompt: str) -> str:
        raise RuntimeError("boom")

    out = await contextualize_chunks(["片段内容"], "文档正文", raising)
    assert out == ["片段内容"]       # LLM 失败降级原文，不加 [背景] 前缀
