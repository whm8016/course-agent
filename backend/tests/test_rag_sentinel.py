"""RAG 第三道防线回归测试：无命中英文哨兵不再被当成课程证据。

覆盖 core/rag/retriever/lightrag.py 的 _is_no_context / _extract_contexts：
- LightRAG naive_query 无 chunk 时 aquery_llm 用 fail_response 兜底，产出形如
  "Sorry, I'm not able to provide an answer to that question.[no-context]" 的非空字符串。
- 修复前它被 _extract_contexts 当成 [证据1] 喂给 Agent + 带 sources 弹来源卡片；
- 修复后 _extract_contexts 在单一 chokepoint 拦截（覆盖 retrieve/retrieve_context/
  _retrieve_hybrid/dense_search/graph_augmented_retrieve/query 全部 6 条路径）。
"""
from __future__ import annotations

from core.rag.retriever.lightrag import _extract_contexts, _is_no_context

# LightRAG fail_response 的真实兜底字符串（site-packages lightrag/prompt.py:331）。
_SENTINEL = "Sorry, I'm not able to provide an answer to that question.[no-context]"


def test_is_no_context_detects_marker():
    assert _is_no_context(_SENTINEL) is True
    # 措辞变化但标记在——仍应命中（按标记匹配，比整句稳健）
    assert _is_no_context("一些其它措辞。[no-context]") is True


def test_is_no_context_real_content_not_flagged():
    assert _is_no_context("基尔霍夫电压定律（KVL）：沿任意回路电压代数和为零。") is False
    assert _is_no_context("") is False


def test_extract_contexts_filters_sentinel_string():
    """非空但无命中的兜底字符串 → 必须返回 []，不能当成证据。"""
    assert _extract_contexts(_SENTINEL) == []
    assert _extract_contexts("  Sorry, whatever.[no-context]  ") == []


def test_extract_contexts_passes_real_string():
    real = "KVL：回路电压代数和为零。"
    assert _extract_contexts(real) == [real]


def test_extract_contexts_passes_list_unchanged():
    chunks = [{"content": "a"}, {"content": "b"}]
    # list 直接透传（已是真实 contexts，无哨兵）
    assert _extract_contexts(chunks) == chunks


def test_extract_contexts_empty_string_returns_empty():
    assert _extract_contexts("") == []
    assert _extract_contexts("   ") == []


def test_extract_contexts_filters_sentinel_in_dict_value():
    """dict 的 context/chunks 等键里若是哨兵字符串，同样拦截。"""
    assert _extract_contexts({"context": _SENTINEL}) == []
    # 真实 dict 值照常提取
    assert _extract_contexts({"context": "真实证据文本"}) == ["真实证据文本"]


def test_extract_contexts_none_returns_empty():
    assert _extract_contexts(None) == []


def test_min_rerank_score_default_is_zero():
    """阈值默认 0.0=不过滤=行为不变（方向安全）。"""
    from settings import get_settings

    assert get_settings().lightrag.min_rerank_score == 0.0
