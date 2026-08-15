"""course_topic_builder 抽取链单测（纯函数 + mock LLM，零真实调用）。

覆盖跨章先修边修复的四件套：
- build_edge_candidates：embedding top-k 语义候选（取代 order 差 ≤3 硬窗口）
- judge_pairs_batch：批量四分类判定（forward/reverse/none/uncertain + score）
- break_cycles / close_transitive：DAG 保证与传递闭包
- 次级修复：merge 保留非 NULL embedding / _llm_json 重试 / failed_sections 记录
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.memory.course_topic_builder import (
    _llm_json,
    break_cycles,
    build_edge_candidates,
    close_transitive,
    extract_topics_from_chunks,
    judge_pairs_batch,
    merge_synonymous,
)


def _resp(content: str):
    """构造 _llm_json 能吃的假 LLM 响应（无 usage 属性 → track 静默跳过）。"""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _fake_llm(contents: list[str]):
    llm = SimpleNamespace()
    llm.chat = SimpleNamespace()
    llm.chat.completions = SimpleNamespace()
    llm.chat.completions.create = AsyncMock(side_effect=[_resp(c) for c in contents])
    return llm


def _topic(label: str, emb=None, sec="第1章", order=0):
    return {
        "topic_id": label, "label": label, "definition": f"{label}的定义",
        "source_section": sec, "order_idx": order, "embedding": emb,
    }


# ---------------------------------------------------------------------------
# build_edge_candidates：语义 top-k 预筛
# ---------------------------------------------------------------------------

def test_build_edge_candidates_topk_prefers_similar():
    # 两簇向量（簇内余弦 ~1.0，簇间 ~0）：高相似对必入选、零相似对不入选。
    # 注：k 是数量不是阈值——b 簇只有 2 个主题时 b1 的 top-2 会溢出到 a 簇末位
    # （小课程自然退化为近全配对，成本可接受，设计内行为）。
    a1, a2, a3 = [1.0, 0.0], [1.0, 0.0], [0.9, 0.1]
    b1, b2 = [0.0, 1.0], [0.0, 1.0]
    topics = [
        _topic("a1", a1), _topic("a2", a2), _topic("a3", a3, order=1),
        _topic("b1", b1), _topic("b2", b2, sec="第9章", order=2),
    ]
    pairs = build_edge_candidates(topics, k=2)
    assert (0, 1) in pairs and (3, 4) in pairs   # 余弦 1.0 的对必互为 top-1
    # a1/a2 与 b 簇余弦 0.0：不进彼此候选（各自 top-2 全在簇内）
    for banned in ((0, 3), (0, 4), (1, 3), (1, 4)):
        assert banned not in pairs
    # 无向去重：每对只出现一次且 i < j
    assert all(i < j for i, j in pairs)
    assert len(pairs) == len(set(pairs))


def test_build_edge_candidates_k_truncates():
    # 5 个全相似主题，k=2 → 每主题最多 2 候选，总对数受控
    topics = [_topic(f"t{i}", [1.0, float(i) / 100]) for i in range(5)]
    pairs = build_edge_candidates(topics, k=2)
    assert len(pairs) <= 5 * 2 // 2 + 2  # 上界粗检（去重后更少）


def test_build_edge_candidates_none_embedding_falls_back_to_window():
    # 无 embedding 的主题退回「同章全部 + order 差 ≤3」保底，不零候选
    topics = [
        _topic("x", None, sec="第1章", order=0),
        _topic("same_sec", [1.0, 0.0], sec="第1章", order=1),
        _topic("far_other", [0.0, 1.0], sec="第8章", order=10),
    ]
    pairs = build_edge_candidates(topics, k=1)
    assert (0, 1) in pairs          # 同章保底
    assert (0, 2) not in pairs      # 跨章且 order 差 >3：降级窗口不含


# ---------------------------------------------------------------------------
# judge_pairs_batch：批量四分类
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_pairs_batch_parse():
    topics = [
        _topic("A", sec="第1章"), _topic("B", sec="第5章"),
        _topic("C", sec="第2章"), _topic("D", sec="第6章"),
        _topic("E", sec="第3章"),
    ]
    # 5 对一批：forward / reverse / none / uncertain / score 不足
    reply = json.dumps({"pairs": [
        {"id": 0, "direction": "forward", "score": 0.9},   # A 前置 B
        {"id": 1, "direction": "reverse", "score": -0.8},  # D 前置 C（翻转）
        {"id": 2, "direction": "none", "score": 0.1},
        {"id": 3, "direction": "uncertain", "score": 0.2},
        {"id": 4, "direction": "forward", "score": 0.3},   # 分数不足不建边
    ]}, ensure_ascii=False)
    llm = _fake_llm([reply])
    edges, uncertain = await judge_pairs_batch("c1", [(0, 1), (2, 3), (1, 2), (3, 4), (0, 4)], topics, llm, "m")
    assert [(e["src"], e["dst"]) for e in edges] == [("A", "B"), ("D", "C")]
    assert all(e["relation"] == "prerequisite" for e in edges)
    assert edges[0]["confidence"] == pytest.approx(0.9)
    assert uncertain == [(3, 4)]


@pytest.mark.asyncio
async def test_judge_pairs_batch_missing_id_goes_uncertain():
    topics = [_topic("A"), _topic("B")]
    llm = _fake_llm([json.dumps({"pairs": [{"id": 9, "direction": "forward", "score": 0.9}]})])
    edges, uncertain = await judge_pairs_batch("c1", [(0, 1)], topics, llm, "m")
    assert edges == []
    assert uncertain == [(0, 1)]


# ---------------------------------------------------------------------------
# break_cycles / close_transitive
# ---------------------------------------------------------------------------

def _e(src: str, dst: str, conf: float):
    return {"src": src, "dst": dst, "relation": "prerequisite", "confidence": conf}


def test_break_cycles_removes_weakest_edge():
    # A→B→C→A 三元环：删 C→A（最低置信）后成 DAG
    edges = [_e("A", "B", 0.9), _e("B", "C", 0.8), _e("C", "A", 0.3)]
    out = break_cycles(edges)
    assert {(e["src"], e["dst"]) for e in out} == {("A", "B"), ("B", "C")}
    # 入参不被原地修改
    assert len(edges) == 3


def test_break_cycles_no_cycle_untouched():
    edges = [_e("A", "B", 0.9), _e("B", "C", 0.8), _e("A", "C", 0.7)]
    assert break_cycles(edges) == edges


def test_close_transitive_fills_and_keeps_direct():
    edges = [_e("A", "B", 0.8), _e("B", "C", 0.6), _e("A", "D", 0.9)]
    out = close_transitive(edges)
    by = {(e["src"], e["dst"]): e["confidence"] for e in out}
    assert by[("A", "C")] == pytest.approx(0.6)   # min(0.8, 0.6) 传递补边
    assert by[("A", "B")] == pytest.approx(0.8)   # 直接边不覆盖
    assert ("D", "B") not in by                    # D 无出边闭包不乱补


def test_close_transitive_direct_edge_not_upgraded():
    # A→C 已有直接边 0.5，经 B 的路径 min(0.9,0.8)=0.8 更高——直接边不被覆盖
    # （LLM 直接判定优先于传递推断）
    edges = [_e("A", "B", 0.9), _e("B", "C", 0.8), _e("A", "C", 0.5)]
    out = close_transitive(edges)
    by = {(e["src"], e["dst"]): e["confidence"] for e in out}
    assert by[("A", "C")] == pytest.approx(0.5)
    assert len(out) == 3   # 无新边补入


# ---------------------------------------------------------------------------
# 次级修复：merge / _llm_json 重试 / failed_sections
# ---------------------------------------------------------------------------

def test_merge_synonymous_keeps_nonnull_embedding():
    topics = [
        _topic("戴维南等效", None, order=0),
        _topic("戴维南等效", [0.1, 0.2], order=1),
    ]
    merged = merge_synonymous(topics)
    assert len(merged) == 1
    assert merged[0]["embedding"] == [0.1, 0.2]


def test_merge_synonymous_label_dedup_and_reindex():
    topics = [
        _topic("伏安特性", order=0), _topic("Other", order=1), _topic("伏安特性", order=2),
    ]
    merged = merge_synonymous(topics)
    assert [t["label"] for t in merged] == ["伏安特性", "Other"]
    assert [t["order_idx"] for t in merged] == [0, 1]


@pytest.mark.asyncio
async def test_llm_json_retries_once_then_succeeds():
    llm = _fake_llm(["不是json", '{"ok": 1}'])
    data = await _llm_json(llm, "m", "prompt")
    assert data == {"ok": 1}
    assert llm.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_llm_json_two_failures_return_none():
    llm = _fake_llm(["崩", "还崩"])
    assert await _llm_json(llm, "m", "prompt") is None
    assert llm.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_extract_topics_returns_failed_sections():
    chunks = [
        {"text": "戴维南等效的正文", "section": "第1章", "order_idx": 0},
        {"text": "另一章正文", "section": "第2章", "order_idx": 1},
    ]
    llm = _fake_llm([
        '{"topics": [{"label": "戴维南等效", "definition": "等效电源定理"}]}',
        "彻底崩", "还是崩",   # 第 2 章两试皆败
    ])
    topics, failed = await extract_topics_from_chunks("c1", chunks, llm, "m")
    assert [t["label"] for t in topics] == ["戴维南等效"]
    assert failed == ["第2章（第1块）"]
