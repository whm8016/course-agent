"""core/rag/retrieval_config.py 单测：RRF / linear 融合的 fused_score 写回 + 不污染入参。

回归点：融合分数历史上写不进返回值（返回的是原对象，score 仍是 dense/sparse 原始分），
让任何「看融合分数」的调试/可观测看到假分数。改后须写回 fused_score 到副本且不动入参。
"""
from __future__ import annotations

import pytest

from core.rag.retrieval_config import linear_fusion, reciprocal_rank_fusion


def _list(prefix: str, n: int) -> list[dict]:
    """构造一路召回：chunk_id=prefix+i，score 按序递减（模拟排名序）。"""
    return [
        {"chunk_id": f"{prefix}{i}", "content": f"{prefix}{i}", "score": 1.0 - i * 0.1}
        for i in range(n)
    ]


class TestRRF:
    def test_writes_fused_score_and_orders_by_it(self):
        out = reciprocal_rank_fusion(_list("d", 3), _list("s", 3), k=60)
        assert all("fused_score" in d for d in out)
        scores = [d["fused_score"] for d in out]
        assert scores == sorted(scores, reverse=True)

    def test_overlapping_id_accumulates_score(self):
        # 两路同 chunk_id "x" 且都排第一 → RRF 分数累加 2 × 1/(k+0+1)
        dense = [{"chunk_id": "x", "content": "x", "score": 0.9}]
        sparse = [{"chunk_id": "x", "content": "x", "score": 0.8}]
        out = reciprocal_rank_fusion(dense, sparse, k=60)
        assert len(out) == 1
        assert out[0]["fused_score"] == pytest.approx(2 * (1 / 61))

    def test_does_not_mutate_inputs(self):
        dense = [{"chunk_id": "d0", "content": "d0", "score": 0.5}]
        sparse = [{"chunk_id": "s0", "content": "s0", "score": 0.4}]
        reciprocal_rank_fusion(dense, sparse, k=60)
        assert "fused_score" not in dense[0]
        assert "fused_score" not in sparse[0]


class TestLinearFusion:
    def test_writes_fused_score(self):
        out = linear_fusion(_list("d", 2), _list("s", 2), alpha=0.5)
        assert all("fused_score" in d for d in out)
        scores = [d["fused_score"] for d in out]
        assert scores == sorted(scores, reverse=True)

    def test_does_not_mutate_inputs(self):
        dense = [{"chunk_id": "d0", "content": "d0", "score": 0.5}]
        sparse = [{"chunk_id": "s0", "content": "s0", "score": 0.4}]
        linear_fusion(dense, sparse, alpha=0.5)
        assert "fused_score" not in dense[0]
        assert "fused_score" not in sparse[0]
