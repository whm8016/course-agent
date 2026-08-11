"""检索流水线配置 + 消融实验预设 + 融合算法。

RetrievalConfig 是查询时可实时切换的旋钮集合（召回路径 / 融合 / 精排），每个实例
代表一组消融实验配置。索引时开关（contextual chunking）不放这里——它改了要重新
索引，由 settings.chunking 控制。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    """每个实例代表一组消融实验配置。查询时可实时切换。"""

    # --- 召回路径 ---
    # 20/20：接上精排后召回深度按精排 token 成本（按 doc 计费）重新定档。20+20 融合后最多
    # 40 条，RRF 做 40→精排候选池的粗筛，精排再做 →top_k 的精筛。旧 50/50 是为「无精排、
    # RRF 直接出结果」设计的，精排在场时深召回的边际收益被 token 成本吃掉（见 rerank.py）。
    # 旋钮可调：消融掉点时先扫回 30/30 或 50/50 再评估，而非直接付 60k token 成本。
    bm25_enabled: bool = True
    bm25_top_k: int = 20
    dense_enabled: bool = True
    dense_top_k: int = 20

    # --- 融合 ---
    fusion_method: str = "rrf"  # "rrf" | "linear"
    rrf_k: int = 60  # RRF 公式中的 k，ES 默认 60
    linear_alpha: float = 0.5  # linear 融合时第一路（bm25）权重

    # --- 精排 ---
    rerank_enabled: bool = True
    rerank_top_n: int = 5
    # 精排后相关性阈值（仅在 rerank_enabled 且拿到 rerank_score 时生效）。0.0=不过滤。
    # 与 settings.rerank.min_score 同口径（qwen3-rerank relevance_score），两后端共用。
    min_rerank_score: float = 0.0

    # --- 标识 ---
    name: str = ""

    def label(self) -> str:
        parts: list[str] = []
        if self.bm25_enabled:
            parts.append("bm25")
        if self.dense_enabled:
            parts.append("dense")
        if len(parts) > 1:
            parts.append(self.fusion_method)
        if self.rerank_enabled:
            parts.append("rerank")
            if self.min_rerank_score > 0:
                parts.append(f"gate{self.min_rerank_score:g}")
        return self.name or "+".join(parts) or "empty"


# ---- 预定义消融组合 ----
ABLATION_CONFIGS: dict[str, "RetrievalConfig"] = {
    "dense_only": RetrievalConfig(bm25_enabled=False, rerank_enabled=False, name="dense_only"),
    "bm25_only": RetrievalConfig(dense_enabled=False, rerank_enabled=False, name="bm25_only"),
    "dense+rerank": RetrievalConfig(bm25_enabled=False, name="dense+rerank"),
    "bm25+rerank": RetrievalConfig(dense_enabled=False, name="bm25+rerank"),
    "hybrid_no_rerank": RetrievalConfig(rerank_enabled=False, name="hybrid_no_rerank"),
    "hybrid_rrf+rerank": RetrievalConfig(name="hybrid_rrf+rerank"),  # 完整流水线
    "hybrid_linear+rerank": RetrievalConfig(fusion_method="linear", name="hybrid_linear+rerank"),
}

DEFAULT_CONFIG = ABLATION_CONFIGS["hybrid_rrf+rerank"]


# ---- 融合算法 ----

def reciprocal_rank_fusion(*ranked_lists, k: int = 60) -> list[dict]:
    """RRF: score(d) = sum(1/(k + rank))。Cormack et al. 2009。

    多路召回按各自「排名」融合，不依赖原始分数绝对值——BM25 与 dense 分数尺度不同，
    RRF 用排名规避了这个差异，是 hybrid search 的业界默认融合法。chunk_id 作 join key，
    同 id 取首次出现的 doc（保留其元信息）。
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in ranked_lists:
        for rank, item in enumerate(results):
            doc_id = item.get("chunk_id") or item.get("id") or ""
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = item
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    # 写回 fused_score 到 dict 副本——返回原对象的话其 score 仍是 dense/sparse 的原始分，
    # 融合分数在系统里任何地方都取不到（调试/可观测看到的会是假分数）。副本避免污染上游
    # ranked_lists 里的 dict（它们就是 dense/sparse 各自列表里的对象，原地写会糊掉阶段边界）。
    return [{**docs[did], "fused_score": scores[did]} for did in sorted_ids if did in docs]


def linear_fusion(*ranked_lists, alpha: float = 0.5) -> list[dict]:
    """线性加权融合：每路分数 min-max 归一化后按权重求和。

    alpha = 第一路权重，其余权重平均分配。比 RRF 更依赖分数质量（归一化对齐尺度）。
    不修改入参 dict（归一化值算到局部 agg，避免污染调用方的 score）。
    """
    if not ranked_lists:
        return []
    n = len(ranked_lists)
    if n == 1:
        weights = [1.0]
    else:
        weights = [alpha] + [(1.0 - alpha) / (n - 1)] * (n - 1)

    agg: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for w, results in zip(weights, ranked_lists):
        if not results:
            continue
        raw = [float(it.get("score", 0.0)) for it in results]
        lo, hi = min(raw), max(raw)
        span = hi - lo if hi > lo else 1.0
        for it, s in zip(results, raw):
            doc_id = it.get("chunk_id") or it.get("id") or ""
            if not doc_id:
                continue
            agg[doc_id] = agg.get(doc_id, 0.0) + w * ((s - lo) / span)
            if doc_id not in docs:
                docs[doc_id] = it
    sorted_ids = sorted(agg, key=agg.get, reverse=True)
    return [{**docs[did], "fused_score": agg[did]} for did in sorted_ids if did in docs]


__all__ = [
    "RetrievalConfig",
    "ABLATION_CONFIGS",
    "DEFAULT_CONFIG",
    "reciprocal_rank_fusion",
    "linear_fusion",
]
