"""统计分析引擎 —— 分布式指标 + 回归检测 + 历史趋势对比。

单一均值掩盖长尾风险：faithfulness 均值 0.85 但 P10=0.20 意味着 10% 的回答严重幻觉，
均值把这个长尾风险藏起来了。业界标准看分布（P50/P90/P95），并用 Welch t-test 检验
两次评测是否有统计显著性下降（回归）。

numpy/scipy 在函数内 lazy import（评测可选依赖，venv 未装时调用报清晰错误）。
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any

from . import config

logger = logging.getLogger(__name__)


_DIST_KEYS = ("mean", "std", "p50", "p90", "p95", "min", "max")


# ---------------------------------------------------------------------------
# 分布式指标
# ---------------------------------------------------------------------------
def compute_distribution(scores: list[float]) -> dict[str, float]:
    """计算分数分布：mean/std/p50/p90/p95/min/max。空列表返回全 0。

    scores 中的 NaN/None 应在调用前清零（ragas_evaluator._safe_float 已处理）。
    """
    import numpy as np

    if not scores:
        return {k: 0.0 for k in _DIST_KEYS}
    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_distribution_by_metric(
    per_question_scores: list[dict[str, float]], metric_names: list[str]
) -> dict[str, dict[str, float]]:
    """对每个 metric，从逐条分数算分布。返回 {metric: {mean/std/p50/...}}。"""
    out: dict[str, dict[str, float]] = {}
    for m in metric_names:
        vals = [row.get(m, 0.0) for row in per_question_scores]
        out[m] = compute_distribution(vals)
    return out


# ---------------------------------------------------------------------------
# 回归检测：Welch t-test
# ---------------------------------------------------------------------------
def regression_test(
    baseline: list[float], candidate: list[float], alpha: float = 0.05
) -> dict[str, Any]:
    """Welch t-test 检验 candidate 相对 baseline 是否有统计显著性下降（回归）。

    判定回归：p_value < alpha 且 candidate 均值 < baseline 均值。
    样本不足（<2）或方差为 0 时 t-test 退化，返回 not_enough_data，不误报回归。
    """
    from scipy.stats import ttest_ind

    if len(baseline) < 2 or len(candidate) < 2:
        return {"p_value": None, "regression": False, "reason": "样本不足（<2），无法检验"}
    try:
        _stat, p_value = ttest_ind(baseline, candidate, equal_var=False, nan_policy="omit")
    except Exception as e:
        return {"p_value": None, "regression": False, "reason": f"t-test 失败: {e}"}
    if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
        return {"p_value": None, "regression": False, "reason": "方差为 0，t-test 退化"}

    b_mean = sum(baseline) / len(baseline)
    c_mean = sum(candidate) / len(candidate)
    regression = bool(p_value < alpha and c_mean < b_mean)
    if regression:
        reason = "显著性下降（回归）"
    elif c_mean < b_mean:
        reason = "下降但未达显著"
    else:
        reason = "无回归"
    return {
        "p_value": float(p_value),
        "baseline_mean": b_mean,
        "candidate_mean": c_mean,
        "delta": c_mean - b_mean,
        "regression": regression,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# 历史趋势对比
# ---------------------------------------------------------------------------
def load_last_summary() -> dict[str, Any] | None:
    """读 results/ 下最近一次 eval_summary_*.json，无则 None（首次运行优雅降级）。"""
    files = sorted(config.RESULTS_DIR.glob("eval_summary_*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text("utf-8"))
    except Exception as e:
        logger.warning("读取历史结果失败: %s", e)
        return None


def diff_avg_against(
    current: dict[str, dict[str, float]],
    baseline_avg: dict[str, dict[str, float]] | None,
    metric_names: list[str],
) -> dict[str, dict[str, float]]:
    """当前 avg vs 历史 baseline avg 的逐指标 delta（current - baseline）。

    baseline_avg 为 None（首次运行）时 delta 全 0，并在结果里标记无历史。
    """
    diff: dict[str, dict[str, float]] = {}
    for mode, scores in current.items():
        base = (baseline_avg or {}).get(mode, {})
        diff[mode] = {m: round(scores.get(m, 0.0) - base.get(m, 0.0), 4) for m in metric_names}
    return diff


# ---------------------------------------------------------------------------
# 延迟分布（Phase 3.5：runner 层已采集 retrieve_ms/query_ms）
# ---------------------------------------------------------------------------
def compute_latency_distribution(mode_results: list[dict]) -> dict[str, dict[str, float]]:
    """从 mode_results 的 retrieve_ms/query_ms 算延迟分布。"""
    retrieve_ms = [r.get("retrieve_ms", 0) for r in mode_results]
    query_ms = [r.get("query_ms", 0) for r in mode_results]
    total_ms = [r.get("retrieve_ms", 0) + r.get("query_ms", 0) for r in mode_results]
    return {
        "retrieve_ms": compute_distribution([float(x) for x in retrieve_ms]),
        "query_ms": compute_distribution([float(x) for x in query_ms]),
        "total_ms": compute_distribution([float(x) for x in total_ms]),
    }
