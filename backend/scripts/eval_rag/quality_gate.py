"""CI 质量门禁 —— 阈值检查 + exit code。

run_eval 结束时调用 check_quality_gate(summary)：
  - 全部达标 → exit 0（CI 通过）
  - 任一不达标 → exit 1（阻断 CI，打印哪些指标不合格）

gate 命名约定（见 config.QUALITY_GATES）：
  - 分数类：`<metric>_min`（下限，取所有 mode 最差值 ≥ 阈值）
            `<metric>_max`（上限，如 noise_sensitivity，取所有 mode 最差值 ≤ 阈值）
  - 延迟类：`latency__<field>__<stat>`（取所有 mode 该 field 的 stat 最大值 ≤ 阈值）

"最差值"取所有 mode 的 min（_min 类）/ max（_max 类），最保守——任一 mode 不达标即判失败。
"""
from __future__ import annotations

import logging
from typing import Any

from . import config

logger = logging.getLogger(__name__)


def check_quality_gate(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    """检查 summary 是否满足 config.QUALITY_GATES。

    返回 (全部通过, 失败原因列表)。本轮未评测的指标（avg_scores 中无该 metric）跳过，
    不判失败——避免"没跑的指标"误判为不达标。
    """
    failures: list[str] = []
    avg_scores = summary.get("avg_scores", {})
    latency = summary.get("latency", {})
    modes = summary.get("modes") or list(avg_scores.keys())

    for gate, threshold in config.QUALITY_GATES.items():
        if gate.startswith("latency__"):
            failure = _check_latency_gate(gate, threshold, latency, modes)
            if failure:
                failures.append(failure)
        elif gate.endswith("_min"):
            metric = gate[: -len("_min")]
            vals = _collect_scores(avg_scores, modes, metric)
            if vals and min(vals) < threshold:
                failures.append(
                    f"{metric} 最低 {min(vals):.4f} < 阈值 {threshold}（要求 ≥）"
                )
        elif gate.endswith("_max"):
            metric = gate[: -len("_max")]
            vals = _collect_scores(avg_scores, modes, metric)
            if vals and max(vals) > threshold:
                failures.append(
                    f"{metric} 最高 {max(vals):.4f} > 阈值 {threshold}（要求 ≤）"
                )
        else:
            logger.warning("未知门禁规则（无法解析）: %s，跳过", gate)

    return (not failures), failures


def _collect_scores(
    avg_scores: dict[str, dict[str, float]], modes: list[str], metric: str
) -> list[float]:
    """收集所有 mode 中该 metric 的 avg 分数。"""
    vals: list[float] = []
    for mode in modes:
        v = avg_scores.get(mode, {}).get(metric)
        if v is not None:
            vals.append(float(v))
    return vals


def _check_latency_gate(
    gate: str, threshold: float, latency: dict, modes: list[str]
) -> str | None:
    """解析 latency__<field>__<stat>，取所有 mode 该 field 的 stat 最大值，超阈值返回描述。"""
    parts = gate.split("__")  # ["latency", field, stat]
    if len(parts) != 3:
        return None
    _, field, stat = parts
    vals: list[float] = []
    for mode in modes:
        v = latency.get(mode, {}).get(field, {}).get(stat)
        if v is not None:
            vals.append(float(v))
    if vals and max(vals) > threshold:
        return f"延迟 {field}.{stat} 最高 {max(vals):.0f}ms > 阈值 {threshold:.0f}ms"
    return None
