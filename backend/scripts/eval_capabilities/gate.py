"""质量门禁：按 capability 阈值检查分数（对齐 eval_rag/quality_gate 的语义）。

"最差值 ≥ 阈值"留 buffer：LLM-as-judge 二分类与人类仅 Cohen's κ≈0.3-0.5（10-20% 误判），
但排序相关 ρ≈0.8-0.9——judge 更适合排名，做硬门禁必须比目标松一档（见 config.QUALITY_GATES）。

本轮未评测的指标（scores 无该 key）跳过，不误判失败——与 eval_rag quality_gate 一致。
"""
from __future__ import annotations

from . import config


def check_gate(capability: str, scores: dict[str, float]) -> tuple[bool, list[str]]:
    """检查某 capability 的聚合分数是否满足 config.QUALITY_GATES。

    返回 (全部通过, 失败原因列表)。scores 为该 capability 各指标聚合值
    （如 chat 的 {accuracy, faithfulness}）。
    """
    gates = config.QUALITY_GATES.get(capability, {})
    failures: list[str] = []
    for metric, threshold in gates.items():
        val = scores.get(metric)
        if val is None:
            continue
        if val < threshold:
            failures.append(f"{capability}.{metric} = {val:.3f} < 阈值 {threshold}")
    return (not failures), failures
