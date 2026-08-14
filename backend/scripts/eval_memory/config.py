"""L3 记忆评测配置：门禁阈值 + 结果目录。

照 LongMemEval（ICLR 2025）的能力分类，对我们四层记忆系统做 knowledge updates /
abstention / decay 三维程序化判分（无需 LLM，可入 CI）。门禁阈值=1.0：这些是确定性
不变式（掌握度必须正确演进、无数据不得编造），全过才算达标。
"""
from __future__ import annotations

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 维度 → 最低通过率（passed/total）。确定性不变式场景，要求全过。
GATES: dict[str, float] = {
    "knowledge_update": 1.0,  # 掌握度随观测正确演进（改善/退步）
    "abstention": 1.0,        # 无数据/低置信时不得编造薄弱点
    "decay": 1.0,             # 旧观测软衰减（排序靠后，不物理删除）
    "stitch_gate": 1.0,       # 拼接门控 When 决策正确（正例拼/负例不拼，decide_stitch 纯函数）
}
