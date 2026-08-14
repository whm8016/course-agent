"""Elo/IRT 折叠器：从 mastery_observation 追加日志确定性算出掌握度状态。

感知与记账分离（plan「掌握度估计器重构」）：LLM 只做闭集序数判定（poor/fair/good/
unknown，judge.py），数字由本模块的确定性 Elo 折叠算出。选定 Elo 而非 BKT 的理由：
无训练数据（未上线）、观测稀疏时天然稳定（不确定度函数 U(n)）、题目难度参数
difficulty_b 本身就是班级先验（Pelánek, Computers & Education 2016）。

纯函数，无 IO——读写库由调用方（consolidation / 校准脚本）负责。

两条硬纪律（plan §阶段A）：
1. 折叠顺序相关（K_u 随 n_obs 递减、p 随 theta 变化），fold_observations 入口处
   显式按 created_at 升序排序，不依赖 SQL 返回顺序；
2. 迟到观测必须整体重折：增量更新只在「新观测就是最新一条」时等价。fold_observations
   设计为「吃全量日志、返回终态」，增量路径只是它的快捷方式，不是另一套实现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---- Pelánek 推荐起点 ----
A_U = 1.0        # 学生侧不确定度初值
B_RATE_U = 0.05  # 学生侧衰减率：U(n) = a / (1 + b*n)
K_FLOOR = 0.03   # U(n) 下界：长期用户不至于彻底停止更新（medical Elo 实践）
A_I = 0.6        # 题目侧不确定度初值（班级共享参数，更新慢于个人侧）
B_RATE_I = 0.05

# 判定等级 → outcome 映射（judge.py 写观测时用；折叠器只认 outcome）
LEVEL_OUTCOME = {"good": 1.0, "fair": 0.5, "poor": 0.0}

# 观测来源权重：quiz 有对错标签取 1.0；llm_judge 由阶段 E 三臂实测一致率决定
#（W_JUDGE，当前暂取 0.6，kappa 出来后须重定——重定时 WEAK_MARGIN 单测会红，
# 强制重扫常量）；rule 是词表兜底，precision 实测 < 0.6 整条删除，暂取保守值。
W_QUIZ = 1.0
W_JUDGE = 0.6
W_RULE = 0.3

# C2 复发的 poor 判据：outcome <= 此值（干净分开 poor=0.0 与 fair=0.5，避免浮点等值）
POOR_OUTCOME_MAX = 0.25


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class FoldState:
    """一个 (user, course, topic) 的折叠状态（学生侧 + 题目侧）。"""

    theta: float = 0.0
    n_obs: int = 0
    b_topic: float = 0.0
    n_item: int = 0
    _updated_b: bool = field(default=False, repr=False)

    @property
    def mastery(self) -> float:
        return sigmoid(self.theta - self.b_topic)

    @property
    def risk(self) -> float:
        return 1.0 - self.mastery

    @property
    def logit(self) -> float:
        """theta - b_topic：门控判据用 logit 空间（不被 sigmoid 压扁，plan §3e）。"""
        return self.theta - self.b_topic


def _u_student(n_obs: int) -> float:
    """学生侧不确定度 U(n) = max(0.03, a / (1 + b*n))。"""
    return max(K_FLOOR, A_U / (1.0 + B_RATE_U * n_obs))


def elo_update(
    state: FoldState, outcome: float, weight: float, source: str
) -> FoldState:
    """单条观测的 Elo 更新（纯函数，返回新状态不改入参）。

    theta += weight · K_u · (outcome - p)；题目侧仅 source == "quiz" 时更新
    （b_topic 是班级共享参数，LLM 判定是对某一学生某一段对话的观察，不冒充班级
    难度证据——plan §3b 修正项，同时避免冷启动两侧同时 K=1 导致差值位移翻倍）。
    """
    p = sigmoid(state.theta - state.b_topic)
    k_u = _u_student(state.n_obs)
    new_theta = state.theta + weight * k_u * (outcome - p)
    n_obs = state.n_obs + 1

    b_topic, n_item, updated_b = state.b_topic, state.n_item, state._updated_b
    if source == "quiz":
        k_i = A_I / (1.0 + B_RATE_I * state.n_item)
        b_topic = state.b_topic - weight * k_i * (outcome - p)
        n_item = state.n_item + 1
        updated_b = True

    return FoldState(new_theta, n_obs, b_topic, n_item, updated_b)


def fold_observations(
    obs_list: list[dict], init: FoldState | None = None
) -> FoldState:
    """全量日志 → 终态。obs 每条 {created_at, outcome, weight, source}。

    入口显式按 created_at 升序排（Elo 顺序相关，近期证据权重更大是预期行为）；
    调用方吃全量日志即可，迟到观测整体重折天然幂等。
    """
    state = init or FoldState()
    for obs in sorted(obs_list, key=lambda o: o["created_at"]):
        state = elo_update(
            state, float(obs["outcome"]), float(obs.get("weight", 1.0)), str(obs.get("source", ""))
        )
    return state


__all__ = [
    "FoldState",
    "sigmoid",
    "elo_update",
    "fold_observations",
    "LEVEL_OUTCOME",
    "POOR_OUTCOME_MAX",
    "W_QUIZ",
    "W_JUDGE",
    "W_RULE",
]
