"""阶段A：Elo 折叠器单测（无 LLM、无 DB，纯函数）。

关键纪律：
- 折叠顺序相关（近期证据权重大）是预期行为，测试防「优化成顺序无关」；
- WEAK_MARGIN 必须由 fold_observations 实算夹出（1 次 poor 不入闸、2 次入闸），
  weight 变了此测试会红，强制重扫常量（plan §3e）。
"""
import math

from core.memory.mastery_state import (
    A_U,
    B_RATE_U,
    K_FLOOR,
    W_JUDGE,
    FoldState,
    elo_update,
    fold_observations,
    sigmoid,
)


def _poor(ts: float, i: int) -> dict:
    return {"created_at": ts, "outcome": 0.0, "weight": W_JUDGE, "source": "llm_judge"}


def _good(ts: float, i: int) -> dict:
    return {"created_at": ts, "outcome": 1.0, "weight": W_JUDGE, "source": "llm_judge"}


# ── 重放一致性（幂等的正确含义） ────────────────────────────────────────────


def test_fold_deterministic_same_log_twice():
    log = [_poor(1.0, 0), _good(2.0, 1), _poor(3.0, 2)]
    assert fold_observations(log).theta == fold_observations(log).theta


def test_append_then_refold_equals_incremental():
    """日志追加一条后整体重折 == 在旧状态上增量更新该条。"""
    log = [_poor(1.0, 0), _good(2.0, 1)]
    state2 = fold_observations(log)

    late = _poor(3.0, 2)  # 新观测就是最新一条 → 增量路径等价
    refolded = fold_observations(log + [late])
    incremental = elo_update(state2, late["outcome"], late["weight"], late["source"])
    assert math.isclose(refolded.theta, incremental.theta, abs_tol=1e-12)
    assert refolded.n_obs == incremental.n_obs == 3


def test_fold_sorts_by_created_at_regardless_of_input_order():
    """入口显式排序：乱序输入与按序输入结果一致（不依赖 SQL 返回顺序）。"""
    log = [_poor(3.0, 2), _good(1.0, 0), _poor(2.0, 1)]
    assert fold_observations(log).theta == fold_observations(
        [_good(1.0, 0), _poor(2.0, 1), _poor(3.0, 2)]
    ).theta


# ── 顺序敏感性（防「优化」成顺序无关） ────────────────────────────────────


def test_order_matters_recent_evidence_weighs_more():
    """poor→good 与 good→poor 结果不等，符号符合实算（近期证据权重更大）。"""
    pg = fold_observations([_poor(1.0, 0), _good(2.0, 1)])
    gp = fold_observations([_good(1.0, 0), _poor(2.0, 1)])
    assert not math.isclose(pg.theta, gp.theta, abs_tol=1e-9)
    # 后发生的事件方向主导：先 poor 后 good 微正，先 good 后 poor 微负
    assert pg.theta > 0 > gp.theta


# ── 基本性质 ────────────────────────────────────────────────────────────────


def test_weight_zero_no_state_change():
    s0 = FoldState(theta=0.7, n_obs=5, b_topic=0.2, n_item=3)
    s1 = elo_update(s0, 0.0, 0.0, "llm_judge")
    assert math.isclose(s1.theta, s0.theta, abs_tol=1e-12)
    assert s1.n_obs == s0.n_obs + 1  # 计数仍进（观测发生过），数值不动


def test_u_floor_003_kicks_in():
    """U(n) 下界 0.03：n 足够大后位移收敛到 weight·0.03·|outcome-p| 量级。"""
    s = FoldState(theta=2.0, n_obs=int((A_U / K_FLOOR - 1) / B_RATE_U) + 100, b_topic=0.0)
    before = s.theta
    s1 = elo_update(s, 1.0, 1.0, "quiz")
    # p = sigmoid(2) ≈ 0.88，位移 = 1·0.03·(1-0.88) ≈ 0.0036，远小于无下界时
    assert s1.theta - before <= 1.0 * K_FLOOR * 0.13


def test_n_obs_up_same_obs_moves_less():
    """观测越多同一观测位移越小（U(n) 单调递减）。"""
    deltas = []
    for n in (0, 5, 50):
        s0 = FoldState(theta=0.0, n_obs=n)
        s1 = elo_update(s0, 1.0, 1.0, "llm_judge")
        deltas.append(s1.theta - s0.theta)
    assert deltas[0] > deltas[1] > deltas[2]


# ── 题目难度侧：仅 quiz 可更新（plan §3b 修正项） ──────────────────────────


def test_llm_judge_never_touches_b_topic():
    log = [_poor(1.0, 0), _poor(2.0, 1), _poor(3.0, 2)]
    s = fold_observations(log)
    assert s.b_topic == 0.0 and s.n_item == 0


def test_quiz_updates_b_topic_and_n_item():
    obs = [{"created_at": 1.0, "outcome": 0.0, "weight": 1.0, "source": "quiz"}]
    s = fold_observations(obs)
    assert s.n_item == 1
    assert s.b_topic > 0.0  # 答错 → 题变难（b 上移）


def test_quiz_cold_start_single_side_displacement():
    """冷启动首条判分观测：theta 与 b 各动一侧，logit 位移 = weight·K·(outcome-p)
    的两侧之和但 K 由各自 n 驱动（不出现两侧同时 K=1 的翻倍）。"""
    obs = [{"created_at": 1.0, "outcome": 0.0, "weight": 1.0, "source": "quiz"}]
    s = fold_observations(obs)
    # p=0.5（无先验），theta -= 1·A_U·0.5 = -0.5；b += 1·A_I·0.5 = +0.3 → logit = -0.8
    assert math.isclose(s.theta, -0.5, abs_tol=1e-9)
    assert math.isclose(s.b_topic, 0.3, abs_tol=1e-9)


# ── 派生量 ──────────────────────────────────────────────────────────────────


def test_mastery_risk_logit_derived():
    s = FoldState(theta=1.0, b_topic=0.5, n_obs=10)
    assert math.isclose(s.mastery, sigmoid(0.5))
    assert math.isclose(s.risk, 1 - sigmoid(0.5))
    assert math.isclose(s.logit, 0.5)


# ── WEAK_MARGIN 纪律：由折叠器实算夹出（plan §3e） ─────────────────────────


def _weak_margin_from_proactive() -> float:
    from core.memory.proactive import WEAK_MARGIN

    return WEAK_MARGIN


def test_weak_margin_sandwiched_between_1_and_2_poor():
    """WEAK_MARGIN 必须落在 1 次 poor 与 2 次 poor 的 logit 之间：
    1 次不入闸、2 次入闸（「至少两次判 poor」语义锚点）。weight 改了这里会红。"""
    m = _weak_margin_from_proactive()
    one = fold_observations([_poor(1.0, 0)]).logit
    two = fold_observations([_poor(1.0, 0), _poor(2.0, 1)]).logit
    assert one > m > two, (
        f"WEAK_MARGIN={m} 未夹在 1次poor({one:.4f}) 与 2次poor({two:.4f}) 之间——"
        "weight 调整后须重定常量（plan §3e）"
    )


def test_weak_margin_equivalent_risk_ge_06():
    """口径锚：WEAK_MARGIN = logit(0.4) ≈ -0.4055（等价 risk >= 0.6，便于人读）。"""
    m = _weak_margin_from_proactive()
    assert math.isclose(m, math.log(0.4 / 0.6), abs_tol=1e-4)


def test_fair_lands_far_from_margin():
    """1 次 fair 观测 logit≈0，远离 WEAK_MARGIN（不会被误判成前置缺口）。"""
    m = _weak_margin_from_proactive()
    fair = fold_observations([_fair_obs()]).logit
    assert fair > m + 0.3


def _fair_obs() -> dict:
    return {"created_at": 1.0, "outcome": 0.5, "weight": W_JUDGE, "source": "llm_judge"}
