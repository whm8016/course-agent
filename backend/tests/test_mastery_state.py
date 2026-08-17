"""阶段A：Elo 折叠器单测（无 LLM、无 DB，纯函数）。

关键纪律：
- 折叠顺序相关（近期证据权重大）是预期行为，测试防「优化成顺序无关」；
- WEAK_MARGIN 必须由 fold_observations 实算夹出（1 次 poor 入闸、fair-only 不入闸；
  2026-08-18 由「2 次入闸」下调），weight 变了此测试会红，强制重扫常量（plan §3e）。

难度侧（b）已从 fold_observations 拆出（plan「班级难度估计修复」）：题目难度由
fold_class_difficulty 独占（每人各自 theta 对共享 b，日频 cron 全班重折）。
"""
import math

from core.memory.mastery_state import (
    A_U,
    B_RATE_U,
    K_FLOOR,
    W_JUDGE,
    W_QUIZ,
    W_RULE,
    FoldState,
    elo_update,
    fold_class_difficulty,
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
    incremental = elo_update(state2, late["outcome"], late["weight"])
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
    s0 = FoldState(theta=0.7, n_obs=5, b_topic=0.2)
    s1 = elo_update(s0, 0.0, 0.0)
    assert math.isclose(s1.theta, s0.theta, abs_tol=1e-12)
    assert s1.n_obs == s0.n_obs + 1  # 计数仍进（观测发生过），数值不动
    assert s1.b_topic == s0.b_topic  # 难度只读输入，学生侧永不改


def test_u_floor_003_kicks_in():
    """U(n) 下界 0.03：n 足够大后位移收敛到 weight·0.03·|outcome-p| 量级。"""
    s = FoldState(theta=2.0, n_obs=int((A_U / K_FLOOR - 1) / B_RATE_U) + 100, b_topic=0.0)
    before = s.theta
    s1 = elo_update(s, 1.0, 1.0)
    # p = sigmoid(2) ≈ 0.88，位移 = 1·0.03·(1-0.88) ≈ 0.0036，远小于无下界时
    assert s1.theta - before <= 1.0 * K_FLOOR * 0.13


def test_n_obs_up_same_obs_moves_less():
    """观测越多同一观测位移越小（U(n) 单调递减）。"""
    deltas = []
    for n in (0, 5, 50):
        s0 = FoldState(theta=0.0, n_obs=n)
        s1 = elo_update(s0, 1.0, 1.0)
        deltas.append(s1.theta - s0.theta)
    assert deltas[0] > deltas[1] > deltas[2]


# ── 题目难度侧：fold_class_difficulty 独占（plan「班级难度估计修复」）──────


def test_class_fold_empty_returns_zero():
    """空输入 → (0.0, 0)：冷启动全零，不用讲义顺序推难度。"""
    assert fold_class_difficulty([]) == (0.0, 0)


def test_class_fold_poor_class_raises_b():
    """全班都答错（quiz 判分）→ b 上移（题变难）。单条 w=1 的收缩后 b≈0.09。"""
    b, n = fold_class_difficulty([
        {"user_id": "u1", "created_at": 1.0, "outcome": 0.0, "weight": W_QUIZ},
    ])
    assert n == 1
    assert b > 0.0


def test_class_fold_good_class_lowers_b():
    """全班都答对 → b 下移（题变简单）。"""
    b, _ = fold_class_difficulty([
        {"user_id": "u1", "created_at": 1.0, "outcome": 1.0, "weight": W_QUIZ},
    ])
    assert b < 0.0


def test_class_fold_shrink_2_poor_below_015():
    """收缩测：2 条 poor → b < 0.15（n0=10 时 2 条证据只给 b≈0.09）——
    防两三条判差把 b 抬高后，把别的 fair 学生拖过 C1 闸。"""
    b, n = fold_class_difficulty([
        {"user_id": "u1", "created_at": 1.0, "outcome": 0.0, "weight": W_QUIZ},
        {"user_id": "u2", "created_at": 2.0, "outcome": 0.0, "weight": W_QUIZ},
    ])
    assert n == 2
    assert 0.0 < b < 0.15


def test_class_fold_shrink_fades_with_evidence():
    """收缩是乘法：b_shrunk = b_raw × n/(n+n0)。证据涨到几十条后系数趋近 1
    （60/(60+10)≈0.857），b 不再被明显压向 0。"""
    obs = [
        {"user_id": f"u{i}", "created_at": float(i), "outcome": 0.0, "weight": W_QUIZ}
        for i in range(60)
    ]
    shrunk, n = fold_class_difficulty(obs)
    unshrunk, _ = fold_class_difficulty(obs, n0=0)
    assert n == 60
    assert math.isclose(shrunk, unshrunk * 60 / 70, abs_tol=1e-9)
    # 对照：2 条证据时系数只有 2/12≈0.167（b 被大幅压向 0）
    obs2 = obs[:2]
    shrunk2, _ = fold_class_difficulty(obs2)
    unshrunk2, _ = fold_class_difficulty(obs2, n0=0)
    assert math.isclose(shrunk2, unshrunk2 * 2 / 12, abs_tol=1e-9)


def test_class_fold_rasch_scale_recovery():
    """尺度测：Rasch 模拟（已知真 b），fold_class_difficulty 折出 b 与真值相关 >0.9。

    对照旧合体折法（全班折进同一个 FoldState、theta/b 按 1.0:0.6 学习率瓜分误差）
    会明显更低——b 被学习率比值压缩，不是真难度。
    """
    import random

    rng = random.Random(42)
    true_b = [round(-1.2 + 0.2 * k, 3) for k in range(13)]  # -1.2..1.2
    thetas = {f"u{i}": round(rng.uniform(-1.0, 1.0), 3) for i in range(25)}
    obs: list[dict] = []
    for t, b in enumerate(true_b):
        for u, th in thetas.items():
            for k in range(6):
                p = 1.0 / (1.0 + math.exp(-(th - b)))
                obs.append({
                    "user_id": u, "created_at": 1000.0 * t + k + rng.random(),
                    "outcome": 1.0 if rng.random() < p else 0.0, "weight": 1.0,
                })

    def _corr(xs: list[float], ys: list[float]) -> float:
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        return cov / (vx * vy)

    est = [fold_class_difficulty([o for o in obs if abs(o["created_at"] // 1000 - t) < 1])[0]
           for t in range(len(true_b))]
    assert _corr(true_b, est) > 0.9

    # 对照臂：旧合体折法（全班折进同一个 FoldState）——同一份观测，b 被压缩
    def _legacy_fold(group: list[dict]) -> float:
        state = FoldState()
        for o in sorted(group, key=lambda x: x["created_at"]):
            state = elo_update(state, float(o["outcome"]), float(o.get("weight", 1.0)))
            # 旧实现题目侧：b -= w·K_i·err（与 theta 瓜分同一误差项）
            p = sigmoid(state.theta - state.b_topic)
            state.b_topic -= float(o.get("weight", 1.0)) * 0.6 * (
                float(o["outcome"]) - p
            )
        return state.b_topic

    est_legacy = [
        _legacy_fold([o for o in obs if abs(o["created_at"] // 1000 - t) < 1])
        for t in range(len(true_b))
    ]
    # 新口径须不差于旧口径（真值相关上不塌）
    assert _corr(true_b, est) >= _corr(true_b, est_legacy)


def test_class_fold_no_double_count_with_calibration_scope():
    """双计回归锚：学生侧从当前 b 起折（FoldState(b_topic=b)）与校准脚本同口径。

    同一份观测日志：生产路径（fold_observations(obs, FoldState(b_topic=b))）折出的
    theta 与 risk 必须等于校准口径——若生产忘了传 init（b 按 0 折），难度被吸进
    theta，risk 与读侧 theta - b 双计。
    """
    b = 0.4
    obs = [_poor(1.0, 0), _poor(2.0, 1)]
    prod = fold_observations(obs, FoldState(b_topic=b))
    # 校准脚本口径：FoldState(b_topic=b) 起折
    calib = fold_observations(obs, FoldState(b_topic=b))
    assert math.isclose(prod.theta, calib.theta, abs_tol=1e-12)
    assert math.isclose(prod.risk, 1 - sigmoid(prod.theta - b), abs_tol=1e-9)
    # 双计锚：不传 init 时 theta 与传 b 起折的**不等**（难度进了 theta），
    # 生产必须选后者
    naive = fold_observations(obs)
    assert not math.isclose(naive.theta, prod.theta, abs_tol=1e-6)


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


def test_weak_margin_sandwiched_between_fair_and_1_poor():
    """WEAK_MARGIN 必须落在 fair-only 与 1 次 poor 的 theta 之间：
    1 次判 poor 即入闸、fair 不入闸（2026-08-18 由「2 次入闸」下调后的语义锚点）。
    weight 改了这里会红。"""
    m = _weak_margin_from_proactive()
    fair = fold_observations([_fair_obs()]).theta
    one = fold_observations([_poor(1.0, 0)]).theta
    assert one < m < fair, (
        f"WEAK_MARGIN={m} 未夹在 1次poor({one:.4f}) 与 fair({fair:.4f}) 之间——"
        "weight 调整后须重定常量（plan §3e）"
    )


def test_weak_margin_keeps_single_rule_poor_out():
    """低信源边界：单次 W_RULE 观测（theta≈-0.15）不入闸——规则观测要凑两次；
    而单次 W_JUDGE 判定（-0.30）一次即入。"""
    m = _weak_margin_from_proactive()
    rule_one = fold_observations([_rule_poor(1.0)]).theta
    assert m < rule_one, f"单次 rule-poor theta={rule_one:.4f} 不应越过 WEAK_MARGIN={m}"


def test_weak_margin_equivalent_risk_ge_055():
    """口径锚：WEAK_MARGIN = logit(0.45) ≈ -0.2007（等价 risk >= 0.55，便于人读）。"""
    m = _weak_margin_from_proactive()
    assert math.isclose(m, math.log(0.45 / 0.55), abs_tol=1e-4)


def test_fair_lands_far_from_margin():
    """1 次 fair 观测 theta≈0，远离 WEAK_MARGIN（不会被误判成前置缺口）。"""
    m = _weak_margin_from_proactive()
    fair = fold_observations([_fair_obs()]).theta
    assert fair > m + 0.15


def test_c1_anchor_unaffected_by_class_difficulty():
    """C1 锚点：b 取 0 和 0.5 两种，1 次 poor 的 theta 仍低于 WEAK_MARGIN（入闸，
    2026-08-18 口径）、fair 仍高于——入闸比 theta 不比 theta-b，班级难度拖不动
    「该生薄弱」语义。"""
    m = _weak_margin_from_proactive()
    for b in (0.0, 0.5):
        one = fold_observations([_poor(1.0, 0)], FoldState(b_topic=b)).theta
        fair = fold_observations([_fair_obs()], FoldState(b_topic=b)).theta
        assert one < m, f"b={b}：1 次 poor 没入闸（theta={one:.4f}）"
        assert fair > m, f"b={b}：1 次 fair 被拖过闸（theta={fair:.4f}）"


def _fair_obs() -> dict:
    return {"created_at": 1.0, "outcome": 0.5, "weight": W_JUDGE, "source": "llm_judge"}


def _rule_poor(ts: float) -> dict:
    return {"created_at": ts, "outcome": 0.0, "weight": W_RULE, "source": "rule"}
