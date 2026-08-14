"""阶段3：遗忘耦合——p_forget = 1 - exp(-t/S) 换向（旧 eff_risk 方向反了）。

阶段D 口径：mastery_map 用真实折叠状态（3 次 poor → theta -0.74 / risk 0.68），
入闸靠 theta <= WEAK_MARGIN（不是旧 risk >= 0.5；不用 theta - b——班级难度不决定
谁能入闸）。缺省自洽性（S=5）：
  t=1  → p_forget≈0.181 → p_need≈0.12 → 不拼（正在学，别打扰）
  t=30 → p_forget≈0.998 → p_need≈0.67 → 拼（久未触碰，大概率忘了）
"""
import pytest

from core.memory.mastery import forget_prob
from core.memory.mastery_state import W_JUDGE, fold_observations
from core.memory.proactive import decide_stitch

_PREREQ = {"ohm"}  # kcl 的前置闭包


def _folded_ohm(levels=("poor",) * 3) -> dict[str, dict]:
    obs = [
        {"created_at": float(i), "outcome": {"poor": 0.0, "fair": 0.5, "good": 1.0}[lv],
         "weight": W_JUDGE}
        for i, lv in enumerate(levels)
    ]
    s = fold_observations(obs)
    return {
        "ohm": {
            "risk": round(s.risk, 4), "risk_used": round(s.risk, 4),
            "theta": round(s.theta, 4), "logit": round(s.logit, 4),
            "p_forget": 0.0, "observation_count": s.n_obs,
        }
    }


def test_forget_prob_math():
    assert forget_prob(0) == pytest.approx(0.0)
    assert forget_prob(1, 5.0) == pytest.approx(1 - 2.718281828459045 ** (-1 / 5))
    assert forget_prob(30, 5.0) == pytest.approx(0.9975, abs=1e-3)
    assert forget_prob(10, 10.0) < forget_prob(10, 5.0)  # S 大忘得慢


def _gate(p_forget: float) -> tuple[bool, float]:
    m = _folded_ohm()
    m["ohm"]["p_forget"] = p_forget
    brief = decide_stitch("kcl", _PREREQ, m, set())
    return brief.should_stitch, brief.p_need


def test_recent_observation_no_stitch():
    """t=1（p_forget≈0.181）：刚观测过的薄弱点不拼——正学着呢。risk≈0.678。"""
    should, p_need = _gate(forget_prob(1, 5.0))
    assert p_need == pytest.approx(0.678 * 0.181, abs=0.005)
    assert should is False


def test_stale_observation_stitches():
    """t=30（p_forget≈0.998）：久未触碰的薄弱前置要拼。"""
    should, p_need = _gate(forget_prob(30, 5.0))
    assert p_need == pytest.approx(0.678 * 0.998, abs=0.005)
    assert should is True


def test_zero_forget_never_stitches_c1():
    """p_forget=0（刚摸过）→ C1 p_need=0，永不拼。"""
    should, p_need = _gate(0.0)
    assert p_need == 0.0
    assert should is False


def test_recurrence_ignores_forgetting():
    """C2 复发不乘遗忘：p_forget=0 也保底 0.7 → 拼。"""
    m = _folded_ohm(("fair",))  # fair 观测（复发与薄弱判据无关）
    m["ohm"]["p_forget"] = 0.0
    brief = decide_stitch("ohm", set(), m, {"ohm"})
    assert brief.should_stitch is True
    assert brief.reason == "recurrence"
    assert brief.p_need == pytest.approx(0.7)


def test_strength_delays_forgetting_gate():
    """教师肯定过的（strength=6+1）：同 t=10 天，p_need 更低可能不过闸。"""
    p_weak = forget_prob(10, 5.0) * 0.5   # S=5
    p_strong = forget_prob(10, 8.0) * 0.5  # 被强化过
    assert p_strong < p_weak
