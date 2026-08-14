"""阶段3：遗忘耦合——p_forget = 1 - exp(-t/S) 换向（旧 eff_risk 方向反了）。

缺省自洽性（risk_used=0.5, S=5）：
  t=1  → p_forget≈0.181 → p_need≈0.091（τ≈0.78）→ 不拼（正在学，别打扰）
  t=30 → p_forget≈0.998 → p_need≈0.499（τ≈0.40）→ 拼（久未触碰，大概率忘了）
"""
import pytest

from core.memory.mastery import forget_prob
from core.memory.proactive import decide_stitch

_PREREQ = {"ohm"}  # kcl 的前置闭包
_MASTERY = {"ohm": {"risk": 0.5, "risk_used": 0.5, "p_forget": 0.0}}


def test_forget_prob_math():
    assert forget_prob(0) == pytest.approx(0.0)
    assert forget_prob(1, 5.0) == pytest.approx(1 - 2.718281828459045 ** (-1 / 5))
    assert forget_prob(30, 5.0) == pytest.approx(0.9975, abs=1e-3)
    assert forget_prob(10, 10.0) < forget_prob(10, 5.0)  # S 大忘得慢


def _gate(p_forget: float) -> tuple[bool, float]:
    brief = decide_stitch(
        "kcl", _PREREQ, {**_MASTERY, "ohm": {**_MASTERY["ohm"], "p_forget": p_forget}},
        set(),
    )
    return brief.should_stitch, brief.p_need


def test_recent_observation_no_stitch():
    """t=1（p_forget≈0.181）：刚观测过的薄弱点不拼——正学着呢。"""
    should, p_need = _gate(forget_prob(1, 5.0))
    assert p_need == pytest.approx(0.091, abs=0.002)
    assert should is False


def test_stale_observation_stitches():
    """t=30（p_forget≈0.998）：久未触碰的薄弱前置要拼。"""
    should, p_need = _gate(forget_prob(30, 5.0))
    assert p_need == pytest.approx(0.499, abs=0.002)
    assert should is True


def test_zero_forget_never_stitches_c1():
    """p_forget=0（刚摸过）→ C1 p_need=0，永不拼。"""
    should, p_need = _gate(0.0)
    assert p_need == 0.0
    assert should is False


def test_recurrence_ignores_forgetting():
    """C2 复发不乘遗忘：p_forget=0 也保底 0.7 → 拼。"""
    brief = decide_stitch(
        "ohm", set(),
        {"ohm": {"risk": 0.5, "risk_used": 0.5, "p_forget": 0.0}},
        {"ohm"},
    )
    assert brief.should_stitch is True
    assert brief.reason == "recurrence"
    assert brief.p_need == pytest.approx(0.7)


def test_strength_delays_forgetting_gate():
    """教师肯定过的（strength=6+1）：同 t=10 天，p_need 更低可能不过闸。"""
    p_weak = forget_prob(10, 5.0) * 0.5   # S=5
    p_strong = forget_prob(10, 8.0) * 0.5  # 被强化过
    assert p_strong < p_weak
