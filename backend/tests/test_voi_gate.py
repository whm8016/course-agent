"""阶段6/阶段D：VoI 三因子三档门控 + cooldown + logit 判据。PRISM 版留作对照臂。

等价边界（value = p_need·severity·c，效用表 L_FULL=0.30 / L_LIGHT=0.12 / 0.6 系数）：
    value < 0.20        → silent（NetVoI 全负）
    0.20 <= value < 0.45 → light（一句轻提醒）
    value >= 0.45        → full（完整简报）
对照：PRISM 单边界 p_need >= 0.434（τ = 1/(1+3·p_need)），落在轻提醒带内部——
这就是两臂的消融叙事。

阶段D 改动：C1 候选判据从 risk >= 0.5 换 logit 空间（theta - b <= WEAK_MARGIN≈
-0.4055，等价「至少两次判 poor」）。测试 mastery_map 用真实折叠器
（fold_observations）算出的状态，不再手拍 0.9。
"""
import time

from core.memory.mastery_state import W_JUDGE, fold_observations
from core.memory.proactive import (
    WEAK_MARGIN,
    decide_stitch,
    decide_stitch_prism,
)

_PREREQ = {"ohm"}


def _folded_map(levels: list[str], b_topic: float = 0.0) -> dict[str, dict]:
    """真实折叠：给定判定等级序列 → mastery_map 单主题条目（plan 阶段D 验收口径）。"""
    obs = [
        {"created_at": float(i), "outcome": {"poor": 0.0, "fair": 0.5, "good": 1.0}[lv],
         "weight": W_JUDGE, "source": "llm_judge"}
        for i, lv in enumerate(levels)
    ]
    s = fold_observations(obs)
    return {
        "ohm": {
            "risk": round(s.risk, 4),
            "risk_used": round(s.risk, 4),
            "logit": round(s.theta - b_topic, 4),
            "p_forget": 1.0,
            "observation_count": s.n_obs,
        }
    }


# ── 三档边界 ────────────────────────────────────────────────────────────────


def test_voi_silent_below_value():
    """候选在闸（2 次 poor）但 value 低（p_forget 小 → p_need 低）→ below_value 沉默。"""
    m = _folded_map(["poor", "poor"])
    m["ohm"]["p_forget"] = 0.15  # p_need ≈ 0.09 < 0.20 → silent
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.level == "silent"
    assert brief.should_stitch is False
    assert brief.reason == "below_value"


def test_voi_light_band():
    m = _folded_map(["poor"] * 6)
    # value = risk_used(≈0.80)·1·1 = 0.80 → full 带；压到轻提醒带用低 confidence
    brief = decide_stitch("kcl", _PREREQ, m, set(), confidence=0.42)
    assert brief.level == "light"
    assert brief.should_stitch is True


def test_voi_full_band():
    brief = decide_stitch("kcl", _PREREQ, _folded_map(["poor"] * 6), set())
    assert brief.level == "full"
    assert brief.should_stitch is True


# ── 阶段D 回归：fair 不入闸 / 1次poor不入闸 / 2次poor入闸 ────────────────────


def test_one_fair_not_candidate():
    """1 次 fair → logit≈0 远高于 WEAK_MARGIN → no_candidate（旧 >=0.5 会误入）。"""
    brief = decide_stitch("kcl", _PREREQ, _folded_map(["fair"]), set())
    assert brief.should_stitch is False
    assert brief.reason == "no_candidate"


def test_one_poor_not_candidate_two_poor_is():
    """语义锚点：1 次 poor 不入闸（logit -0.30 > WEAK_MARGIN -0.4055），
    2 次 poor 入闸（logit -0.54 <= WEAK_MARGIN）。"""
    one = decide_stitch("kcl", _PREREQ, _folded_map(["poor"]), set())
    assert one.reason == "no_candidate"
    two = decide_stitch("kcl", _PREREQ, _folded_map(["poor", "poor"]), set())
    assert two.reason in ("prereq_gap", "below_value")  # 入闸为候选（p_forget=1 时 below_value）


def test_two_poor_old_observation_silent_when_just_seen():
    """刚观测过（p_forget≈0）→ 候选在但 below_value（遗忘耦合：现在正学着呢）。"""
    m = _folded_map(["poor", "poor"])
    m["ohm"]["p_forget"] = 0.05
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.reason == "below_value"


# ── 因子各自的作用方向 ─────────────────────────────────────────────────────


def test_severity_amplifies():
    """severity=2.0（教师标注安全关键）：value 翻倍升档。"""
    m = _folded_map(["poor"] * 3)  # risk ≈ 0.66
    base = decide_stitch("kcl", _PREREQ, m, set(), confidence=0.5)
    amp = decide_stitch("kcl", _PREREQ, m, set(), confidence=0.5, severity=2.0)
    assert amp.level != "silent"
    assert _rank(amp.level) >= _rank(base.level)


def _rank(level: str) -> int:
    return {"silent": 0, "light": 1, "full": 2}[level]


def test_confidence_damps():
    m = _folded_map(["poor"] * 6)  # value=0.80 → full
    brief = decide_stitch("kcl", _PREREQ, m, set(), confidence=0.5)
    assert brief.level == "light"  # value 0.40 落轻提醒带


def test_low_confidence_can_silence():
    m = _folded_map(["poor"] * 3)
    brief = decide_stitch("kcl", _PREREQ, m, set(), confidence=0.2)
    assert brief.level == "silent"


# ── cooldown ─────────────────────────────────────────────────────────────


def test_cooldown_blocks_within_24h():
    now = time.time()
    m = _folded_map(["poor"] * 6)
    m["ohm"]["last_stitched_at"] = now - 3600
    brief = decide_stitch("kcl", _PREREQ, m, set(), now=now)
    assert brief.should_stitch is False
    assert brief.reason == "cooldown"


def test_cooldown_expires_after_24h():
    now = time.time()
    m = _folded_map(["poor"] * 6)
    m["ohm"]["last_stitched_at"] = now - 25 * 3600
    brief = decide_stitch("kcl", _PREREQ, m, set(), now=now)
    assert brief.should_stitch is True
    assert brief.level == "full"


def test_no_stitch_history_no_cooldown():
    brief = decide_stitch("kcl", _PREREQ, _folded_map(["poor"] * 6), set())
    assert brief.should_stitch is True


# ── PRISM 对照臂 ───────────────────────────────────────────────────────────


def test_prism_arm_single_boundary():
    """PRISM：p_need≈0.33 < τ≈0.5 → silent；VoI 同 value 在轻提醒带。"""
    m2 = _folded_map(["poor"] * 2)  # risk ≈ 0.63
    m2["ohm"]["p_forget"] = 0.53   # p_need ≈ 0.33
    prism = decide_stitch_prism("kcl", _PREREQ, m2, set())
    voi = decide_stitch("kcl", _PREREQ, m2, set())
    assert prism.should_stitch is False or prism.level == "full"  # 单边界
    assert voi.level in ("light", "silent")


def test_prism_arm_full_on_high_pneed():
    m = _folded_map(["poor"] * 6)
    prism = decide_stitch_prism("kcl", _PREREQ, m, set())
    assert prism.should_stitch is True
    assert prism.level == "full"


# ── WEAK_MARGIN 与折叠器的耦合纪律（阶段D 落点） ────────────────────────────


def test_weak_margin_is_negative_and_meaningful():
    """口径锚：WEAK_MARGIN ≈ logit(0.4) ≈ -0.4055（等价 risk >= 0.6）。"""
    import math
    assert math.isclose(WEAK_MARGIN, math.log(0.4 / 0.6), abs_tol=1e-4)
