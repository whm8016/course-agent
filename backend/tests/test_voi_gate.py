"""阶段6：VoI 三因子三档门控 + cooldown。PRISM 版留作对照臂。

等价边界（value = p_need·severity·c，效用表 L_FULL=0.30 / L_LIGHT=0.12 / 0.6 系数）：
    value < 0.20        → silent（NetVoI 全负）
    0.20 <= value < 0.45 → light（一句轻提醒）
    value >= 0.45        → full（完整简报）
对照：PRISM 单边界 p_need >= 0.434（τ = 1/(1+3·p_need)），落在轻提醒带内部——
这就是两臂的消融叙事。
"""
import time

import pytest

from core.memory.proactive import decide_stitch, decide_stitch_prism

_PREREQ = {"ohm"}


def _m(risk_used: float, **extra) -> dict:
    # risk 固定 0.9（>= 0.5 候选阈值必入 C1），变量是 risk_used（班级收缩后的有效值）
    return {"ohm": {"risk": 0.9, "risk_used": risk_used, "p_forget": 1.0, **extra}}


# ── 三档边界 ────────────────────────────────────────────────────────────────


def test_voi_silent_below_value():
    brief = decide_stitch("kcl", _PREREQ, _m(0.10), set())
    assert brief.level == "silent"
    assert brief.should_stitch is False
    assert brief.reason == "below_value"


def test_voi_light_band():
    brief = decide_stitch("kcl", _PREREQ, _m(0.30), set())
    assert brief.level == "light"
    assert brief.should_stitch is True  # should ≡ level != silent


def test_voi_full_band():
    brief = decide_stitch("kcl", _PREREQ, _m(0.50), set())
    assert brief.level == "full"
    assert brief.should_stitch is True


# ── 因子各自的作用方向 ─────────────────────────────────────────────────────


def test_severity_amplifies():
    """severity=2.0（教师标注安全关键）：0.30 从 light 升 full。"""
    brief = decide_stitch("kcl", _PREREQ, _m(0.30), set(), severity=2.0)
    assert brief.level == "full"


def test_confidence_damps():
    """匹配含糊（c=0.5）：0.80 从 full 降到 light（value 0.40 落轻提醒带）。"""
    brief = decide_stitch("kcl", _PREREQ, _m(0.80), set(), confidence=0.5)
    assert brief.level == "light"


def test_low_confidence_can_silence():
    """c 足够低时 value 掉出轻提醒带 → silent。"""
    brief = decide_stitch("kcl", _PREREQ, _m(0.30), set(), confidence=0.2)
    assert brief.level == "silent"


# ── cooldown ───────────────────────────────────────────────────────────────


def test_cooldown_blocks_within_24h():
    now = time.time()
    brief = decide_stitch(
        "kcl", _PREREQ, _m(0.90, last_stitched_at=now - 3600), set(), now=now
    )
    assert brief.should_stitch is False
    assert brief.reason == "cooldown"


def test_cooldown_expires_after_24h():
    now = time.time()
    brief = decide_stitch(
        "kcl", _PREREQ, _m(0.90, last_stitched_at=now - 25 * 3600), set(), now=now
    )
    assert brief.should_stitch is True
    assert brief.level == "full"


def test_no_stitch_history_no_cooldown():
    brief = decide_stitch("kcl", _PREREQ, _m(0.90, last_stitched_at=None), set())
    assert brief.should_stitch is True


# ── PRISM 对照臂 ───────────────────────────────────────────────────────────


def test_prism_arm_single_boundary():
    """PRISM：p_need=0.30 < τ=0.434 → silent；VoI 同值 → light（边界带内差异）。"""
    prism = decide_stitch_prism("kcl", _PREREQ, _m(0.30), set())
    voi = decide_stitch("kcl", _PREREQ, _m(0.30), set())
    assert prism.should_stitch is False
    assert prism.level == "silent"
    assert voi.level == "light"


def test_prism_arm_full_on_high_pneed():
    prism = decide_stitch_prism("kcl", _PREREQ, _m(0.90), set())
    assert prism.should_stitch is True
    assert prism.level == "full"
