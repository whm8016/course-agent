"""阶段1：topic_match 纯函数单测——行为信号推断 / 错题路由 / 偏好抽取（零 LLM）。"""
from core.memory.topic_match import (
    extract_preferences,
    infer_signal,
    should_route_error_llm,
)


# ── infer_signal ────────────────────────────────────────────────────────────


def test_infer_signal_confusion():
    s = infer_signal("欧姆定律我没看懂", "欧姆定律是...")
    assert "confusion" in s["signals"]
    assert s["risk_delta"] > 0
    assert should_route_error_llm(s) is True


def test_infer_signal_correction_beats_plain():
    """纠正语命中 correction（也是强信号）。"""
    s = infer_signal("不对，应该是串联分压", "抱歉，你说得对")
    assert "correction" in s["signals"]
    assert should_route_error_llm(s) is True
    # 纠正的 risk 加权应高于单纯困惑
    s_conf = infer_signal("不明白", "")
    assert s["risk_delta"] > s_conf["risk_delta"]


def test_infer_signal_affirm_improves_mastery():
    s = infer_signal("对了，明白了", "")
    assert "affirm" in s["signals"]
    assert s["mastery_delta"] > 0
    assert s["risk_delta"] == 0.0
    assert should_route_error_llm(s) is False


def test_infer_signal_repeat_from_history_hint():
    """同主题此前观测过（hint）又来问 = repeat 强信号，文本本身中性也路由。"""
    s = infer_signal("欧姆定律公式是什么", "", history_hint={"topic_repeat_count": 2})
    assert "repeat" in s["signals"]
    assert s["topic_repeat_count"] == 2
    assert should_route_error_llm(s) is True


def test_infer_signal_neutral_neither_routes_nor_deltas():
    s = infer_signal("欧姆定律公式是什么", "U=IR")
    assert s["signals"] == []
    assert s["risk_delta"] == 0.0
    assert s["mastery_delta"] == 0.0
    assert should_route_error_llm(s) is False


def test_infer_signal_preference_not_error_route():
    """偏好是 persona 线索，不是错误信号——不路由 LLM。"""
    s = infer_signal("我喜欢分步讲解", "")
    assert "preference" in s["signals"]
    assert should_route_error_llm(s) is False


# ── extract_preferences ─────────────────────────────────────────────────────


def test_extract_preferences_grabs_sentence():
    prefs = extract_preferences("前面讲得挺好。我喜欢分步讲解，一步步来")
    assert len(prefs) == 1
    assert "我喜欢分步讲解" in prefs[0]


def test_extract_preferences_none():
    assert extract_preferences("欧姆定律是什么") == []


# ── should_route_error_llm 边界 ────────────────────────────────────────────


def test_route_negative_ack():
    assert should_route_error_llm({"signals": ["negative_ack"]}) is True


def test_route_empty_signal():
    assert should_route_error_llm({"signals": []}) is False
