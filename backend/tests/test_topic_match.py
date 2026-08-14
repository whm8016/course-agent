"""阶段0：topic_match 纯函数单测。

infer_signal 已退出掌握度主路径，降级为 Snorkel 口径的 labeling function——本文件
只断言 LF 的**召回口径**（evidence 命中与否），不断言 risk_delta / mastery_delta
数值（0.15 / 0.20 等即将失去意义，权重须由阶段 E 实测 precision 决定）。
match_topics / rank_by_similarity 的用例在 test_consolidation_topic_match.py。
"""
from core.memory.topic_match import infer_signal


# ── infer_signal（LF 召回口径：只看 evidence 命中） ─────────────────────────


def test_infer_signal_confusion():
    assert "confusion" in infer_signal("欧姆定律我没看懂", "欧姆定律是...")["signals"]


def test_infer_signal_correction():
    assert "correction" in infer_signal("不对，应该是串联分压", "抱歉，你说得对")["signals"]


def test_infer_signal_affirm():
    assert "affirm" in infer_signal("对了，明白了", "")["signals"]


def test_infer_signal_repeat_from_history_hint():
    """同主题此前观测过（hint）又来问 = repeat 信号。"""
    s = infer_signal("欧姆定律公式是什么", "", history_hint={"topic_repeat_count": 2})
    assert "repeat" in s["signals"]
    assert s["topic_repeat_count"] == 2


def test_infer_signal_negative_ack():
    assert "negative_ack" in infer_signal("还是错", "")["signals"]


def test_infer_signal_neutral_no_signal():
    assert infer_signal("欧姆定律公式是什么", "U=IR")["signals"] == []


def test_infer_signal_preference_not_mastery_signal():
    """偏好是 persona 线索，不参与 mastery 口径（confidence 只作 LF 内部证据）。"""
    s = infer_signal("我喜欢分步讲解", "")
    assert "preference" in s["signals"]


# ── extract_preferences 已删（persona 走 mem0 原生抽取，见 test_promote_two_slots）──
