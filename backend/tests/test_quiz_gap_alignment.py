"""门控↔出题闭环评测（run_stitch_quiz_loop）的程序化判定纯函数测试。

live 全栈评测由用户持真实凭据跑；这里只锁零凭据可跑的部分：
  - _judge_alignment 的定向判定口径（targeted 直中 / resolved 文本兜底 / 指名即准）
  - 正例筛选口径（gold.stitch=True 才出题）
"""
from __future__ import annotations


def _j(target: str, fallback: str, gold: str) -> dict:
    from scripts.eval_memory.run_stitch_quiz_loop import _judge_alignment

    return _judge_alignment(target, fallback, gold)


def test_targeted_direct_hit():
    """plan 指名 target_topic_id 且等于金标 → aligned（targeted 口径）。"""
    v = _j("电阻串并联电路", "别的", "电阻串并联电路")
    assert v["targeted_hit"] and v["aligned"] and not v["resolved_hit"]


def test_targeted_miss_not_rescued_by_text():
    """指名了但打偏 → 不 aligned：指名即准，topic 文本碰巧命中不兜底（口径不混算）。"""
    v = _j("示波器的原理与主要技术指标", "电阻串并联电路", "电阻串并联电路")
    assert not v["targeted_hit"] and not v["aligned"] and not v["resolved_hit"]


def test_resolved_fallback_when_no_target():
    """未指名（target 空）→ match_topics 解析 topic 文本命中金标 → aligned（resolved）。"""
    v = _j("", "电阻串并联电路", "电阻串并联电路")
    assert v["resolved_hit"] and v["aligned"] and not v["targeted_hit"]


def test_no_target_no_fallback_not_aligned():
    """未指名且文本解析不出 → 不 aligned。"""
    v = _j("", "", "电阻串并联电路")
    assert not v["aligned"]


def test_empty_gold_never_aligned():
    """金标无 evidence_topic（理论不出现：正例都带）→ 不 aligned，不误计。"""
    v = _j("x", "x", "")
    assert not v["targeted_hit"] and not v["resolved_hit"] and not v["aligned"]


def test_positive_filter_only_gold_stitch():
    """正例筛选口径：gold.stitch=True 才出题——直接锁 STITCH_CASES_LIVE 的正例数。"""
    from scripts.eval_memory.stitch_cases_table_e_live import STITCH_CASES_LIVE

    positives = [c for c in STITCH_CASES_LIVE if c["gold"]["stitch"]]
    assert len(positives) == 19  # 7 前置 + 3 复发 + 1 遗忘周 + 3 冷却 + 1 单次失手
    # + 2 多缺口 + 2 时钟三态（2026-08-18 档案冻结口径，新增 case 需同步改注释）
    assert all(c["gold"].get("evidence_topic") for c in positives)
