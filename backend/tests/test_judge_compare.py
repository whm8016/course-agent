"""阶段E：判定器三臂评测框架自检（零 LLM、零真实 key——真跑离线，见 run_judge_compare）。"""
import asyncio
import json
from pathlib import Path

from scripts.eval_memory.run_judge_compare import (
    _arm_lf,
    poor_precision_recall,
    quadratic_weighted_kappa,
)


def _item(u, a, gold="poor", topic="ohm"):
    return {
        "segment_turns": [[u, a]], "topic_id": topic,
        "topic_label": "欧姆定律", "topic_definition": "", "gold_level": gold,
    }


# ── 指标纯函数 ──────────────────────────────────────────────────────────────


def test_kappa_perfect_agreement():
    preds = ["poor", "fair", "good", "fair", "poor"]
    assert quadratic_weighted_kappa(preds, list(preds)) == 1.0


def test_kappa_zero_when_chance():
    """预测与 gold 独立（打乱后）→ kappa ≈ 0。"""
    gold = ["poor"] * 5 + ["fair"] * 5 + ["good"] * 5
    pred = ["good"] * 5 + ["poor"] * 5 + ["fair"] * 5  # 循环错位，无相关
    assert abs(quadratic_weighted_kappa(pred, gold)) < 0.5


def test_kappa_ordinal_penalty():
    """同错档数下，差两档的分布罚重于全差一档（kappa 更低）。"""
    gold = ["poor", "fair", "good", "poor", "fair"]
    k_one = quadratic_weighted_kappa(["fair", "good", "poor", "fair", "good"], gold)
    k_two = quadratic_weighted_kappa(["good", "good", "poor", "fair", "good"], gold)
    # 两组总错档数近同，但 k_two 两条差两档（权重 (2/2)²=1 vs (1/2)²=0.25）→ 罚更重
    assert k_two < k_one < 0.0


def test_poor_precision_recall():
    prec, rec = poor_precision_recall(
        ["poor", "poor", "fair"], ["poor", "good", "poor"]
    )
    assert prec == 0.5  # 2 报 poor，1 对
    assert rec == 0.5   # 2 真 poor，抓 1


# ── LF 臂（零 LLM 可真跑） ──────────────────────────────────────────────────


def test_arm_lf_confusion_maps_to_poor():
    r = asyncio.run(_arm_lf(_item("欧姆定律我没看懂", "U=IR")))
    assert r["level"] == "poor"


def test_arm_lf_affirm_maps_to_good():
    r = asyncio.run(_arm_lf(_item("对了，明白了", "很好")))
    assert r["level"] == "good"


def test_arm_lf_neutral_maps_to_fair():
    r = asyncio.run(_arm_lf(_item("欧姆定律公式是什么", "U=IR")))
    assert r["level"] == "fair"


# ── 框架自检（corpus 缺失不产假数字） ───────────────────────────────────────


def test_run_reports_corpus_missing(tmp_path: Path):
    from scripts.eval_memory.run_judge_compare import _run

    out = asyncio.run(_run(["lf"], str(tmp_path / "nonexistent.jsonl")))
    assert out["self_check"] == "corpus_missing"
    assert out["items"] == 0


def test_run_end_to_end_with_lf_arm(tmp_path: Path):
    """LF 臂零 LLM：小语料端到端可跑（kappa 计算路径打通）。"""
    from scripts.eval_memory.run_judge_compare import _run

    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps(_item(u, a, g)) for u, a, g in [
        ("欧姆定律我没看懂", "U=IR", "poor"),
        ("对了，明白了", "很好", "good"),
        ("欧姆定律公式是什么", "U=IR", "fair"),
    ]), encoding="utf-8")
    out = asyncio.run(_run(["lf"], str(corpus)))
    assert out["items"] == 3
    assert out["lf"]["kappa_qw"] == 1.0  # LF 完美区分这 3 例（gold 由规则生成）
    assert out["lf"]["abstain_rate"] == 0.0
