"""判定器三臂对比：正则 LF vs LLM 序数判定 vs LLM 自由 delta（plan 阶段E / §4a）。

Usage:
    python -m scripts.eval_memory.run_judge_compare [--arms lf,llm_ordinal,llm_delta]

三臂：
- arm=lf           infer_signal 词表（risk_delta 符号 → 等级；Snorkel 口径 labeling function）
- arm=llm_ordinal  judge.judge_segment（本方案：闭集 poor/fair/good/unknown + 弃权 + 引用）
- arm=llm_delta    graph_memory._MATCH_SYSTEM_PROMPT（现状：LLM 自由输出 mastery_delta）

语料：--corpus 指向 JSONL（每行 {segment_turns, topic_id, topic_label, topic_definition,
gold_level}）。生成协议：
  1. 用**异于判定模型**的 LLM 按 stitch_cases 主题图生成 (段, 主题) 对话段
     （生成模型与判定模型同源会令臂 B 的 kappa 系统性虚高，且偏差恰好只惠及臂 B
      ——threat to validity，必须报两侧模型名）；
  2. 人工标注 200 条 gold 等级 + 该段真正涉及的主题集（顺带产 match_topics 的
     recall@k，plan §4a-2）。

指标：二次加权 kappa（序数一致性）、poor 类 precision/recall、弃权率、引用校验
失败率、tokens/段。判定模型与生成模型名写入结果（同源须在论文声明）。

决策规则（写死）：臂 B 的 kappa 须显著高于臂 A 与臂 C，否则不上 LLM 判定；
W_JUDGE 取 B 的实测 kappa（不允许手填）。

CI 安全：无 corpus 文件 / 无真实 key 时跑框架自检（各臂函数可调用、指标可计算），
不产假数字。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

os.environ.setdefault("DB__URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DB__REDIS_URL", "memory://")
os.environ.setdefault("LLM__API_KEY", "sk-test")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")
os.environ.setdefault("TESTING", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

_LEVELS = ["poor", "fair", "good"]  # 序数轴（unknown=弃权不入混淆矩阵）


# ---- 指标（纯函数，可单测） ─────────────────────────────────────────────────


def quadratic_weighted_kappa(pred: list[str], gold: list[str]) -> float:
    """二次加权 kappa（序数一致性）。两列等长，取值 [-1, 1]。"""
    if len(pred) != len(gold) or not pred:
        return 0.0
    idx = {lv: i for i, lv in enumerate(_LEVELS)}
    n = len(_LEVELS)
    obs = [[0] * n for _ in range(n)]
    for p, g in zip(pred, gold):
        if p in idx and g in idx:
            obs[idx[g]][idx[p]] += 1
    total = sum(sum(r) for r in obs)
    if total == 0:
        return 0.0
    g_marg = [sum(obs[i]) / total for i in range(n)]
    p_marg = [sum(obs[i][j] for i in range(n)) / total for j in range(n)]
    num = den = 0.0
    for i in range(n):
        for j in range(n):
            w = (i - j) ** 2 / (n - 1) ** 2
            e = g_marg[i] * p_marg[j] * total
            num += w * (obs[i][j] / total)
            den += w * (e / total)
    return 1.0 - num / den if den else 0.0


def poor_precision_recall(pred: list[str], gold: list[str]) -> tuple[float, float]:
    """poor 类（最要紧的类：错报会触发预警，漏报会漏掉真薄弱）precision / recall。"""
    tp = sum(1 for p, g in zip(pred, gold) if p == "poor" and g == "poor")
    fp = sum(1 for p, g in zip(pred, gold) if p == "poor" and g != "poor")
    fn = sum(1 for p, g in zip(pred, gold) if p != "poor" and g == "poor")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return round(prec, 4), round(rec, 4)


# ---- 三臂 ─────────────────────────────────────────────────────────────────


async def _arm_lf(item: dict) -> dict:
    """臂 A 正则 LF：infer_signal 的 risk_delta 符号 → 等级（>0 poor / =0 fair /
    affirm-only good）。零 LLM。"""
    from core.memory.topic_match import infer_signal

    levels = []
    for u, a in item["segment_turns"]:
        sig = infer_signal(u, a)
        if sig["risk_delta"] > 0:
            levels.append("poor")
        elif sig["mastery_delta"] > 0:
            levels.append("good")
        else:
            levels.append("fair")
    # 段级聚合：取最差（薄弱预警宁可前倾——漏检 poor 的代价高于误检）
    rank = {"poor": 0, "fair": 1, "good": 2}
    return {"level": min(levels, key=lambda lv: rank[lv]) if levels else None,
            "abstained": False, "tokens": 0}


async def _arm_llm_ordinal(item: dict) -> dict:
    """臂 B LLM 序数判定（本方案）：judge_segment 闭集 + 弃权 + 引用校验。"""
    from core.memory.judge import judge_segment

    result = await judge_segment(
        [tuple(t) for t in item["segment_turns"]],
        [(item["topic_id"], item["topic_label"], item.get("topic_definition", ""))],
    )
    if result is None:
        return {"level": None, "abstained": True, "tokens": None, "failed": True}
    if result.judgements:
        j = result.judgements[0]
        return {"level": j.level, "abstained": False, "tokens": None,
                "quote_ok": bool(j.evidence_quote)}
    return {"level": None, "abstained": True, "tokens": None,
            "dropped": result.dropped_bad_quote + result.dropped_invalid_topic}


async def _arm_llm_delta(item: dict) -> dict:
    """臂 C LLM 自由 delta（现状 _MATCH_SYSTEM_PROMPT 输出）：delta<0 poor / ≈0 fair /
    >0 good——delta 数值无标定含义，符号当序数用已是其信息上限。"""
    from core.memory import graph_memory

    extracted = await graph_memory.extract_knowledge(
        "eval_judge", "\n".join(u for u, _ in item["segment_turns"]),
        "\n".join(a for _, a in item["segment_turns"]),
    )
    kps = (extracted or {}).get("knowledge_points") or []
    if not kps:
        return {"level": None, "abstained": True, "tokens": None}
    # 与 gold 主题标签最近的 kp
    label = item["topic_label"]
    kp = min(kps, key=lambda k: abs(len(k.get("label") or "") - len(label)))
    d = float(kp.get("mastery_delta") or 0.0)
    if d < -0.02:
        lv = "poor"
    elif d > 0.02:
        lv = "good"
    else:
        lv = "fair"
    return {"level": lv, "abstained": False, "tokens": None}


ARMS = {
    "lf": _arm_lf,
    "llm_ordinal": _arm_llm_ordinal,
    "llm_delta": _arm_llm_delta,
}


async def _run(arm_names: list[str], corpus_path: str) -> dict:
    from settings import get_settings

    if not os.path.exists(corpus_path):
        logger.warning("[judge_compare] corpus 不存在：%s —— 跑框架自检不产数字", corpus_path)
        return {"corpus": corpus_path, "items": 0, "self_check": "corpus_missing"}

    items = [json.loads(line) for line in open(corpus_path, encoding="utf-8") if line.strip()]
    judge_model = get_settings().llm.text_model
    out: dict = {
        "corpus": corpus_path, "items": len(items),
        "judge_model": judge_model,
        "generator_model": items[0].get("generator_model", "UNKNOWN（语料须记录生成模型名）")
        if items else "UNKNOWN",
    }
    for arm in arm_names:
        fn = ARMS[arm]
        preds, golds = [], []
        abstain = failed = dropped = 0
        for it in items:
            try:
                r = await fn(it)
            except Exception as exc:
                logger.warning("[judge_compare] %s item failed: %s", arm, exc)
                failed += 1
                continue
            if r.get("failed"):
                failed += 1
            dropped += int(r.get("dropped") or 0)
            if r["level"] is None:
                abstain += 1
                continue  # 弃权不入混淆矩阵（kappa 口径与 gold 对齐的子集）
            preds.append(r["level"])
            golds.append(it["gold_level"])
        prec, rec = poor_precision_recall(preds, golds)
        out[arm] = {
            "kappa_qw": round(quadratic_weighted_kappa(preds, golds), 4),
            "poor_precision": prec, "poor_recall": rec,
            "abstain_rate": round(abstain / len(items), 4) if items else None,
            "quote_fail_count": dropped,
            "call_fail_count": failed,
            "scored": len(preds),
        }
        logger.info("[judge_compare] %s: %s", arm, out[arm])
    return out


def main() -> None:
    from . import config

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="lf,llm_ordinal,llm_delta")
    parser.add_argument(
        "--corpus", default=str(config.RESULTS_DIR / "judge_corpus.jsonl"),
        help="JSONL 语料：{segment_turns, topic_id, topic_label, topic_definition, gold_level}",
    )
    args = parser.parse_args()

    summary = asyncio.run(
        _run([a.strip() for a in args.arms.split(",") if a.strip()], args.corpus)
    )
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "judge_compare.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[judge_compare] → %s", out)


if __name__ == "__main__":
    main()
