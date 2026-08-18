r"""表E runner：学情拼接门控 When 决策评测（四臂，vs 教师规则金标）。

Usage（backend 目录）:
    python -m scripts.eval_memory.run_stitch_table_e             # 零 LLM 三臂
    python -m scripts.eval_memory.run_stitch_table_e --with-llm  # 加 llm_self 臂（真实调用）

臂（评的都是「本轮该不该把未问及的跨会话学情拼进回答」这一决策）：
- never               永不拼（基线：正例全漏、负例全对）
- always              每轮都拼（基线：正例全中、负例全错）
- gate                现网门控 decide_stitch（零 LLM：折叠×遗忘 vs τ + 冷却）
- llm_self（可选）    把该生学情档案连同本轮问句交给 LLM 自决（**不给先修图**——
                      候选集结构正是门控的护城河；默认温度 3 次独立调用，多数票）

指标：
- 均衡准确率  (TPR+TNR)/2                     —— 类不平衡下不被多数类淹没
- 打扰率      FP/(FP+TN)  不该提却塞了学情     —— always=100%、never=0%
- 漏提醒率    FN/(FN+TP)  该提却沉默           —— never=100%、always=0%
- 代价        llm_calls/例 + 自一致率（llm 臂 3 次同输入决策一致的比例）

金标=教师规则句子（stitch_cases_table_e.teacher_rule，用户逐句点头），非算法公式；
门控与金标的偏离处（冷却类漏提醒）是评测的信息量所在，不是要修的 bug。
输出 results/stitch_table_e.json（逐例明细，论文失败分析数据源）+ 控制台 markdown 表。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

ZERO_LLM_ARMS = ("never", "always", "gate")

_LLM_RUNS = 3  # llm_self 臂同输入独立调用次数（自一致率 + 多数票）

_LEVEL_ZH = {"poor": "判未掌握", "fair": "判表现一般", "good": "判掌握"}

_LLM_SYSTEM = (
    "你是电路分析实验课的助教。下面给你一名学生的学情档案和本轮提问。"
    "请判断：本轮回答中，要不要顺带提醒学生一个他自己没问到的薄弱知识点"
    "（例如他要学的内容踩在没掌握的基础上时，回答开头提醒一句）。"
    '只输出 JSON：{"stitch": true 或 false, '
    '"evidence_topic": "要提醒的主题名（不提醒则为 null）", "reason": "一句话理由"}'
)


def _dossier(case: dict) -> str:
    """档案 → 助教可读文本。判定记录 + 距上次接触天数（与门控信息等价，加工方式不同：
    门控=公式折叠，LLM=自己读原文）。"""
    lines = [
        f"- {m['label']}：{'、'.join(_LEVEL_ZH[lv] for lv in m['levels'])}，"
        f"最近一次接触在 {m['days_since_last']} 天前"
        for m in case["mastery"]
    ]
    return "\n".join(lines) if lines else "- （无掌握度记录）"


async def _llm_decide_once(case: dict) -> dict:
    """LLM 自决一次（默认温度，无 seed——即兴判断的真实样子）。失败返回 {"failed": True}。"""
    from core.llm.json_extract import extract_json_from_llm
    from core.llm.llm import client as async_openai_client
    from settings import get_settings

    try:
        resp = await async_openai_client.chat.completions.create(
            model=get_settings().llm.text_model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": f"学情档案：\n{_dossier(case)}\n\n本轮提问：{case['query']}"},
            ],
            # 不设 temperature：走 provider 默认，保留即兴判断的自然波动
            #（自一致率是本臂的测量对象）。max_tokens 上限给推理模型留正文空间。
            max_tokens=8000,
            stream=False,
        )
        data = extract_json_from_llm(resp.choices[0].message.content or "")
        return {
            "stitch": bool(data.get("stitch")),
            "evidence_topic": data.get("evidence_topic"),
            "reason": data.get("reason"),
        }
    except Exception as exc:
        logger.warning("[table_e] llm 调用失败（记 failed，不重试）：%s", exc)
        return {"failed": True}


def _llm_vote(runs: list[dict]) -> dict:
    """多数票决策 + 自一致标记。平票（有效票各半）偏向 stitch=True——过度提醒者
    应为此付出打扰率代价，不替它遮掩。全失败 → stitch=None 不入混淆矩阵。"""
    valid = [r["stitch"] for r in runs if not r.get("failed")]
    if not valid:
        return {"stitch": None, "all_agree": None}
    stitch = sum(valid) * 2 >= len(valid)
    return {
        "stitch": bool(stitch),
        "all_agree": len(set(valid)) == 1,
    }


def _metrics(tp: int, fn: int, fp: int, tn: int) -> dict:
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "balanced_accuracy": round((tpr + tnr) / 2, 4),
        "interruption_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,  # 打扰率
        "miss_rate": round(fn / (fn + tp), 4) if (fn + tp) else None,          # 漏提醒率
    }


def _bump(stats: dict, arm: str, gold: bool, pred: bool) -> None:
    """混淆矩阵累加（三层表E runner 共用——同一阶梯复制三份曾致口径漂移）。"""
    s = stats[arm]
    if gold and pred:
        s["tp"] += 1
    elif gold and not pred:
        s["fn"] += 1
    elif not gold and pred:
        s["fp"] += 1
    else:
        s["tn"] += 1


async def run(with_llm: bool = False) -> dict:
    from .stitch_cases_table_e import STITCH_CASES, gate_decide, load_graph

    edges = load_graph()["edges"]
    arms = list(ZERO_LLM_ARMS) + (["llm_self"] if with_llm else [])
    stats = {a: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for a in arms}
    rows: list[dict] = []

    for c in STITCH_CASES:
        brief = gate_decide(c, edges)
        gold = c["gold"]["stitch"]
        row = {
            "id": c["id"], "label": c["label"], "teacher_rule": c["teacher_rule"],
            "query": c["query"], "gold_stitch": gold,
            "gold_evidence": c["gold"].get("evidence_topic"),
            "gate": {
                "stitch": brief.should_stitch, "reason": brief.reason,
                "p_need": round(brief.p_need, 4), "tau": round(brief.tau, 4),
                "evidence": brief.evidence_topic,
            },
        }
        preds: dict[str, bool | None] = {
            "never": False, "always": True, "gate": brief.should_stitch,
        }
        if with_llm:
            runs = [await _llm_decide_once(c) for _ in range(_LLM_RUNS)]
            vote = _llm_vote(runs)
            row["llm_self"] = {
                "runs": runs, "vote": vote["stitch"], "all_agree": vote["all_agree"],
            }
            preds["llm_self"] = vote["stitch"]  # None（全失败）不入混淆矩阵
        for arm, pred in preds.items():
            if pred is None:
                continue
            _bump(stats, arm, gold, pred)
        rows.append(row)

    summary: dict = {"suite": "table_e", "cases": len(STITCH_CASES), "arms": {}}
    for arm in arms:
        m = _metrics(**stats[arm])
        if arm == "llm_self":
            n_agree = sum(1 for r in rows if r["llm_self"]["all_agree"])
            scored = sum(1 for r in rows if r["llm_self"]["vote"] is not None)
            m["self_consistency"] = round(n_agree / scored, 4) if scored else None
            m["llm_calls_per_case"] = _LLM_RUNS
        else:
            m["llm_calls_per_case"] = 0
        summary["arms"][arm] = m
    summary["details"] = rows
    return summary


def _markdown_table(summary: dict) -> str:
    head = "| 臂 | 均衡准确率 | 打扰率 | 漏提醒率 | LLM调用/例 | 自一致率 |\n|---|---|---|---|---|---|"
    lines = []
    for arm, m in summary["arms"].items():
        sc = m.get("self_consistency")
        lines.append(
            f"| {arm} | {m['balanced_accuracy']:.4f} | "
            f"{m['interruption_rate']:.0%} | {m['miss_rate']:.0%} | "
            f"{m.get('llm_calls_per_case', '—')} | {sc if sc is not None else '—'} |"
        )
    return head + "\n" + "\n".join(lines)


def main() -> None:
    from . import config

    parser = argparse.ArgumentParser(description="表E：门控 When 决策四臂评测")
    parser.add_argument("--with-llm", action="store_true", help="加 llm_self 臂（真实 LLM 调用×3/例）")
    args = parser.parse_args()

    summary = asyncio.run(run(with_llm=args.with_llm))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "stitch_table_e.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[table_e] %d 例 -> %s\n%s", summary["cases"], out, _markdown_table(summary))
    for r in summary["details"]:
        g = r["gate"]
        if g["stitch"] != r["gold_stitch"]:
            logger.info(
                "[table_e] 偏离例 %s: gold=%s gate=%s reason=%s p_need=%s",
                r["id"], r["gold_stitch"], g["stitch"], g["reason"], g["p_need"],
            )


if __name__ == "__main__":
    main()
