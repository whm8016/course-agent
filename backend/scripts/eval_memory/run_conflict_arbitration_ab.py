"""双臂对比：矛盾检测 规则（现状） vs 规则+灰区 LLM 仲裁（P1 新路径）。

内置人工标注记忆对数据集（同 kind=persona、时间差 >7 天满足规则臂门槛），逐对调
_detect_and_clean_conflicts——规则臂关仲裁、LLM 臂开仲裁——对比「剔旧」判定的
precision/recall/F1。数据集覆盖三个相似度区（实测 SequenceMatcher 值）：
- rule 区（>0.85）：字面更新，规则臂应独立抓到
- gray 区（[0.4, 0.85)）：语义矛盾但措辞不同（规则盲区=LLM 臂的增量）+ 互补对
  （LLM 臂的误杀风险）
- below 区（<0.4）：典型偏好反转但相似度太低——两臂设计上均不覆盖，单列暴露
  conflict_llm_sim_floor 的已知缺口，供后续调参参考

产出 backend/scripts/eval_memory/results/conflict_arbitration_ab.json。
--rule-only 可无 LLM 干跑（自测数据集与规则臂口径），LLM 臂真实跑由用户执行。
决策规则（先写死再跑）：LLM 臂 F1 > 规则臂 F1 且 precision >= 0.8 → 建议开默认；
否则维持默认关。
"""
import argparse
import asyncio
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, ".")

from settings import get_settings  # noqa: E402,F401  (触发 .env 加载)

RESULTS_DIR = Path(__file__).parent / "results"

# (旧文本, 新文本, expect)——expect: CONFLICT=应剔旧 / KEEP=应两条都保留。
# 相似度实测值写在注释；改动文本须重算并更新注释。
_DATASET = [
    # ── rule 区（>0.85）：规则臂应抓到 ──
    ("学生使用第七版电路教材", "学生使用第八版电路教材", "CONFLICT"),          # 0.909
    # ── gray 区：语义矛盾（规则盲区）──
    ("学生偏好快速的讲解风格与简洁的总结", "学生偏好缓慢的讲解风格与详细的推导", "CONFLICT"),  # 0.647
    ("学生下周三有实验考试", "学生下周三有理论考试", "CONFLICT"),              # 0.800
    ("学生没有示波器可用", "学生刚借到一台示波器", "CONFLICT"),                # 0.526
    ("学生偏好看视频学习", "学生偏好看图解学习", "CONFLICT"),                  # 0.778
    ("学生偏好晚上学习安静环境复习", "学生偏好周末白天去图书馆复习", "CONFLICT"),  # 0.429
    ("学生习惯晚上十点后复习功课", "学生现在改为清晨复习功课", "CONFLICT"),      # 0.480
    # ── gray 区：互补对（LLM 臂误杀即 FP）──
    ("学生下周三有实验考试", "学生下周三前要交实验报告", "KEEP"),              # 0.636
    ("学生用第7版教材", "学生用的是第7版教材的配套习题册", "KEEP"),            # 0.667
    # ── below 区（< floor）：典型反转但两臂设计上均不覆盖，单列 ──
    ("学生希望每题都给出完整推导过程", "学生希望直接给答案省时间", "CONFLICT"),  # 0.370
    # ── below 区：独立对（两臂都应保留）──
    ("学生目标是期末85分", "学生每天只能晚上学习一小时", "KEEP"),               # 0.174
    ("学生用第7版教材", "学生实验室用Multisim 14", "KEEP"),                    # 0.240
]

_OLD_AT = "2026-01-01T00:00:00+00:00"
_NEW_AT = "2026-03-01T00:00:00+00:00"  # 时间差 > conflict_min_days_gap（规则臂门槛）


def _zone(sim: float, floor: float, threshold: float) -> str:
    if sim >= threshold:
        return "rule"
    if sim >= floor:
        return "gray"
    return "below"


async def run_arm(arbitration: bool, with_llm: bool) -> list[dict]:
    """跑一臂：逐对调 _detect_and_clean_conflicts（arbitration 开/关），记录是否剔旧。"""
    from core.memory.mem0_client import _detect_and_clean_conflicts

    cfg = get_settings().mem0
    cfg.conflict_detect_enabled = True
    cfg.conflict_llm_arbitration_enabled = arbitration and with_llm
    settings = get_settings()

    rows = []
    for old_text, new_text, expect in _DATASET:
        sim = SequenceMatcher(None, old_text, new_text).ratio()
        results = [
            {"id": "old", "kind": "persona", "memory": old_text,
             "created_at": _OLD_AT, "metadata": {"kind": "persona", "topic_ids": []}},
            {"id": "new", "kind": "persona", "memory": new_text,
             "created_at": _NEW_AT, "metadata": {"kind": "persona", "topic_ids": []}},
        ]
        try:
            cleaned = await _detect_and_clean_conflicts("eval-user", results, settings)
            dropped = len(cleaned) == 1  # 剔旧=只剩 new
            error = ""
        except Exception as exc:
            dropped, error = False, str(exc)
        rows.append({
            "old": old_text, "new": new_text, "sim": round(sim, 3),
            "zone": _zone(sim, cfg.conflict_llm_sim_floor, cfg.conflict_similarity_threshold),
            "expect": expect, "dropped": dropped, "error": error,
        })
    return rows


def _metrics(rows: list[dict]) -> dict:
    """剔旧=预测正类，expect=CONFLICT=真值正类。ERROR 对不进分母（LLM 没给出判定）。"""
    scored = [r for r in rows if not r["error"]]
    tp = sum(1 for r in scored if r["dropped"] and r["expect"] == "CONFLICT")
    fp = sum(1 for r in scored if r["dropped"] and r["expect"] == "KEEP")
    fn = sum(1 for r in scored if not r["dropped"] and r["expect"] == "CONFLICT")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else None)
    return {
        "n": len(scored), "tp": tp, "fp": fp, "fn": fn,
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "errors": len(rows) - len(scored),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-only", action="store_true",
                        help="只跑规则臂（无 LLM 干跑，自测数据集口径用）")
    args = parser.parse_args()

    async def _run():
        cfg = get_settings().mem0
        report = {
            "model": None if args.rule_only else get_settings().llm.text_model,
            "conflict_similarity_threshold": cfg.conflict_similarity_threshold,
            "conflict_llm_sim_floor": cfg.conflict_llm_sim_floor,
            "rule_arm": _metrics(await run_arm(arbitration=False, with_llm=False)),
            "rule_rows": [],
        }
        if not args.rule_only:
            llm_rows = await run_arm(arbitration=True, with_llm=True)
            report["rule_plus_llm_arm"] = _metrics(llm_rows)
            report["llm_rows"] = llm_rows
        return report

    report = asyncio.run(_run())

    # 决策规则（LLM 臂存在才评）
    if "rule_plus_llm_arm" in report:
        ra, la = report["rule_arm"], report["rule_plus_llm_arm"]
        report["decision"] = (
            "SUGGEST_ENABLE_DEFAULT"
            if la["f1"] is not None and ra["f1"] is not None
            and la["f1"] > ra["f1"] and la["precision"] is not None
            and la["precision"] >= 0.8
            else "KEEP_DISABLED"
        )
        report["decision_rule"] = (
            "LLM 臂 F1 > 规则臂 F1 且 precision >= 0.8 → 建议开默认；否则维持默认关"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "conflict_arbitration_ab.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    slim = {k: v for k, v in report.items() if not k.endswith("_rows")}
    print(json.dumps(slim, ensure_ascii=False, indent=2))
    print(f"\n(逐对明细已写 {out_path})")


if __name__ == "__main__":
    main()
