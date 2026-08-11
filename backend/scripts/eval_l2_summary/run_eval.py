"""L2 摘要 v2 评测 CLI：载数据集 -> 逐 case 跑新管线 vs 旧基线 -> 落盘 + 汇总对比。

Usage:
    python -m scripts.eval_l2_summary.run_eval
    python -m scripts.eval_l2_summary.run_eval --limit 5
    python -m scripts.eval_l2_summary.run_eval --category knowledge_update

输出两份文件（results/）：
  l2_detail.jsonl   每 case 一行（含新/基线渲染文本，便于复盘）
  l2_summary.json   汇总：新管线 vs 基线各分类通过率 + 总体

核心看点：
  - knowledge_update 类：新管线通过（单值槽覆盖），基线失败（精确去重保不住改口，
    两条值并存）-- 这正是改造的核心收益（对标论文 arXiv:2606.01435 max(ts) 裁决）。
  - temporal 类 resolved：新管线消除 open_question，基线无此机制 -> 基线失败。
  - multi_session 新类（misconception/intent）：基线 N/A（无法表达），新管线通过。
  - abstention 边界：空/全 resolved/未知 kind/预算，新管线全通过。
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from . import config
from .runner import run_case

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_items(dataset: Path, limit: int | None = None, category: str | None = None) -> list[dict]:
    items: list[dict] = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if category:
        items = [it for it in items if it.get("category") == category]
    return items[:limit] if limit else items


def _summarize(records: list[dict]) -> dict:
    """按分类汇总新管线 vs 基线通过率。"""
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"new_pass": 0, "new_n": 0, "base_pass": 0, "base_n": 0, "base_na": 0})
    for r in records:
        cat = r["category"]
        by_cat[cat]["new_n"] += 1
        if r["new_status"] == "pass":
            by_cat[cat]["new_pass"] += 1
        # 基线：n/a 不计入通过率分母
        if r["baseline_status"] == "n/a":
            by_cat[cat]["base_na"] += 1
        else:
            by_cat[cat]["base_n"] += 1
            if r["baseline_status"] == "pass":
                by_cat[cat]["base_pass"] += 1

    rows = []
    for cat, c in sorted(by_cat.items()):
        rows.append({
            "category": cat,
            "new_pass": c["new_pass"], "new_n": c["new_n"],
            "new_rate": round(c["new_pass"] / c["new_n"], 3) if c["new_n"] else 0.0,
            "base_pass": c["base_pass"], "base_n": c["base_n"], "base_na": c["base_na"],
            "base_rate": round(c["base_pass"] / c["base_n"], 3) if c["base_n"] else 0.0,
        })
    total_new = sum(r["new_status"] == "pass" for r in records)
    total_base = sum(r["baseline_status"] == "pass" for r in records)
    total_base_na = sum(r["baseline_status"] == "n/a" for r in records)
    return {
        "rows": rows,
        "total": {
            "n": len(records),
            "new_pass": total_new,
            "new_rate": round(total_new / len(records), 3) if records else 0.0,
            "base_pass": total_base,
            "base_n": len(records) - total_base_na,
            "base_na": total_base_na,
            "base_rate": round(total_base / (len(records) - total_base_na), 3) if (len(records) - total_base_na) else 0.0,
        },
    }


def _print_table(summary: dict) -> None:
    print("\n========== 新管线 vs 旧基线 通过率 ==========")
    print(f"{'分类':<18} {'新管线':<14} {'旧基线':<18}")
    print("-" * 52)
    for r in summary["rows"]:
        new = f"{r['new_pass']}/{r['new_n']} ({r['new_rate']:.0%})"
        base = f"{r['base_pass']}/{r['base_n']} ({r['base_rate']:.0%})" + (f" +{r['base_na']}N/A" if r["base_na"] else "")
        print(f"{r['category']:<18} {new:<14} {base:<18}")
    t = summary["total"]
    new = f"{t['new_pass']}/{t['n']} ({t['new_rate']:.0%})"
    base = f"{t['base_pass']}/{t['base_n']} ({t['base_rate']:.0%})" + (f" +{t['base_na']}N/A" if t["base_na"] else "")
    print("-" * 52)
    print(f"{'总计':<18} {new:<14} {base:<18}")


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 摘要 v2 评测（新管线 vs 旧基线）")
    parser.add_argument("--dataset", default=str(config.DATASETS_DIR / "l2_cases.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条 case")
    parser.add_argument("--category", default=None,
                        help="只跑某分类：knowledge_update / multi_session / temporal / abstention")
    parser.add_argument("--detail", default=str(config.RESULTS_DIR / "l2_detail.jsonl"))
    parser.add_argument("--summary", default=str(config.RESULTS_DIR / "l2_summary.json"))
    args = parser.parse_args()

    items = _load_items(Path(args.dataset), args.limit, args.category)
    logger.info("载入 %d 条 case（新管线 vs 旧基线，纯函数无 LLM）", len(items))

    records: list[dict] = []
    detail_path = Path(args.detail)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as detail_f:
        for item in items:
            rec = run_case(item)
            records.append(rec)
            detail_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            detail_f.flush()
            new_mark = "✓" if rec["new_status"] == "pass" else "✗"
            base_mark = {"pass": "✓", "fail": "✗", "n/a": "-"}.get(rec["baseline_status"], "?")
            logger.info(
                "[%s] %s new=%s base=%s items=%s tok=%s %s",
                rec["category"], rec["id"], new_mark, base_mark,
                rec["items_count"], rec["token_total"], rec["desc"],
            )
            if rec["new_failures"]:
                logger.warning("  new failures: %s", rec["new_failures"])
            if rec["baseline_failures"] and rec["baseline_status"] == "fail":
                logger.info("  baseline failures: %s", rec["baseline_failures"])

    summary = _summarize(records)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    _print_table(summary)
    print(f"\n明细（含渲染文本）：{detail_path}")
    print(f"汇总：{summary_path}")


if __name__ == "__main__":
    main()
