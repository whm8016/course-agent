"""走完整 turn 的上下文预算 A/B 评测 CLI：载数据集 -> 逐 case 逐臂跑 -> 落盘 + 汇总对比。

Usage:
    python -m scripts.eval_turn_budget.run_eval
    python -m scripts.eval_turn_budget.run_eval --limit 2 --arms policy_default
    python -m scripts.eval_turn_budget.run_eval --course my_real_course

输出两份文件（results/）：
  turn_budget_ab_detail.jsonl  每 case 每臂一行（含全量 events 轨迹，便于复盘）
  turn_budget_ab_summary.json  两臂均值对比（核心结论：该不该切 coordinator）

汇总核心看点：
  - 成本：input/output/cache_read tokens、cost_usd -- coordinator 是否更省
  - 压缩：cleared_tool_results(coordinator) vs masked_turns(policy) -- 各自压掉多少
  - 回合前裁剪：dropped_count / carry_forward_added -- coordinator 是否真裁了历史
  - 时延：total_elapsed_ms / first_event_ms -- coordinator 的 plan_turn 是否拖慢 TTFT
  - 质量：rounds / answer_chars -- 压缩是否引发 trajectory elongation（多跑轮）

真机跑前置：见 config.py + 本文件 --help 的 --course 说明（须 LLM key + DB + 已索引课程）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
from pathlib import Path

from . import config
from .runner import run_case

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# 汇总均值字段（error case 不计入均值，但 n/ok 体现）。
# extra_llm_calls 不在此列：本 runner 不走 set_arm，该字段结构性恒 0，无对比价值
# （仍保留在 detail 记录里以对齐计划字段表 / eval_context schema 对照）。
_SUMMARY_KEYS = [
    "rounds", "input_tokens", "output_tokens", "cache_read_tokens", "cost_usd",
    "cleared_tool_results", "masked_turns",
    "dropped_count", "carry_forward_added",
    "total_elapsed_ms", "first_event_ms", "answer_chars",
    "history_before_count", "history_after_count",
]


def _load_items(dataset: Path, limit: int | None = None) -> list[dict]:
    items: list[dict] = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit] if limit else items


def _summarize(by_arm: dict[str, list[dict]]) -> list[dict]:
    """每臂均值汇总（error case 不计入均值，但 n/ok 体现）。"""
    rows: list[dict] = []
    for label, recs in by_arm.items():
        ok = [r for r in recs if not r.get("error")]

        def _mean(key: str) -> float:
            vals = [r.get(key) or 0 for r in ok]
            return round(statistics.mean(vals), 2) if vals else 0.0

        row = {"label": label, "n": len(recs), "ok": len(ok)}
        row.update({f"{k}_mean": _mean(k) for k in _SUMMARY_KEYS})
        rows.append(row)
    return rows


def _print_table(summary: list[dict]) -> None:
    """打印两臂对比表（核心指标）。"""
    if not summary:
        return
    # 列：label + 核心 mean 字段
    cols = ["label", "ok",
            "input_tokens_mean", "output_tokens_mean", "cache_read_tokens_mean", "cost_usd_mean",
            "cleared_tool_results_mean", "masked_turns_mean",
            "dropped_count_mean", "carry_forward_added_mean",
            "rounds_mean", "total_elapsed_ms_mean", "first_event_ms_mean", "answer_chars_mean"]
    widths = {c: max(len(c), max(len(f"{r.get(c, ''):.2f}" if isinstance(r.get(c), float) else str(r.get(c, ""))) for r in summary)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in summary:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append((f"{v:.2f}" if isinstance(v, float) else str(v)).ljust(widths[c]))
        print("  ".join(cells))


async def main() -> None:
    parser = argparse.ArgumentParser(description="走完整 turn 的上下文预算 A/B 评测")
    parser.add_argument("--dataset", default=str(config.DATASETS_DIR / "turn_budget_multiturn.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条 case")
    parser.add_argument("--arms", default="", help="逗号分隔 label 子集，空=全部两臂")
    parser.add_argument("--course", default=None,
                        help="覆盖数据集每条 case 的 course_id（须指向已建索引的真实课程）；"
                             "不传则用数据集自带值")
    parser.add_argument("--detail", default=str(config.RESULTS_DIR / "turn_budget_ab_detail.jsonl"))
    parser.add_argument("--summary", default=str(config.RESULTS_DIR / "turn_budget_ab_summary.json"))
    args = parser.parse_args()

    items = _load_items(Path(args.dataset), args.limit)
    arms = config.ARMS
    if args.arms:
        want = {s.strip() for s in args.arms.split(",") if s.strip()}
        arms = [a for a in arms if a["label"] in want]
    if args.course:
        for it in items:
            it["course_id"] = args.course
    logger.info("载入 %d 条 case，%d 个臂（串行，每 case×每臂一次真 turn）",
                len(items), len(arms))

    # 串行：case × arm。settings 是全局单例，并行改同一单例会乱序；串行 + finally 复原才安全。
    by_arm: dict[str, list[dict]] = {a["label"]: [] for a in arms}
    detail_path = Path(args.detail)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as detail_f:
        for item in items:
            for arm in arms:
                logger.info("=== case=%s arm=%s ===", item.get("id"), arm["label"])
                try:
                    rec = await run_case(item, arm)
                except Exception as e:  # noqa: BLE001
                    # run_case 内部已 try/finally 复原 settings；这里兜底 runner 自身异常
                    logger.exception("[%s/%s] runner 崩溃: %s", item.get("id"), arm["label"], e)
                    rec = {"case_id": item.get("id"), "arm": arm["label"],
                           "error": repr(e), "question": item.get("question")}
                detail_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                detail_f.flush()
                # summary 只读数字字段；events/history_before/tool_results/thinking/answer
                # 等大字段已落盘 detail.jsonl，不必在内存 by_arm 里留到结尾--append 精简副本省内存。
                slim = {k: rec.get(k) for k in (*_SUMMARY_KEYS, "case_id", "error")}
                by_arm[arm["label"]].append(slim)
                if config.QUERY_DELAY:
                    await asyncio.sleep(config.QUERY_DELAY)

    summary = _summarize(by_arm)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"summary": summary, "detail_file": str(detail_path)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== 两臂均值对比 ==========")
    _print_table(summary)
    print(f"\n明细（含全量 events）：{detail_path}")
    print(f"汇总：{summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
