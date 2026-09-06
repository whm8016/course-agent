"""CLI 入口：跑全部任务 × 两臂，落 results/<ts>.json + 终端汇总表。

用法（backend 根目录）：
    python -m scripts.eval_mcp_discovery.run_eval
    python -m scripts.eval_mcp_discovery.run_eval --arms deferred_exact --limit 2
真实 LLM 调用（.env 配置的主模型），由用户执行。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from . import config
from .harness import ARMS, ensure_setup, run_case
from .tasks import get_tasks

logger = logging.getLogger("eval_mcp_discovery")


def _positive_int(value: str) -> int:
    v = int(value)
    if v < 1:
        raise argparse.ArgumentTypeError(f"须为正整数，得到 {value}")
    return v


def summarize(recs: list[dict]) -> dict:
    """按臂聚合：平均 input/output tokens、轮次、墙钟、命中率（只统计无 error 的）。

    全 error 臂的 avg 字段为 None（无数据），print_summary 显示 "-"——避免 0 被误读成
    "命中率 0"。
    """
    by_arm: dict[str, list[dict]] = {}
    for r in recs:
        by_arm.setdefault(r["arm"], []).append(r)
    summary: dict[str, dict] = {}
    for arm, rows in by_arm.items():
        ok = [r for r in rows if not r.get("error")]
        n = len(ok)
        if n == 0:
            summary[arm] = {"cases": len(rows), "errors": len(rows)}
            continue
        summary[arm] = {
            "cases": len(rows),
            "errors": len(rows) - n,
            "avg_input_tokens": round(sum(r["input_tokens"] for r in ok) / n),
            "avg_output_tokens": round(sum(r["output_tokens"] for r in ok) / n),
            "avg_rounds": round(sum(r["rounds"] for r in ok) / n, 2),
            "avg_wall_ms": round(sum(r["wall_ms"] for r in ok) / n),
            "avg_recall": round(sum(r["recall"] for r in ok) / n, 4),
            "avg_false_calls": round(sum(len(r["false_calls"]) for r in ok) / n, 2),
            "avg_system_prompt_chars": round(sum(r["system_prompt_chars"] for r in ok) / n),
        }
    return summary


def print_summary(summary: dict) -> None:
    cols = [
        ("arm", "{:<18}"), ("cases", "{:>5}"), ("errors", "{:>6}"),
        ("avg_in_tok", "{:>11}"), ("avg_out_tok", "{:>11}"),
        ("avg_rounds", "{:>10}"), ("avg_wall_s", "{:>10}"),
        ("avg_recall", "{:>10}"), ("avg_false", "{:>9}"), ("avg_prompt_ch", "{:>12}"),
    ]
    header = "".join(fmt.format(name) for name, fmt in cols)
    print("\n" + header)
    print("-" * len(header))
    for arm, s in summary.items():
        row = [
            arm, s["cases"], s["errors"],
            s.get("avg_input_tokens", "-"), s.get("avg_output_tokens", "-"),
            s.get("avg_rounds", "-"),
            round(s["avg_wall_ms"] / 1000, 1) if "avg_wall_ms" in s else "-",
            s.get("avg_recall", "-"), s.get("avg_false_calls", "-"),
            s.get("avg_system_prompt_chars", "-"),
        ]
        print("".join(fmt.format(v) for (name, fmt), v in zip(cols, row)))


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="MCP 主动发现 vs deferred 清单对比 demo")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS), help="要跑的臂")
    parser.add_argument("--tasks", nargs="*", default=None, help="任务 id 子集（默认全部）")
    parser.add_argument("--limit", type=_positive_int, default=None, help="每臂只跑前 N 个任务（smoke）")
    parser.add_argument("--delay", type=float, default=config.QUERY_DELAY, help="case 间隔秒数")
    parser.add_argument("--max-iterations", type=_positive_int, default=config.MAX_ITERATIONS)
    parser.add_argument("--model", default=None, help="覆盖 config.MODEL")
    parser.add_argument("--out", default=None, help="结果 JSON 路径（默认 results/<ts>.json）")
    args = parser.parse_args()

    ensure_setup()
    # 实际模型名（config.MODEL 为 None 时回落生产 text_model），仅用于结果回报
    from settings import get_settings

    effective_model = args.model or config.MODEL or get_settings().llm.text_model
    all_tasks = get_tasks()
    tasks = all_tasks
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t["id"] in wanted]
        missing = wanted - {t["id"] for t in tasks}
        if missing:
            parser.error(f"未知任务 id: {sorted(missing)}（可选 {[t['id'] for t in all_tasks]}）")
    if args.limit:
        tasks = tasks[: args.limit]

    run_tag = time.strftime("%Y%m%d-%H%M%S")
    recs: list[dict] = []
    for arm in args.arms:
        for task in tasks:
            logger.info("=== %s / %s ===", arm, task["id"])
            rec = await run_case(
                task, arm,
                model=args.model,
                max_iterations=args.max_iterations,
                run_tag=run_tag,
            )
            if rec.get("error"):
                logger.error("%s/%s 失败: %s", arm, task["id"], rec["error"])
            else:
                logger.info(
                    "%s/%s rounds=%s in=%s out=%s recall=%.2f false=%s tools=%s",
                    arm, task["id"], rec["rounds"], rec["input_tokens"],
                    rec["output_tokens"], rec["recall"], rec["false_calls"],
                    rec["domain_tools_used"],
                )
            recs.append(rec)
            if args.delay:
                await asyncio.sleep(args.delay)

    summary = summarize(recs)
    out_path = Path(args.out) if args.out else config.RESULTS_DIR / f"{run_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"run_tag": run_tag, "model": effective_model, "records": recs, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(summary)
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
