"""上下文预算策略消融 CLI：载 chat_longhorizon.jsonl → run_ablation → 落盘 JSON + 打印汇总。

Usage:
    python -m scripts.eval_context.run_eval
    python -m scripts.eval_context.run_eval --limit 5 --configs masking_M3,hybrid

汇总每臂均值：rounds / input_tokens / output_tokens / extra_llm_calls（核心看点：
masking 相对 raw 成本是否减半、summary_only 是否 trajectory elongation、hybrid 是否最优）。
真机跑需 LLM key + 可用课程（dataset.metadata.course_id 指向已索引课程）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
from pathlib import Path

from . import config
from .ablation_runner import run_ablation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_items(dataset: Path, limit: int | None = None) -> list[dict]:
    items: list[dict] = []
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit] if limit else items


def _summarize(all_results: dict[str, list[dict]]) -> list[dict]:
    """每臂均值汇总（error 的 case 不计入均值，但计入 n/ok）。"""
    rows: list[dict] = []
    for label, recs in all_results.items():
        ok = [r for r in recs if not r.get("error")]

        def _mean(key: str) -> float:
            vals = [r.get(key, 0) for r in ok]
            return round(statistics.mean(vals), 2) if vals else 0.0

        rows.append({
            "label": label,
            "n": len(recs),
            "ok": len(ok),
            "rounds_mean": _mean("rounds"),
            "input_tokens_mean": _mean("input_tokens"),
            "output_tokens_mean": _mean("output_tokens"),
            "extra_llm_calls_mean": _mean("extra_llm_calls"),
            "answer_chars_mean": _mean("answer_chars"),
        })
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description="上下文预算策略消融（arXiv:2508.21433 对照）")
    parser.add_argument("--dataset", default=str(config.DATASETS_DIR / "chat_longhorizon.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条 case")
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "context_ablation.json"))
    parser.add_argument("--configs", default="", help="逗号分隔 label 子集，空=全部")
    args = parser.parse_args()

    items = _load_items(Path(args.dataset), args.limit)
    configs = config.CONTEXT_POLICY_CONFIGS
    if args.configs:
        want = {s.strip() for s in args.configs.split(",") if s.strip()}
        configs = [c for c in configs if c["label"] in want]
    logger.info("载入 %d 条 case，%d 个配置", len(items), len(configs))

    all_results = await run_ablation(items, configs)
    summary = _summarize(all_results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "detail": all_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
