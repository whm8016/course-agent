"""四能力离线质量评测入口（Inspect AI）。

Usage:
    python -m scripts.eval_capabilities.run --capability chat
    python -m scripts.eval_capabilities.run --capability solve

跑某能力 task → 读 EvalLog 分数 → 过 config.QUALITY_GATES 门禁 → 落盘 summary。
门禁不达标 exit 1（与 eval_rag 一致；exit 1 = 门禁不达标，非评测失败）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import config
from .gate import check_gate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# scorer 工厂名 → gate metric key（让 gate 配置用可读名 validity/quality…，scorer 名独立）
_SCORER_TO_METRIC = {
    "model_graded_qa": "accuracy",
    "quiz_validity": "validity",
    "quiz_quality": "quality",
    "solve_trajectory": "trajectory_legal",
    "solve_answer": "answer_correctness",
    "research_race": "race_overall",
    "research_fact": "fact",
}


def _extract_scores(log, capability: str) -> dict[str, float]:
    """从 EvalLog.results.scores 提取 {metric_key: value}（取各 scorer 的主数值）。"""
    scores: dict[str, float] = {}
    results = getattr(log, "results", None)
    if results is None:
        return scores
    raw_scores = getattr(results, "scores", []) or []
    for sc in raw_scores:
        name = sc.get("name") if isinstance(sc, dict) else getattr(sc, "name", "")
        metrics = sc.get("metrics") if isinstance(sc, dict) else getattr(sc, "metrics", {})
        metric_key = _SCORER_TO_METRIC.get(name, name)
        val = None
        if isinstance(metrics, dict):
            for _mk, mv in metrics.items():
                v = getattr(mv, "value", mv)
                if isinstance(v, (int, float)):
                    val = float(v)
                    break
        if val is not None:
            scores[metric_key] = val
    return scores


def main():
    parser = argparse.ArgumentParser(description="四能力离线质量评测（Inspect AI）")
    parser.add_argument(
        "--capability", required=True, choices=["chat", "quiz", "solve", "research"]
    )
    args = parser.parse_args()

    # 延迟 import：inspect eval 较重，且读 dataset 文件，仅在实际跑时加载
    from inspect_ai import eval as inspect_eval

    from .tasks import TASKS

    task_factory = TASKS[args.capability]
    logger.info("开始评测能力：%s", args.capability)

    logs = inspect_eval(task_factory)
    log = logs[0]

    scores = _extract_scores(log, args.capability)
    logger.info("%s 分数：%s", args.capability, {k: round(v, 4) for k, v in scores.items()})

    passed, failures = check_gate(args.capability, scores)
    if passed:
        logger.info("[PASS] %s 质量门禁全部达标", args.capability)
    else:
        logger.warning("[FAIL] %s 质量门禁不达标：%s", args.capability, failures)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "capability": args.capability,
        "scores": scores,
        "passed": passed,
        "failures": failures,
    }
    out = config.RESULTS_DIR / f"eval_capabilities_{args.capability}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary：%s", out)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
