"""RAG 评测主入口脚本。

Usage:
    # 全量评测（5 模式 x 4 指标）
    python -m scripts.eval_rag.run_eval

    # 只跑零成本指标（快速验证）
    python -m scripts.eval_rag.run_eval --metrics context_precision,context_recall

    # 指定模式和课程
    python -m scripts.eval_rag.run_eval --modes fs,mix --course circuit_analysis

    # 清除缓存重新查询
    python -m scripts.eval_rag.run_eval --no-cache
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

# 加载 .env 环境变量
from dotenv import load_dotenv
load_dotenv()

# 确保 LIGHTRAG_ENABLED=True
os.environ.setdefault("LIGHTRAG_ENABLED", "true")

# 设置 OPENAI_API_KEY（LangChain 内部需要）
from scripts.eval_rag import config  # noqa: E402
if config.LLM_API_KEY and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = config.LLM_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="RAG 评测系统")
    parser.add_argument(
        "--course",
        default=config.COURSE_ID,
        help=f"课程 ID（默认: {config.COURSE_ID})",
    )
    parser.add_argument(
        "--modes",
        default="naive,local,global,mix",
        help="评测模式列表（逗号分隔）",
    )
    parser.add_argument(
        "--metrics",
        default="context_precision,context_recall,faithfulness,answer_relevancy",
        help="评测指标列表（逗号分隔）",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="清除缓存，重新查询所有模式",
    )
    parser.add_argument(
        "--per-question",
        action="store_true",
        help="逐条计算指标（更精确但更慢）",
    )
    args = parser.parse_args()

    # ---- Step 1: 加载评测集 ----
    logger.info("加载评测集: %s", config.DATASET_PATH)
    if not config.DATASET_PATH.exists():
        logger.error("评测集文件不存在: %s", config.DATASET_PATH)
        sys.exit(1)

    qa_items = json.loads(config.DATASET_PATH.read_text("utf-8"))
    logger.info("评测集大小: %d 条", len(qa_items))

    # 分类统计
    cat_counts: dict[str, int] = {}
    for item in qa_items:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    logger.info("分类分布: %s", cat_counts)

    # ---- Step 2: 逐模式查询 ----
    modes = [m.strip() for m in args.modes.split(",")]
    metric_names = [m.strip() for m in args.metrics.split(",")]

    logger.info("评测模式: %s", modes)
    logger.info("评测指标: %s", metric_names)

    from scripts.eval_rag.rag_runner import run_all_modes

    all_results = await run_all_modes(
        args.course,
        qa_items,
        modes,
        no_cache=args.no_cache,
    )

    # ---- Step 3-4: RAGAS 指标计算 ----
    from scripts.eval_rag.ragas_evaluator import evaluate_mode, evaluate_mode_per_question

    # 整体平均分
    avg_scores: dict[str, dict[str, float]] = {}
    for mode in modes:
        logger.info("计算 %s 模式的指标...", mode)
        scores = evaluate_mode(qa_items, all_results[mode], metric_names)
        avg_scores[mode] = scores
        logger.info("%s 模式结果: %s", mode, scores)

    # 逐条指标（可选，用于 CSV）
    per_question_scores: dict[str, list[dict[str, float]]] = {}
    if args.per_question:
        for mode in modes:
            logger.info("逐条计算 %s 模式的指标...", mode)
            pq_scores = evaluate_mode_per_question(
                qa_items, all_results[mode], metric_names
            )
            per_question_scores[mode] = pq_scores
    else:
        # 用整体平均分作为每条的近似值（节省 API 成本）
        for mode in modes:
            per_question_scores[mode] = [
                avg_scores.get(mode, {}) for _ in qa_items
            ]

    # ---- Step 5: 生成报告 ----
    from scripts.eval_rag.report_generator import generate_csv, generate_markdown

    csv_path = generate_csv(
        qa_items,
        all_results,
        modes,
        per_question_scores,
        avg_scores,
        metric_names,
    )

    md_path = generate_markdown(
        qa_items,
        all_results,
        modes,
        avg_scores,
        per_question_scores,
        metric_names,
    )

    logger.info("=" * 60)
    logger.info("评测完成！")
    logger.info("CSV 报告: %s", csv_path)
    logger.info("Markdown 报告: %s", md_path)
    logger.info("=" * 60)

    # 输出总体对比概要
    logger.info("\n总体对比概要:")
    for mode in modes:
        scores = avg_scores.get(mode, {})
        line = f"  {mode:10s} | " + " | ".join(
            f"{m}: {scores.get(m, 0):.4f}" for m in metric_names
        )
        logger.info(line)


if __name__ == "__main__":
    asyncio.run(main())