"""RAG 评测主入口脚本。

Usage:
    # 自动合成评测集 + 跑核心指标（推荐，针对一门已建索引的课）
    python -m scripts.eval_rag.run_eval --course <course_id> --generate 15

    # 合成时直接扫描一个文档目录（不连库）
    python -m scripts.eval_rag.run_eval --docs-dir ./data/kb/my_course --generate 15

    # 用现成的人工评测集（qa_dataset.json）评测
    python -m scripts.eval_rag.run_eval

    # 只跑零成本指标（快速验证）
    python -m scripts.eval_rag.run_eval --metrics context_precision,context_recall

    # 指定模式
    python -m scripts.eval_rag.run_eval --modes naive,local,mix --course circuit_analysis

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
        "--generate",
        type=int,
        default=None,
        metavar="N",
        help="先用 RAGAS 基于该课知识库自动合成 N 道 QA 再评测（需 --course 有知识库，或 --docs-dir）",
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="合成评测集时直接扫描该目录文档（不连库；与 --course 二选一，--course 优先）",
    )
    parser.add_argument(
        "--modes",
        default="naive,local,global,mix",
        help="评测模式列表（逗号分隔）",
    )
    parser.add_argument(
        "--metrics",
        default="faithfulness,context_precision",
        help="评测指标列表（逗号分隔）；默认只跑 faithfulness(防幻觉)+context_precision 省成本，"
             "全量可用 context_precision,context_recall,faithfulness,answer_relevancy",
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
    if args.generate:
        # 自动合成路线：基于课程知识库出题，无需手写标准答案
        from scripts.eval_rag.dataset_generator import generate_dataset

        logger.info("自动合成评测集: %d 道（course=%s, docs_dir=%s）",
                    args.generate, args.course, args.docs_dir)
        qa_items = await generate_dataset(
            course_id=args.course,
            docs_dir=args.docs_dir,
            n=args.generate,
        )
    else:
        # 人工评测集（qa_dataset.json）
        logger.info("加载人工评测集: %s", config.DATASET_PATH)
        if not config.DATASET_PATH.exists():
            logger.error("评测集文件不存在: %s", config.DATASET_PATH)
            logger.error("提示：用 --generate N 自动合成评测集")
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