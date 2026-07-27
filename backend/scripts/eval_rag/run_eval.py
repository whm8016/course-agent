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

# pydantic settings 用双下划线分隔嵌套字段（lightrag.enabled），单下划线不生效
os.environ.setdefault("LIGHTRAG__ENABLED", "true")

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
    # 落盘日志：把 root logger 的全部输出（含 ragas.executor 的 Job[N] 异常）写到文件，
    # 终端关闭后仍可回溯判分异常的真实类型。RAGAS Executor 把判分异常静默转 np.nan，
    # 不落盘就抓不到"真凶"（NaN→0 假性0分的根因）。
    from datetime import datetime as _dt
    _log_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _log_path = config.RESULTS_DIR / f"eval_run_{_log_ts}.log"
    _file_handler = logging.FileHandler(_log_path, encoding="utf-8")
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(_file_handler)
    logger.info("本次评测日志落盘: %s", _log_path)

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
        "--dataset",
        choices=["qa", "synthetic"],
        default="qa",
        help="评测集来源：qa=人工手写 qa_dataset.json（默认）；"
             "synthetic=加载 synthetic_dataset.json（直接用现成题目，不重新合成，省 LLM）",
    )
    parser.add_argument(
        "--modes",
        default="fact,relationship",
        help="评测策略列表（逗号分隔）；实际是生产自适应路由的 strategy："
             "fact=纯向量检索（默认）、relationship=图谱邻域+naive 事实。两条都走生产检索方法，"
             "answer 统一用主对话 LLM 生成（对齐 tool_registry._execute_rag）",
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
        "--production-parity",
        action="store_true",
        help="（deprecated）contexts 现已恒走生产路径，此开关不再有实际效果，保留仅为向后兼容",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="检索 top_k（默认读 config.EVAL_TOP_K=5，对齐生产）",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="消融模式：遍历 retrieval_config.ABLATION_CONFIGS 全组合（dense/bm25/rerank/融合方式），"
             "每个 RetrievalConfig 跑 hybrid retrieve 对比检索召回质量（context_precision/recall，answer 留空省 LLM）。"
             "需 settings.elasticsearch.enabled=true 才有 BM25 路；未启用则 bm25 配置退化为纯 dense。",
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
    elif args.dataset == "synthetic":
        # 直接加载现成合成题（synthetic_dataset.json），不重新合成，省 LLM
        dataset_path = config.EVAL_DIR / "synthetic_dataset.json"
        logger.info("加载合成评测集: %s", dataset_path)
        if not dataset_path.exists():
            logger.error("评测集文件不存在: %s", dataset_path)
            sys.exit(1)
        qa_items = json.loads(dataset_path.read_text("utf-8"))
    else:
        # 人工评测集（qa_dataset.json）
        logger.info("加载人工评测集: %s", config.DATASET_PATH)
        if not config.DATASET_PATH.exists():
            logger.error("评测集文件不存在: %s", config.DATASET_PATH)
            logger.error("提示：用 --generate N 自动合成评测集，或 --dataset synthetic 用合成集")
            sys.exit(1)
        qa_items = json.loads(config.DATASET_PATH.read_text("utf-8"))
    logger.info("评测集大小: %d 条", len(qa_items))

    # 分类统计
    cat_counts: dict[str, int] = {}
    for item in qa_items:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    logger.info("分类分布: %s", cat_counts)

    # ---- Step 2: 查询 ----
    if args.ablation:
        # 消融模式：遍历 ABLATION_CONFIGS，每个 RetrievalConfig 跑 hybrid retrieve
        from core.rag.retrieval_config import ABLATION_CONFIGS
        from scripts.eval_rag.ablation_runner import run_ablation

        modes = [cfg.label() for cfg in ABLATION_CONFIGS.values()]
        # ablation answer 留空（只对比检索召回），强制 context 指标，跳过需 answer 的 faithfulness/answer_relevancy
        metric_names = ["context_precision", "context_recall"]
        logger.info("消融模式：配置=%s", modes)
        logger.info("消融指标（强制）: %s", metric_names)

        all_results = await run_ablation(
            args.course, qa_items, ABLATION_CONFIGS, top_k=args.top_k,
        )
    else:
        # 原模式路径：逐 LightRAG mode（naive/local/global/mix）查询
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
            top_k=args.top_k,
            production_parity=args.production_parity,
        )

    # ---- Step 3-4: RAGAS 指标计算 ----
    # evaluate_mode 单次跑完直接返回 {avg, per_question}：逐条分数来自同一次 evaluate
    # 的结果（每条样本一项），无需逐条重跑，也不再把整体均值复制 N 遍（Bug 3 修复）
    from scripts.eval_rag.ragas_evaluator import evaluate_mode

    avg_scores: dict[str, dict[str, float]] = {}
    per_question_scores: dict[str, list[dict[str, float]]] = {}
    total_tokens_by_mode: dict[str, int] = {}
    invalid_by_mode: dict[str, dict[str, list[int]]] = {}
    for mode in modes:
        logger.info("计算 %s 模式的指标...", mode)
        res = evaluate_mode(qa_items, all_results[mode], metric_names)
        avg_scores[mode] = res["avg"]
        per_question_scores[mode] = res["per_question"]
        total_tokens_by_mode[mode] = res.get("total_tokens", 0)
        invalid_by_mode[mode] = res.get("invalid", {})
        logger.info("%s 模式结果: %s", mode, res["avg"])
        # 判崩排除清单：RAGAS Executor 异常→NaN→记0 的假性0分，已剔出均值，列出供人工复核
        for m, idxs in (invalid_by_mode[mode] or {}).items():
            if idxs:
                qids = [qa_items[i]["id"] for i in idxs if i < len(qa_items)]
                logger.warning(
                    "  [%s] %s 有 %d 题判分异常(NaN)已剔出均值: %s",
                    mode, m, len(idxs), qids,
                )

    # 成本估算（Phase 3.6：按 total_tokens 粗估，input/output 混合平均单价）
    total_tokens = sum(total_tokens_by_mode.values())
    est_cost = total_tokens / 1_000_000 * config.COST_PER_M_TOKENS
    logger.info("本轮评测总 token: %d，估算成本: $%.4f（单价 $%s/M）",
                total_tokens, est_cost, config.COST_PER_M_TOKENS)

    # ---- Step 4.5: 分布 + 延迟 + 历史对比 + 落盘（Phase 4）----
    from datetime import datetime
    from scripts.eval_rag import stats

    # 历史要先读（此时 results/ 仍是上次结果），再落盘当前，避免读到刚写的自己
    last_summary = stats.load_last_summary()
    last_avg = (last_summary or {}).get("avg_scores")

    distributions: dict[str, dict[str, dict[str, float]]] = {}
    latency: dict[str, dict[str, dict[str, float]]] = {}
    for mode in modes:
        distributions[mode] = stats.compute_distribution_by_metric(
            per_question_scores[mode], metric_names
        )
        latency[mode] = stats.compute_latency_distribution(all_results[mode])

    delta = stats.diff_avg_against(avg_scores, last_avg, metric_names)
    if last_avg:
        logger.info("历史对比（vs 上次 %s）:", last_summary.get("timestamp", "?"))
        for mode in modes:
            line = f"  {mode:10s} | " + " | ".join(
                f"{m}: {delta[mode].get(m, 0):+.4f}" for m in metric_names
            )
            logger.info(line)
    else:
        logger.info("无历史结果（首次运行），跳过 delta 对比")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "course": args.course,
        "modes": modes,
        "metrics": metric_names,
        "avg_scores": avg_scores,
        "distributions": distributions,
        "latency": latency,
        "delta": delta,
        "has_baseline": last_avg is not None,
        "total_tokens_by_mode": total_tokens_by_mode,
        "total_tokens": total_tokens,
        "est_cost": est_cost,
        "invalid_by_mode": invalid_by_mode,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.RESULTS_DIR / f"eval_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    logger.info("Summary JSON: %s", summary_path)

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
        course_name=args.course,
        summary=summary,
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

    # ---- Step 6: CI 质量门禁（Phase 5）----
    # 全部达标 exit 0（CI 通过），任一不达标 exit 1（阻断 CI）。日志用 ASCII 避免 GBK 控制台编码问题
    # sys 用顶层第 29 行的 import；这里不再局部 import，否则会让 sys 变成 main() 的局部名，
    # 导致第 117 行（数据集缺失 fallback）的 sys.exit 报 UnboundLocalError
    if args.ablation:
        logger.info("消融模式：跳过质量门禁（配置对比分析，非达标门禁），直接通过")
        sys.exit(0)

    from scripts.eval_rag.quality_gate import check_quality_gate
    passed, failures = check_quality_gate(summary)
    if passed:
        logger.info("[PASS] 质量门禁：全部指标达标")
    else:
        logger.warning("[FAIL] 质量门禁：以下指标不达标：")
        for msg in failures:
            logger.warning("  - %s", msg)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())