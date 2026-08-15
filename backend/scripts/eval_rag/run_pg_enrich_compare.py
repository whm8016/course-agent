"""PG 后端 RAGAS 对比：contextual enrichment 开 vs 关。

流程：
  1. 从 src_course_id 读原文件，分别索引到 {src}_pg_base / {src}_pg_enrich
  2. 两路均走 llamaindex_pg hybrid retrieve + 主 LLM 生成 answer
  3. 用 qa_dataset.json 跑 RAGAS，输出对比报告

用法（docker backend 容器内）：
    cd /app && .venv/bin/python -P scripts/eval_rag/run_pg_enrich_compare.py mycourse
    cd /app && .venv/bin/python -P scripts/eval_rag/run_pg_enrich_compare.py mycourse --skip-index
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

load_dotenv(_BACKEND_ROOT / ".env", override=False)

from scripts.eval_rag import config  # noqa: E402
from scripts.eval_rag.ragas_evaluator import evaluate_mode  # noqa: E402
from scripts.eval_rag.report_generator import generate_csv, generate_markdown  # noqa: E402
from scripts.eval_rag import stats  # noqa: E402

if config.LLM_API_KEY and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = config.LLM_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_CONTEXT_SEP = "\n\n---\n\n"
_RAG_SYSTEM = (
    "你是课程助教。只能根据下面提供的参考资料回答学生的课程问题；"
    "如果资料不足以回答，请直接说明“根据现有资料暂无法确认”，不要编造。"
    "引用资料中的原始术语、编号与接法表述时，同一概念前后表述必须一致，不要自相矛盾。"
)


def _split_context_string(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in text.split(_CONTEXT_SEP) if p.strip()]
    return parts or [text.strip()]


async def _load_file_paths(course_id: str) -> list[str]:
    from sqlalchemy import select  # noqa: PLC0415
    from core.db.database import AsyncSessionLocal, KBFile, KnowledgeBase  # noqa: PLC0415

    async with AsyncSessionLocal() as db:
        kb_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.course_id == course_id)
        )
        kb = kb_result.scalar_one_or_none()
        if not kb:
            raise SystemExit(f"course_id={course_id} 未找到知识库")
        files_result = await db.execute(select(KBFile).where(KBFile.kb_id == kb.id))
        return [f.file_path for f in files_result.scalars().all()]


async def _index_course(
    course_id: str,
    file_paths: list[str],
    *,
    contextual_enrichment: bool,
) -> None:
    from core.rag.indexer.llamaindex_pg import LlamaIndexIndexer  # noqa: PLC0415
    from settings import get_settings  # noqa: PLC0415

    settings = get_settings()
    settings.chunking.contextual_enrichment = contextual_enrichment
    label = "enrich" if contextual_enrichment else "base"
    logger.info(
        "索引 %s (contextual_enrichment=%s) files=%s",
        course_id, contextual_enrichment, [Path(p).name for p in file_paths],
    )
    indexer = LlamaIndexIndexer()
    await indexer.delete(course_id)
    result = await indexer.index(course_id, file_paths)
    logger.info(
        "[%s] status=%s chunks=%d error=%s",
        label, result.status, result.chunks_created, result.error,
    )
    if result.status != "success":
        raise SystemExit(f"索引失败 course={course_id}: {result.error}")


async def _run_pg_query(course_id: str, question: str, *, top_k: int) -> dict:
    from core.llm.llm import chat_complete  # noqa: PLC0415
    from core.rag import get_retriever  # noqa: PLC0415
    from settings import get_settings  # noqa: PLC0415

    retriever = get_retriever("llamaindex_pg")
    contexts: list[str] = []
    ctx_str = ""
    answer = ""
    retrieve_ms = 0
    query_ms = 0

    t0 = time.perf_counter()
    try:
        ctx_str = await retriever.retrieve_context(
            course_id=course_id, query=question, top_k=top_k,
        )
        contexts = _split_context_string(ctx_str)
    except Exception as exc:
        logger.error("[%s] retrieve 失败: %s", course_id, exc)
    retrieve_ms = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    try:
        user_msg = f"【参考资料】\n{ctx_str}\n\n【问题】\n{question}"
        answer = await chat_complete(
            _RAG_SYSTEM, [], user_msg,
            model=get_settings().llm.text_model,
            temperature=0.3,
            max_tokens=1024,
        )
    except Exception as exc:
        logger.error("[%s] answer 生成失败: %s", course_id, exc)
    query_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "answer": answer,
        "contexts": contexts,
        "retrieve_ms": retrieve_ms,
        "query_ms": query_ms,
    }


async def _run_all_queries(
    course_id: str,
    qa_items: list[dict],
    mode_label: str,
    *,
    top_k: int,
    no_cache: bool,
) -> list[dict]:
    from scripts.eval_rag.rag_runner import load_cache, save_cache  # noqa: E402

    results: list[dict] = []
    for item in qa_items:
        qid = item["id"]
        question = item["question"]
        if not no_cache:
            cached = load_cache(qid, mode_label, course_id)
            if cached is not None:
                logger.info("[%s] %s → 缓存命中", mode_label, qid)
                results.append(cached)
                continue
        logger.info("[%s] %s → 查询中...", mode_label, qid)
        row = await _run_pg_query(course_id, question, top_k=top_k)
        save_cache(qid, mode_label, course_id, row)
        results.append(row)
        await asyncio.sleep(config.QUERY_DELAY)
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="PG contextual enrichment RAGAS 对比")
    parser.add_argument("src_course", help="源课程 ID（读 KB 原文件）")
    parser.add_argument(
        "--metrics",
        default="faithfulness,context_precision,context_recall",
        help="RAGAS 指标（逗号分隔）",
    )
    parser.add_argument("--skip-index", action="store_true", help="跳过索引，直接用已有 PG 数据")
    parser.add_argument("--no-cache", action="store_true", help="清除查询缓存重跑")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    src = args.src_course
    base_id = f"{src}_pg_base"
    enrich_id = f"{src}_pg_enrich"
    modes = ["pg_base", "pg_enrich"]
    course_by_mode = {"pg_base": base_id, "pg_enrich": enrich_id}
    metric_names = [m.strip() for m in args.metrics.split(",")]
    top_k = args.top_k if args.top_k is not None else config.EVAL_TOP_K

    if not config.DATASET_PATH.exists():
        raise SystemExit(f"评测集不存在: {config.DATASET_PATH}")
    qa_items = json.loads(config.DATASET_PATH.read_text("utf-8"))
    logger.info("评测集 %d 题，指标 %s", len(qa_items), metric_names)

    file_paths = await _load_file_paths(src)
    if not args.skip_index:
        await _index_course(base_id, file_paths, contextual_enrichment=False)
        await _index_course(enrich_id, file_paths, contextual_enrichment=True)

    all_results: dict[str, list[dict]] = {}
    for mode in modes:
        cid = course_by_mode[mode]
        logger.info("=== 查询模式 %s (course=%s) ===", mode, cid)
        all_results[mode] = await _run_all_queries(
            cid, qa_items, mode, top_k=top_k, no_cache=args.no_cache,
        )

    avg_scores: dict[str, dict[str, float]] = {}
    per_question_scores: dict[str, list[dict[str, float]]] = {}
    total_tokens_by_mode: dict[str, int] = {}
    invalid_by_mode: dict[str, dict[str, list[int]]] = {}
    for mode in modes:
        logger.info("RAGAS 评分 %s ...", mode)
        res = evaluate_mode(qa_items, all_results[mode], metric_names)
        avg_scores[mode] = res["avg"]
        per_question_scores[mode] = res["per_question"]
        total_tokens_by_mode[mode] = res.get("total_tokens", 0)
        invalid_by_mode[mode] = res.get("invalid", {})
        logger.info("%s → %s", mode, res["avg"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "src_course": src,
        "modes": modes,
        "course_ids": course_by_mode,
        "backend": "llamaindex_pg",
        "metrics": metric_names,
        "avg_scores": avg_scores,
        "total_tokens_by_mode": total_tokens_by_mode,
        "invalid_by_mode": invalid_by_mode,
        "delta_enrich_minus_base": {
            m: avg_scores.get("pg_enrich", {}).get(m, 0)
            - avg_scores.get("pg_base", {}).get(m, 0)
            for m in metric_names
        },
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.RESULTS_DIR / f"pg_enrich_compare_{ts}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")

    distributions = {
        mode: stats.compute_distribution_by_metric(per_question_scores[mode], metric_names)
        for mode in modes
    }
    latency = {
        mode: stats.compute_latency_distribution(all_results[mode])
        for mode in modes
    }
    summary["distributions"] = distributions
    summary["latency"] = latency
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")

    md_path = generate_markdown(
        qa_items, all_results, modes, avg_scores, per_question_scores,
        metric_names, course_name=f"{src} (PG enrich compare)", summary=summary,
    )
    csv_path = generate_csv(
        qa_items, all_results, modes, per_question_scores, avg_scores, metric_names,
    )

    logger.info("=" * 60)
    logger.info("PG enrichment 对比完成")
    logger.info("Summary: %s", summary_path)
    logger.info("Markdown: %s", md_path)
    logger.info("CSV: %s", csv_path)
    logger.info("--- 均值对比 ---")
    for mode in modes:
        scores = avg_scores.get(mode, {})
        line = f"  {mode:12s} | " + " | ".join(
            f"{m}: {scores.get(m, 0):.4f}" for m in metric_names
        )
        logger.info(line)
    logger.info("--- enrich - base ---")
    for m in metric_names:
        logger.info("  %s: %+.4f", m, summary["delta_enrich_minus_base"][m])


if __name__ == "__main__":
    asyncio.run(main())
