"""ARQ worker 入口。

启动方式（Docker / 命令行）：
    python -m arq worker.WorkerSettings

包含四类后台任务：
1. run_indexing         – LightRAG 知识库索引（替代 BackgroundTasks）
2. run_llamaindex_build – LlamaIndex 向量索引（替代 BackgroundTasks）
3. run_deep_research    – Deep Research Pipeline（WS 轮询进度）
4. run_deep_solve       – Deep Solve Pipeline（WS 轮询进度）

进度推送协议（run_deep_research / run_deep_solve）：
  Worker   RPUSH  job:{job_id}:events  json(event)
  WS 端    LRANGE job:{job_id}:events  {offset} -1  轮询读取并转发给客户端
  最终事件 type="result" 或 type="error" 标志任务结束
  列表 TTL：任务完成后保留 1 小时，供客户端追取。
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import os

# 让 import 能找到同目录下所有模块（与 main.py 一致）
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)

_JOB_EVENTS_TTL = 3600  # 进度事件列表在 Redis 中保留 1 小时


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

async def _push_event(redis, job_id: str, event: dict) -> None:
    """向 Redis list 追加一条进度事件（供 WS 端轮询）。"""
    key = f"job:{job_id}:events"
    try:
        await redis.rpush(key, json.dumps(event, ensure_ascii=False))
        await redis.expire(key, _JOB_EVENTS_TTL)
    except Exception:
        logger.debug("_push_event failed job_id=%s", job_id, exc_info=True)


# ---------------------------------------------------------------------------
# 任务 1 & 2：知识库索引（复用 admin.py 中的实现）
# ---------------------------------------------------------------------------

async def run_indexing(
    ctx,
    kb_id: str,
    course_id: str,
    file_paths: list[str],
    resume_from_chunk: int = 0,
) -> None:
    """LightRAG 知识库索引后台任务。"""
    from api.admin import _run_indexing
    await _run_indexing(kb_id, course_id, file_paths, resume_from_chunk)


async def run_llamaindex_build(
    ctx,
    kb_id: str,
    course_id: str,
    file_paths: list[str],
) -> None:
    """LlamaIndex 向量索引后台任务。"""
    from api.llama_rag import _run_llamaindex_build
    await _run_llamaindex_build(kb_id, course_id, file_paths)


# ---------------------------------------------------------------------------
# 任务 3：Deep Research
# ---------------------------------------------------------------------------

async def run_deep_research(
    ctx,
    job_id: str,
    research_id: str,
    topic: str,
    language: str,
    kb_name: str | None,
    runtime_config: dict,
) -> None:
    """Deep Research Pipeline 后台任务，进度通过 Redis list 推送。"""
    redis = ctx["redis"]

    loop = asyncio.get_event_loop()

    def progress_callback(event: dict) -> None:
        try:
            if loop.is_running():
                asyncio.ensure_future(_push_event(redis, job_id, {"type": "progress", **event}))
        except Exception:
            pass

    try:
        from core.research import ResearchPipeline

        pipeline = ResearchPipeline(
            config=runtime_config,
            research_id=research_id,
            kb_name=kb_name,
            progress_callback=progress_callback,
        )
        result = await pipeline.run(topic)
        await _push_event(
            redis,
            job_id,
            {
                "type": "result",
                "research_id": result["research_id"],
                "report": result["report"],
                "final_report_path": result.get("final_report_path", ""),
                "metadata": result.get("metadata", {}),
            },
        )
    except Exception:
        logger.exception("Deep research job failed job_id=%s", job_id)
        await _push_event(redis, job_id, {"type": "error", "message": "deep research failed"})


# ---------------------------------------------------------------------------
# 任务 4：Deep Solve
# ---------------------------------------------------------------------------

async def run_deep_solve(
    ctx,
    job_id: str,
    question: str,
    kb_name: str | None,
    language: str,
    detailed: bool,
    enabled_tools: list[str],
    runtime_config: dict,
) -> None:
    """Deep Solve Pipeline 后台任务，进度通过 Redis list 推送。"""
    redis = ctx["redis"]
    loop = asyncio.get_event_loop()

    def send_progress(stage: str, progress: dict) -> None:
        try:
            if loop.is_running():
                asyncio.ensure_future(
                    _push_event(redis, job_id, {"type": "progress", "stage": stage, **progress})
                )
        except Exception:
            pass

    def trace_bridge(event: dict) -> None:
        try:
            if loop.is_running():
                asyncio.ensure_future(_push_event(redis, job_id, {"type": "trace", **event}))
        except Exception:
            pass

    try:
        rag_enabled = "rag" in {t.lower() for t in enabled_tools}
        effective_kb = kb_name if rag_enabled else None

        from core.solve import MainSolver

        solver = MainSolver(
            config=runtime_config,
            kb_name=effective_kb or "",
            language=language,
            enabled_tools=enabled_tools,
            disable_planner_retrieve=not (rag_enabled and effective_kb),
        )
        solver._send_progress_update = send_progress
        solver.set_trace_callback(trace_bridge)

        result = await solver.solve(question, verbose=True, detailed=detailed)
        await _push_event(
            redis,
            job_id,
            {
                "type": "result",
                "final_answer": result.get("final_answer", ""),
                "output_dir": result.get("output_dir", ""),
                "output_md": result.get("output_md", ""),
                "metadata": result.get("metadata", {}),
            },
        )
    except Exception:
        logger.exception("Deep solve job failed job_id=%s", job_id)
        await _push_event(redis, job_id, {"type": "error", "message": "deep solve failed"})


# ---------------------------------------------------------------------------
# 任务 5 & 6：定时学习总结
# ---------------------------------------------------------------------------

async def daily_summary_job(ctx) -> None:
    """每日学习总结（ARQ cron 触发）。"""
    from core.memory.scheduled_summaries import run_daily_summary
    await run_daily_summary(ctx)


async def weekly_summary_job(ctx) -> None:
    """每周学习总结（ARQ cron 触发）。"""
    from core.memory.scheduled_summaries import run_weekly_summary
    await run_weekly_summary(ctx)


# ---------------------------------------------------------------------------
# WorkerSettings
# ---------------------------------------------------------------------------

class WorkerSettings:
    """arq WorkerSettings：`python -m arq worker.WorkerSettings` 读取此配置。"""

    from config import REDIS_URL as _redis_url
    import urllib.parse as _up

    _parsed = _up.urlparse(_redis_url)

    from arq.connections import RedisSettings

    redis_settings = RedisSettings(
        host=_parsed.hostname or "localhost",
        port=_parsed.port or 6379,
        password=_parsed.password or None,
        database=int((_parsed.path or "/0").lstrip("/") or 0),
    )

    functions = [run_indexing, run_llamaindex_build, run_deep_research, run_deep_solve]
    max_jobs = 10
    job_timeout = 3600   # 单个任务最长 1 小时
    keep_result = 300    # 任务结果保留 5 分钟

    from arq.cron import cron
    cron_jobs = [
        cron(daily_summary_job, hour=22, minute=0),
        cron(weekly_summary_job, weekday=4, hour=22, minute=10),
    ]
