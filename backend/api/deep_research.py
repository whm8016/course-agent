"""Deep Research WebSocket API (Step 0 minimal).

WebSocket: /api/deep-research/run

Client -> server:
  {"type": "start", "topic": "...", "kb_name": "course_id", "config": {...optional...}}

Server -> client:
  {"type": "progress", "stage": "planning|researching|reporting", "status": "...", ...}
  {"type": "result", "research_id": "...", "report": "...", "metadata": {...}}
  {"type": "error", "message": "..."}

当 ARQ worker 可用时，重计算在独立 worker 进程中执行，WS 通过 Redis list 轮询进度。
ARQ 不可用时（测试 / 开发环境）退回原地内联执行。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from api.auth import ws_authenticate
from api.courses import check_course_access
from core.db.database import AsyncSessionLocal
from config import TEXT_MODEL
from core.research import ResearchPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deep-research", tags=["deep-research"])

_DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "notes",
    "depth": "quick",
    "sources": ["kb"],
}

_POLL_INTERVAL = 0.3   # 秒
_JOB_TIMEOUT = 600     # 秒（10 分钟）


def _build_runtime_config(*, language: str, kb_name: str | None) -> dict[str, Any]:
    return {
        "system": {
            "language": language,
            "reports_dir": "./data/research/reports",
        },
        "llm": {"model": TEXT_MODEL},
        "rag": {"kb_name": kb_name} if kb_name else {},
    }


async def _poll_job_events(redis, job_id: str, websocket: WebSocket) -> None:
    """轮询 Redis list，将进度事件转发给 WebSocket 客户端。"""
    key = f"job:{job_id}:events"
    offset = 0
    deadline = asyncio.get_event_loop().time() + _JOB_TIMEOUT

    while asyncio.get_event_loop().time() < deadline:
        raw_events: list[str] = await redis.lrange(key, offset, -1)
        for raw in raw_events:
            event = json.loads(raw)
            try:
                await websocket.send_text(raw)
            except (WebSocketDisconnect, RuntimeError):
                return  # 客户端断开，任务继续在 worker 中运行
            if event.get("type") in ("result", "error"):
                return
        offset += len(raw_events)
        if not raw_events:
            await asyncio.sleep(_POLL_INTERVAL)

    # 超时
    try:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "任务超时（10 分钟）"}, ensure_ascii=False)
        )
    except Exception:
        pass


async def _run_inline(
    websocket: WebSocket,
    *,
    research_id: str,
    topic: str,
    language: str,
    kb_name: str | None,
    runtime_config: dict[str, Any],
) -> None:
    """ARQ 不可用时，在当前协程中内联执行（原有逻辑）。"""

    async def send(msg: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    def progress_callback(event: dict[str, Any]) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(send({"type": "progress", **event}))
        except Exception:
            pass

    try:
        pipeline = ResearchPipeline(
            config=runtime_config,
            research_id=research_id,
            kb_name=kb_name,
            progress_callback=progress_callback,
        )
    except Exception:
        logger.exception("Deep research pipeline init failed (%s)", research_id)
        await send({"type": "error", "message": "pipeline init failed"})
        return

    try:
        result = await pipeline.run(topic)
        await send(
            {
                "type": "result",
                "research_id": result["research_id"],
                "report": result["report"],
                "final_report_path": result.get("final_report_path", ""),
                "metadata": result.get("metadata", {}),
            }
        )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during research %s", research_id)
    except Exception:
        logger.exception("Deep research failed for %s", research_id)
        await send({"type": "error", "message": "deep research failed"})


@router.websocket("/run")
async def websocket_deep_research(websocket: WebSocket) -> None:
    user = await ws_authenticate(websocket)
    if user is None:
        return

    async def send(msg: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await send({"type": "error", "message": f"Invalid initial message: {exc}"})
        await websocket.close()
        return

    if payload.get("type") != "start":
        await send({"type": "error", "message": "Expected message type 'start'"})
        await websocket.close()
        return

    topic = str(payload.get("topic", "")).strip()
    if not topic:
        await send({"type": "error", "message": "topic is required"})
        await websocket.close()
        return

    kb_name: str | None = (payload.get("kb_name") or "").strip() or None
    if kb_name:
        try:
            async with AsyncSessionLocal() as db:
                await check_course_access(db, kb_name, user)
        except HTTPException as exc:
            await send({"type": "error", "message": exc.detail})
            await websocket.close()
            return

    research_id: str = payload.get("research_id") or f"research_{uuid.uuid4().hex[:12]}"
    language: str = str(payload.get("language") or "zh")
    _ = payload.get("config") or _DEFAULT_CONFIG

    runtime_config = _build_runtime_config(language=language, kb_name=kb_name)

    # 尝试通过 ARQ 提交任务
    from core.arq_pool import get_arq_pool
    arq_pool = await get_arq_pool()

    try:
        if arq_pool is not None:
            import redis.asyncio as aioredis
            from config import REDIS_URL
            job_id = uuid.uuid4().hex[:16]
            await arq_pool.enqueue_job(
                "run_deep_research",
                job_id=job_id,
                research_id=research_id,
                topic=topic,
                language=language,
                kb_name=kb_name,
                runtime_config=runtime_config,
            )
            # 等待 worker 推送进度
            async with aioredis.from_url(REDIS_URL, decode_responses=True) as r:
                await _poll_job_events(r, job_id, websocket)
        else:
            await _run_inline(
                websocket,
                research_id=research_id,
                topic=topic,
                language=language,
                kb_name=kb_name,
                runtime_config=runtime_config,
            )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Deep research websocket error")
        await send({"type": "error", "message": "deep research failed"})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
