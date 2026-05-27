"""Deep Solve WebSocket API (Plan → ReAct → Write).

WebSocket: /api/deep-solve/run

Client -> server:
  {"type": "start", "question": "...", "kb_name": "course_id", "language": "zh", "detailed": false}

Server -> client:
  {"type": "progress", "stage": "plan|solve|write", "status": "...", ...}
  {"type": "result", "final_answer": "...", "metadata": {...}}
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
from config import TEXT_MODEL
from core.db.database import AsyncSessionLocal
from core.solve import MainSolver

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deep-solve", tags=["deep-solve"])

_POLL_INTERVAL = 0.3   # 秒
_JOB_TIMEOUT = 600     # 秒（10 分钟）


def _build_runtime_config(*, language: str) -> dict[str, Any]:
    return {
        "system": {
            "language": language,
            "output_base_dir": "./data/solve/output",
        },
        "llm": {"model": TEXT_MODEL},
        "agents": {},
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
    question: str,
    kb_name: str | None,
    language: str,
    detailed: bool,
    enabled_tools: list[str],
    runtime_config: dict[str, Any],
) -> None:
    """ARQ 不可用时，在当前协程中内联执行（原有逻辑）。"""

    async def send(msg: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    loop = asyncio.get_event_loop()

    def send_progress(stage: str, progress: dict[str, Any]) -> None:
        try:
            if loop.is_running():
                asyncio.ensure_future(
                    send({"type": "progress", "stage": stage, **progress})
                )
        except Exception:
            pass

    def trace_bridge(event: dict[str, Any]) -> None:
        try:
            if loop.is_running():
                asyncio.ensure_future(send({"type": "trace", **event}))
        except Exception:
            pass

    rag_enabled = "rag" in {t.lower() for t in enabled_tools}
    effective_kb = kb_name if rag_enabled else None

    try:
        solver = MainSolver(
            config=runtime_config,
            kb_name=effective_kb or "",
            language=language,
            enabled_tools=enabled_tools,
            disable_planner_retrieve=not (rag_enabled and effective_kb),
        )
        solver._send_progress_update = send_progress
        solver.set_trace_callback(trace_bridge)
    except Exception:
        logger.exception("Deep solve init failed")
        await send({"type": "error", "message": "solver init failed"})
        return

    try:
        result = await solver.solve(question, verbose=True, detailed=detailed)
        await send(
            {
                "type": "result",
                "final_answer": result.get("final_answer", ""),
                "output_dir": result.get("output_dir", ""),
                "output_md": result.get("output_md", ""),
                "metadata": result.get("metadata", {}),
            }
        )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during deep solve")
    except Exception:
        logger.exception("Deep solve failed")
        await send({"type": "error", "message": "deep solve failed"})


@router.websocket("/run")
async def websocket_deep_solve(websocket: WebSocket) -> None:
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

    question = str(payload.get("question", "")).strip()
    if not question:
        await send({"type": "error", "message": "question is required"})
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

    language: str = str(payload.get("language") or "zh")
    detailed = bool(payload.get("detailed", False))
    tools = payload.get("tools")
    enabled_tools = ["rag"] if tools is None else [str(x) for x in tools]

    runtime_config = _build_runtime_config(language=language)
    runtime_config.setdefault("solve", {})
    runtime_config["solve"]["detailed_answer"] = detailed

    # 尝试通过 ARQ 提交任务
    from core.arq_pool import get_arq_pool
    arq_pool = await get_arq_pool()

    try:
        if arq_pool is not None:
            import redis.asyncio as aioredis
            from config import REDIS_URL
            job_id = uuid.uuid4().hex[:16]
            await arq_pool.enqueue_job(
                "run_deep_solve",
                job_id=job_id,
                question=question,
                kb_name=kb_name,
                language=language,
                detailed=detailed,
                enabled_tools=enabled_tools,
                runtime_config=runtime_config,
            )
            async with aioredis.from_url(REDIS_URL, decode_responses=True) as r:
                await _poll_job_events(r, job_id, websocket)
        else:
            await _run_inline(
                websocket,
                question=question,
                kb_name=kb_name,
                language=language,
                detailed=detailed,
                enabled_tools=enabled_tools,
                runtime_config=runtime_config,
            )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Deep solve websocket error")
        await send({"type": "error", "message": "deep solve failed"})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
