"""Deep Solve WebSocket API (Plan → ReAct → Write).

WebSocket: /api/deep-solve/run

Client -> server:
  {"type": "start", "question": "...", "kb_name": "course_id", "language": "zh", "detailed": false}

Server -> client:
  {"type": "progress", "stage": "plan|solve|write", "status": "...", ...}
  {"type": "result", "final_answer": "...", "metadata": {...}}
  {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import TEXT_MODEL
from core.solve import MainSolver

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deep-solve", tags=["deep-solve"])


def _build_runtime_config(*, language: str) -> dict[str, Any]:
    return {
        "system": {
            "language": language,
            "output_base_dir": "./data/solve/output",
        },
        "llm": {"model": TEXT_MODEL},
        "agents": {},
    }


@router.websocket("/run")
async def websocket_deep_solve(websocket: WebSocket) -> None:
    await websocket.accept()

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
    language: str = str(payload.get("language") or "zh")
    detailed = bool(payload.get("detailed", False))
    tools = payload.get("tools")
    enabled_tools = ["rag"] if tools is None else [str(x) for x in tools]
    rag_enabled = "rag" in {t.lower() for t in enabled_tools}
    effective_kb = kb_name if rag_enabled else None

    runtime_config = _build_runtime_config(language=language)
    runtime_config.setdefault("solve", {})
    runtime_config["solve"]["detailed_answer"] = detailed

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
    except Exception as exc:
        logger.exception("Deep solve init failed")
        await send({"type": "error", "message": f"solver init failed: {exc}"})
        await websocket.close()
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
    except Exception as exc:
        logger.exception("Deep solve failed: %s", exc)
        await send({"type": "error", "message": str(exc)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
