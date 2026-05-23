"""Deep Research WebSocket API (Step 0 minimal).

WebSocket: /api/deep-research/run

Client -> server:
  {"type": "start", "topic": "...", "kb_name": "course_id", "config": {...optional...}}

Server -> client:
  {"type": "progress", "stage": "planning|researching|reporting", "status": "...", ...}
  {"type": "result", "research_id": "...", "report": "...", "metadata": {...}}
  {"type": "error", "message": "..."}
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.auth import ws_authenticate
from config import TEXT_MODEL
from core.research import ResearchPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deep-research", tags=["deep-research"])

_DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "notes",
    "depth": "quick",
    "sources": ["kb"],
}


def _build_runtime_config(*, language: str, kb_name: str | None) -> dict[str, Any]:
    return {
        "system": {
            "language": language,
            "reports_dir": "./data/research/reports",
        },
        "llm": {"model": TEXT_MODEL},
        "rag": {"kb_name": kb_name} if kb_name else {},
    }


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
    research_id: str = payload.get("research_id") or f"research_{uuid.uuid4().hex[:12]}"
    language: str = str(payload.get("language") or "zh")
    _ = payload.get("config") or _DEFAULT_CONFIG

    runtime_config = _build_runtime_config(language=language, kb_name=kb_name)

    def progress_callback(event: dict[str, Any]) -> None:
        import asyncio

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
    except Exception as exc:
        logger.exception("Deep research pipeline init failed (%s)", research_id)
        await send({"type": "error", "message": f"pipeline init failed: {exc}"})
        await websocket.close()
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
    except Exception as exc:
        logger.exception("Deep research failed for %s: %s", research_id, exc)
        await send({"type": "error", "message": str(exc)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
