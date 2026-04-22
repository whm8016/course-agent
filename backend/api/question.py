"""出题：WebSocket /generate，对齐 DeepTutor question 路由思路。"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import QUESTION_LOG_DIR
from core.question.coordinator import AgentCoordinator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/question", tags=["question"])


def _output_dir_for_run() -> str:
    base = Path(QUESTION_LOG_DIR) / "runs"
    base.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    d = base / run_id
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@router.websocket("/generate")
async def websocket_question_generate(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    requirement = data.get("requirement")
    kb_name = data.get("kb_name") or data.get("course_id")
    count = int(data.get("count", 1))

    if not requirement:
        await websocket.send_json({"type": "error", "content": "requirement 必填"})
        return
    if not kb_name:
        await websocket.send_json({"type": "error", "content": "kb_name 或 course_id 必填"})
        return

    out_dir = _output_dir_for_run()

    # 与 DeepTutor 一致：把 coordinator 的 KB 名当作知识库 ID（你项目里即 course_id）
    coordinator = AgentCoordinator(
        kb_name=kb_name,
        output_dir=out_dir,
        language=str(data.get("language", "zh")),
        enable_idea_rag=True,
    )

    log_queue: asyncio.Queue = asyncio.Queue()

    async def ws_callback(entry: dict):
        await log_queue.put(entry)

    coordinator.set_ws_callback(ws_callback)

    async def log_pusher():
        while True:
            entry = await log_queue.get()
            try:
                await websocket.send_json(entry)
            except Exception:
                break
            log_queue.task_done()

    pusher = asyncio.create_task(log_pusher())

    try:
        await websocket.send_json({"type": "status", "content": "started", "output_dir": out_dir})

        req = requirement if isinstance(requirement, dict) else {"knowledge_point": str(requirement)}
        user_topic = str(req.get("knowledge_point", "") or req.get("topic", ""))
        preference = str(req.get("preference", ""))
        difficulty = str(req.get("difficulty", "") or "")
        question_type = str(req.get("question_type", "") or "")

        if not user_topic:
            await websocket.send_json({"type": "error", "content": "requirement 中需含 knowledge_point 或 topic"})
            return

        batch_result = await coordinator.generate_from_topic(
            user_topic=user_topic,
            preference=preference,
            num_questions=count,
            difficulty=difficulty,
            question_type=question_type,
        )

        # 必须走队列，确保在 result 消息全部发出之后再发 complete
        await log_queue.put({"type": "complete", "summary": batch_result})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during question generation")
    except Exception as e:
        logger.exception("question generate failed")
        try:
            await websocket.send_json({"type": "error", "content": str(e)[:800]})
        except Exception:
            pass
    finally:
        # 等队列排空，最多等 15 秒，防止 result 消息被丢弃
        try:
            await asyncio.wait_for(log_queue.join(), timeout=15.0)
        except (asyncio.TimeoutError, Exception):
            pass
        pusher.cancel()
        try:
            await websocket.close()
        except Exception:
            pass