"""出题：WebSocket /mimic、/followup（对齐 DeepTutor question 路由）。

按知识点出题的主链路已迁移到统一能力入口 WS /api/run/quiz
（QuizCapability → QuizPipeline，见 core/capabilities/quiz.py），本路由仅保留
仿卷（/mimic，PDF / 已解析目录出题）与单题追问（/followup）两个仍在使用的端点。
旧的 /generate（AgentCoordinator.generate_from_topic 路径）已移除。
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from api.auth import ws_authenticate
from api.courses import check_course_access
from core.db.database import AsyncSessionLocal
from core.question.exam_mimic import mimic_exam_questions
from core.question.followup_agent import FollowupAgent
from core.question.path import get_question_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/question", tags=["question"])

_MAX_PDF_BYTES = 30 * 1024 * 1024
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _mimic_output_dir() -> Path:
    return get_question_dir() / "mimic_papers"


def _safe_pdf_name(name: str) -> str:
    base = Path(name or "exam.pdf").name
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    if ".." in base or "/" in base or "\\" in base:
        raise ValueError("Invalid pdf_name")
    return base


@router.websocket("/followup")
async def websocket_question_followup(websocket: WebSocket):
    user = await ws_authenticate(websocket)
    if user is None:
        return
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    user_message = str(data.get("user_message", "")).strip()
    qctx = data.get("question_context")
    hist = str(data.get("history_context", ""))
    lang = str(data.get("language", "zh"))

    if not user_message:
        await websocket.send_json({"type": "error", "content": "user_message 必填"})
        await websocket.close()
        return

    ctx: dict = qctx if isinstance(qctx, dict) else {}
    try:
        await websocket.send_json({"type": "status", "content": "started"})
        agent = FollowupAgent(language=lang)
        full = ""
        async for chunk in agent.stream_process(
            user_message=user_message,
            question_context=ctx,
            history_context=hist,
        ):
            full += chunk
            try:
                await websocket.send_json({"type": "token", "content": chunk})
            except (RuntimeError, WebSocketDisconnect):
                return
        await websocket.send_json({"type": "answer", "content": full})
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        logger.debug("followup client disconnected")
    except Exception as e:
        logger.exception("question followup failed")
        try:
            await websocket.send_json({"type": "error", "content": str(e)[:800]})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/mimic")
async def websocket_mimic_generate(websocket: WebSocket):
    user = await ws_authenticate(websocket)
    if user is None:
        return
    log_queue: asyncio.Queue | None = asyncio.Queue()
    pusher_task: asyncio.Task | None = None
    original_stdout = sys.stdout

    try:
        data = await websocket.receive_json()
        mode = data.get("mode", "parsed")
        kb_name = data.get("kb_name", data.get("course_id", ""))
        max_questions = data.get("max_questions")
        language = str(data.get("language", "zh"))

        if not kb_name:
            await websocket.send_json({"type": "error", "content": "kb_name 或 course_id 必填"})
            return

        try:
            async with AsyncSessionLocal() as db:
                await check_course_access(db, str(kb_name), user)
        except HTTPException as exc:
            await websocket.send_json({"type": "error", "content": exc.detail})
            await websocket.close()
            return

        loop = asyncio.get_running_loop()

        def emit_process_log(entry: dict) -> None:
            if log_queue is not None:
                loop.call_soon_threadsafe(log_queue.put_nowait, entry)

        async def log_pusher():
            assert log_queue is not None
            while True:
                entry = await log_queue.get()
                try:
                    await websocket.send_json(entry)
                except Exception:
                    break
                log_queue.task_done()

        pusher_task = asyncio.create_task(log_pusher())

        class StdoutInterceptor:
            def __init__(self, queue: asyncio.Queue, original):
                self.queue = queue
                self.original_stdout = original
                self._closed = False

            def write(self, message):
                if self._closed:
                    return
                try:
                    self.original_stdout.write(message)
                except Exception:
                    pass
                clean = ANSI_ESCAPE_PATTERN.sub("", message).strip()
                if clean:
                    try:
                        emit_process_log(
                            {
                                "type": "process_log",
                                "level": "INFO",
                                "message": clean,
                                "logger": "question.stdout",
                                "timestamp": datetime.now().timestamp(),
                            }
                        )
                    except (asyncio.QueueFull, RuntimeError):
                        pass

            def flush(self):
                if not self._closed:
                    try:
                        self.original_stdout.flush()
                    except Exception:
                        pass

            def close(self):
                self._closed = True

        interceptor = StdoutInterceptor(log_queue, original_stdout)
        sys.stdout = interceptor

        try:
            await websocket.send_json(
                {"type": "status", "stage": "init", "content": "Initializing mimic…"}
            )

            pdf_path = None
            paper_dir = None

            if mode == "upload":
                pdf_data = data.get("pdf_data")
                pdf_name = data.get("pdf_name", "exam.pdf")
                if not pdf_data:
                    await websocket.send_json(
                        {"type": "error", "content": "PDF data is required for upload mode"}
                    )
                    return
                try:
                    pdf_bytes = base64.b64decode(pdf_data)
                except Exception as e:
                    await websocket.send_json({"type": "error", "content": f"Invalid base64 PDF: {e}"})
                    return
                if len(pdf_bytes) > _MAX_PDF_BYTES:
                    await websocket.send_json({"type": "error", "content": "PDF too large (max 30MB)"})
                    return
                try:
                    safe_name = _safe_pdf_name(str(pdf_name))
                except ValueError as e:
                    await websocket.send_json({"type": "error", "content": str(e)})
                    return

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_stem = Path(safe_name).stem
                batch_dir = _mimic_output_dir() / f"mimic_{timestamp}_{pdf_stem}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                pdf_path_obj = batch_dir / safe_name
                await websocket.send_json(
                    {"type": "status", "stage": "upload", "content": f"Saving PDF: {safe_name}"}
                )
                with open(pdf_path_obj, "wb") as f:
                    f.write(pdf_bytes)
                pdf_path = str(pdf_path_obj)
                output_dir = str(batch_dir)

            elif mode == "parsed":
                paper_path = data.get("paper_path")
                if not paper_path:
                    await websocket.send_json(
                        {"type": "error", "content": "paper_path is required for parsed mode"}
                    )
                    return
                paper_dir = str(paper_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_dir = _mimic_output_dir() / f"mimic_{timestamp}_{Path(paper_path).name}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                output_dir = str(batch_dir)
            else:
                await websocket.send_json({"type": "error", "content": f"Unknown mode: {mode}"})
                return

            async def ws_callback(payload: dict):
                try:
                    await websocket.send_json(payload)
                except Exception as e:
                    logger.debug("mimic ws send failed: %s", e)

            await websocket.send_json(
                {
                    "type": "status",
                    "stage": "processing",
                    "content": "Executing question mimic workflow…",
                }
            )

            result = await mimic_exam_questions(
                pdf_path=pdf_path,
                paper_dir=paper_dir,
                kb_name=kb_name,
                output_dir=output_dir,
                max_questions=max_questions,
                ws_callback=ws_callback,
                language=language,
            )

            if result.get("success"):
                await websocket.send_json({"type": "complete"})
            else:
                err = result.get("error") or "Mimic generation finished with failures"
                await websocket.send_json({"type": "error", "content": str(err)[:800]})
        finally:
            interceptor.close()
            sys.stdout = original_stdout

    except WebSocketDisconnect:
        logger.debug("Client disconnected during mimic generation")
    except Exception as e:
        logger.exception("Mimic generation error")
        try:
            await websocket.send_json({"type": "error", "content": str(e)[:800]})
        except Exception:
            pass
    finally:
        sys.stdout = original_stdout
        if pusher_task:
            pusher_task.cancel()
            try:
                await pusher_task
            except asyncio.CancelledError:
                pass
        if log_queue is not None:
            try:
                while not log_queue.empty():
                    log_queue.get_nowait()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
