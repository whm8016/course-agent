"""出题：WebSocket /generate、/mimic、/followup（对齐 DeepTutor question 路由）。"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from api.auth import ws_authenticate
from api.courses import check_course_access
from core.db.database import AsyncSessionLocal
from config import QUESTION_LOG_DIR
from core.question.coordinator import AgentCoordinator
from core.question.exam_mimic import mimic_exam_questions
from core.question.followup_agent import FollowupAgent
from core.question.path import get_question_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/question", tags=["question"])

_MAX_PDF_BYTES = 30 * 1024 * 1024
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _output_dir_for_run() -> str:
    base = Path(QUESTION_LOG_DIR) / "runs"
    base.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    d = base / run_id
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _mimic_output_dir() -> Path:
    return get_question_dir() / "mimic_papers"


def _safe_pdf_name(name: str) -> str:
    base = Path(name or "exam.pdf").name
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    if ".." in base or "/" in base or "\\" in base:
        raise ValueError("Invalid pdf_name")
    return base


def _task_id_for_question_gen(kb_name: str, requirement: object) -> str:
    key = f"question_{kb_name}_{json.dumps(requirement, sort_keys=True, ensure_ascii=False)}"
    return "qgen_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


@router.websocket("/generate")
async def websocket_question_generate(websocket: WebSocket):
    user = await ws_authenticate(websocket)
    if user is None:
        return
    log_queue: asyncio.Queue = asyncio.Queue()
    pusher: asyncio.Task | None = None

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

    try:
        async with AsyncSessionLocal() as db:
            await check_course_access(db, str(kb_name), user)
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "content": exc.detail})
        await websocket.close()
        return

    task_id = _task_id_for_question_gen(str(kb_name), requirement)
    try:
        await websocket.send_json({"type": "task_id", "task_id": task_id})
    except (RuntimeError, WebSocketDisconnect):
        return

    out_dir = _output_dir_for_run()
    coordinator = AgentCoordinator(
        kb_name=kb_name,
        output_dir=out_dir,
        language=str(data.get("language", "zh")),
        enable_idea_rag=True,
    )

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
            await websocket.send_json(
                {"type": "error", "content": "requirement 中需含 knowledge_point 或 topic"}
            )
            return

        batch_result = await coordinator.generate_from_topic(
            user_topic=user_topic,
            preference=preference,
            num_questions=count,
            difficulty=difficulty,
            question_type=question_type,
        )

        await log_queue.put(
            {
                "type": "batch_summary",
                "requested": count,
                "completed": batch_result.get("completed", 0),
                "failed": batch_result.get("failed", 0),
            }
        )
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
        try:
            await asyncio.wait_for(log_queue.join(), timeout=15.0)
        except (asyncio.TimeoutError, Exception):
            pass
        if pusher:
            pusher.cancel()
            try:
                await pusher
            except asyncio.CancelledError:
                pass
        try:
            await websocket.close()
        except Exception:
            pass


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
