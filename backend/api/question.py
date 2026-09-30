"""出题：WebSocket /followup（单题追问）。

按知识点出题的主链路是统一能力入口 WS /api/run/quiz
（QuizCapability → QuizPipeline，见 core/capabilities/quiz.py）；本路由只保留
仍在使用的单题追问端点 /followup。旧的 /generate（AgentCoordinator 路径）与
仿卷 /mimic 已移除。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.auth import ws_authenticate
from core.question.followup_agent import FollowupAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/question", tags=["question"])


@router.websocket("/followup")
async def websocket_question_followup(websocket: WebSocket):
    # WS 不走 BaseHTTPMiddleware：手动绑 request_id（错误信封 ref 的来源，与 run.py 同款）
    from core.observability.context import bind_ws_request_id

    bind_ws_request_id(websocket)
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
            # ws_authenticate 返回 dict（键 "id"，与 run.py 同口径）——旧 getattr
            # (user, "user_id", "") 作用在 dict 上恒为 ""，埋点归因失效
            user_id=str(user["id"]),
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
        from core.observability.errors import normalize_error

        _env = normalize_error(e)
        try:
            # content 字段名保留（前端 msg.message ?? msg.content 双读）；信封加 code/ref
            await websocket.send_json({"type": "error", "content": _env["message"],
                                       "code": _env["code"], "ref": _env["ref"]})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
