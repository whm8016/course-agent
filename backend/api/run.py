"""
Unified Capability WebSocket Endpoint
======================================

WS /api/run/{capability_name}

将所有深度能力（chat、deep_solve、deep_research、quiz）统一到单一 WebSocket 入口。
底层通过 TurnRuntimeManager → CourseOrchestrator → CapabilityRegistry 路由。

【协议消息类型（Client → Server）】

  启动 turn（第一条消息或显式 start_turn）：
    {
      "type": "start_turn",          # 可省略，首条消息默认为 start_turn
      "course_id": "algorithm",
      "question":  "请解释快速排序",
      "language":  "zh",
      "history":   [...],            # 可选，OpenAI message 格式
      "tools":     ["rag"],          # 可选
      "metadata":  { ... }           # 可选，capability 扩展参数
    }

  重连订阅（turn 已在运行，客户端重连后恢复接收）：
    {"type": "subscribe_turn", "turn_id": "...", "after_seq": 5}

  取消当前 turn：
    {"type": "cancel_turn"}

  向 ask_user 工具投递用户回复：
    {"type": "submit_user_reply", "content": "是的，继续"}

  心跳：
    {"type": "ping"}  →  Server 回复 {"type": "pong"}

【Server → Client 事件流】
  {"type": "session",     "turn_id": "..."}     首条，携带 turn_id
  {"type": "stage_start", "stage": "planning"}
  {"type": "thinking",    "content": "..."}
  {"type": "token",       "content": "..."}
  {"type": "answer",      "content": "..."}
  {"type": "result",      ...}
  {"type": "done",        "metadata": {...}}
  {"type": "error",       "message": "..."}
  {"type": "wait_for_input", "prompt": "..."}   ask_user 工具暂停
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from api.auth import ws_authenticate
from api.courses import check_course_access
from core.context import UnifiedContext
from core.db.database import AsyncSessionLocal
from core.observability import bind_context, log_flow
from services.session.turn_runtime import get_turn_runtime_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/run", tags=["run"])

_VALID_CAPABILITIES = {"chat", "deep_solve", "deep_research", "quiz", "summarize", "vision"}
_MAX_HISTORY = 20
_MAX_MESSAGE = 2000


@router.websocket("/{capability_name}")
async def websocket_run(websocket: WebSocket, capability_name: str) -> None:
    """统一 WebSocket 入口：按 capability_name 路由到对应能力。"""

    # ---- 1. 鉴权 ----
    user = await ws_authenticate(websocket)
    if user is None:
        return
    log_flow("ws.run.connect", capability=capability_name, user_id=str(user["id"]))

    async def send(msg: dict) -> None:
        try:
            await websocket.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    # ---- 2. 校验 capability_name ----
    if capability_name not in _VALID_CAPABILITIES:
        await send({
            "type": "error",
            "message": f"未知能力：{capability_name}。可用：{sorted(_VALID_CAPABILITIES)}",
        })
        await websocket.close()
        return

    # ---- 3. 接收第一条消息（start_turn payload）----
    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await send({"type": "error", "message": f"无效的消息格式：{exc}"})
        await websocket.close()
        return

    # ---- 4. 处理 subscribe_turn（重连恢复）----
    if payload.get("type") == "subscribe_turn":
        await _handle_subscribe_turn(websocket, payload, send)
        return

    # ---- 5. 解析 start_turn 参数 ----
    course_id: str = (payload.get("course_id") or "").strip()
    question: str = (payload.get("question") or "").strip()[:_MAX_MESSAGE]
    language: str = str(payload.get("language") or "zh")
    history: list[dict] = payload.get("history") or []
    tools_raw = payload.get("tools")
    enabled_tools: list[str] = (
        ["rag"] if tools_raw is None else [str(t) for t in tools_raw]
    )
    metadata: dict = payload.get("metadata") or {}

    if not question:
        await send({"type": "error", "message": "question 不能为空"})
        await websocket.close()
        return

    if len(history) > _MAX_HISTORY:
        history = history[-_MAX_HISTORY:]

    # ---- 6. 课程权限校验 ----
    if course_id:
        try:
            async with AsyncSessionLocal() as db:
                await check_course_access(db, course_id, user)
        except HTTPException as exc:
            await send({"type": "error", "message": exc.detail})
            await websocket.close()
            return

    # ---- 7. 构造 UnifiedContext ----
    metadata.setdefault("question", question)
    context = UnifiedContext(
        course_id=course_id,
        user_id=str(user["id"]),
        user_message=question,
        conversation_history=history,
        mode=capability_name,
        enabled_tools=enabled_tools,
        language=language,
        metadata=metadata,
    )

    # ---- 8. 通过 TurnRuntimeManager 启动 turn ----
    trm = get_turn_runtime_manager()
    # bind_context 在 start_turn 内部执行（含 turn_id 注入）；此处先绑定 WS 特有字段
    bind_context(user_id=str(user["id"]), course_id=course_id, mode=capability_name)
    turn_id = await trm.start_turn(context)
    log_flow("ws.run.start_turn", turn_id=turn_id, capability=capability_name,
             course_id=course_id, user_id=str(user["id"]))

    # 发送 session 事件，携带 turn_id（客户端保存，断线后可用于 subscribe_turn）
    await send({"type": "session", "turn_id": turn_id})

    # ---- 9. 并发：流式发送事件 + 接收命令消息 ----
    async def stream_events() -> None:
        try:
            async for event in trm.subscribe_turn(turn_id):
                await websocket.send_text(json.dumps(event.to_dict(), ensure_ascii=False))
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def recv_commands() -> None:
        while True:
            try:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "cancel_turn":
                await trm.cancel_turn(turn_id)
                break

            elif msg_type == "submit_user_reply":
                accepted = await trm.submit_user_reply(
                    turn_id,
                    text=msg.get("text"),
                    answers=msg.get("answers"),
                )
                log_flow("ws.run.submit_reply", turn_id=turn_id, accepted=accepted)
                if not accepted:
                    await send({"type": "error", "message": f"turn {turn_id} 当前未等待用户回复"})

            elif msg_type == "ping":
                await send({"type": "pong"})

    stream_task = asyncio.create_task(stream_events())
    recv_task = asyncio.create_task(recv_commands())

    done, pending = await asyncio.wait(
        {stream_task, recv_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

    await trm.cancel_turn(turn_id)
    log_flow("ws.run.disconnect", turn_id=turn_id, capability=capability_name)
    try:
        await websocket.close()
    except Exception:
        pass


async def _handle_subscribe_turn(
    websocket: WebSocket,
    payload: dict,
    send,
) -> None:
    """处理断线重连：按 turn_id + after_seq 回放并继续接收。"""
    turn_id = str(payload.get("turn_id", ""))
    after_seq = int(payload.get("after_seq", 0))

    trm = get_turn_runtime_manager()
    try:
        async for event in trm.subscribe_turn(turn_id, after_seq=after_seq):
            await websocket.send_text(json.dumps(event.to_dict(), ensure_ascii=False))
    except KeyError:
        await send({"type": "error", "message": f"turn_id 不存在或已过期：{turn_id}"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/capabilities")
async def list_capabilities() -> dict:
    """列出所有已注册的能力及其元数据（无需鉴权，供前端发现）。"""
    from core.orchestrator import get_orchestrator
    return {"capabilities": get_orchestrator().get_manifests()}
