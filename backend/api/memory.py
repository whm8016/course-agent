"""用户画像（Learner Memory）REST，对齐 DeepTutor 的 /memory 路由。

- GET    /api/memory                 -> 当前用户的 summary + profile 快照
- PUT    /api/memory                 -> 手动编辑某一份文档
- POST   /api/memory/refresh         -> 用最近若干条会话消息触发 LLM 重写
- POST   /api/memory/clear           -> 清空指定/全部文档
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from core.db.database import get_db
from core.memory.learner_profile import (
    MEMORY_FILES,
    clear_memory,
    read_snapshot,
    refresh_from_source,
    write_file,
)
from core.memory.graph_memory import (
    delete_graph_node,
    load_graphs,
)
from core.memory.memory import get_messages, get_session

router = APIRouter(prefix="/memory")

MemoryFile = Literal["summary", "profile"]
_VALID_FILES = set(MEMORY_FILES)


class FileUpdateRequest(BaseModel):
    file: MemoryFile
    content: str = ""


class MemoryRefreshRequest(BaseModel):
    session_id: str | None = None
    max_messages: int = 10


class MemoryClearRequest(BaseModel):
    file: MemoryFile | None = None


@router.get("")
async def get_memory(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await read_snapshot(db, user["id"])


@router.put("")
async def update_memory(
    payload: FileUpdateRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if payload.file not in _VALID_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid file: {payload.file}")
    snap = await write_file(db, user["id"], payload.file, payload.content)
    return {**snap, "saved": True}


@router.post("/refresh")
async def refresh_memory(
    payload: MemoryRefreshRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session_id = (payload.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    session = await get_session(db, session_id)
    if session is None or session.get("user_id") != user["id"]:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await get_messages(db, session_id)
    relevant = [
        m for m in messages
        if m.get("role") in {"user", "assistant"} and str(m.get("content") or "").strip()
    ][-max(1, payload.max_messages):]

    if not relevant:
        raise HTTPException(status_code=400, detail="No messages to refresh from")

    transcript = "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {str(m['content']).strip()}"
        for m in relevant
    )
    source = (
        f"[Session] {session_id}\n"
        f"[Course] {session.get('course_id') or '(unknown)'}\n"
        f"[Capability] {session.get('mode') or 'chat'}\n\n"
        f"[Recent Transcript]\n{transcript}"
    )

    result = await refresh_from_source(db, user["id"], source)
    return result


@router.post("/clear")
async def clear_memory_endpoint(
    payload: MemoryClearRequest | None = None,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    target = payload.file if payload else None
    if target is not None and target not in _VALID_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid file: {target}")
    snap = await clear_memory(db, user["id"], target)
    return {**snap, "cleared": True}


# ---------------------------------------------------------------------------
# 知识图谱 / 错题图谱
# ---------------------------------------------------------------------------

@router.get("/graph")
async def get_graphs(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    kg, eg = await load_graphs(db, user["id"])
    return {"knowledge_graph": kg, "error_graph": eg}


class GraphDeleteRequest(BaseModel):
    node_id: str


@router.post("/graph/delete")
async def delete_node(
    payload: GraphDeleteRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not payload.node_id:
        raise HTTPException(status_code=400, detail="node_id is required")
    result = await delete_graph_node(db, user["id"], payload.node_id)
    return result


# ---------------------------------------------------------------------------
# 学生仪表盘
# ---------------------------------------------------------------------------

@router.get("/dashboard")
async def get_dashboard(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    snap = await read_snapshot(db, user["id"])
    kg, eg = await load_graphs(db, user["id"])

    high_risk = sorted(
        [n for n in (kg.get("nodes") or []) if n.get("status") == "active"],
        key=lambda n: -(n.get("risk") or 0),
    )[:5]

    frequent_errors = sorted(
        [n for n in (eg.get("nodes") or []) if (n.get("error_count") or 0) > 1],
        key=lambda n: -(n.get("error_count") or 0),
    )[:5]

    return {
        "summary": snap.get("summary", ""),
        "profile": snap.get("profile", ""),
        "high_risk_points": high_risk,
        "frequent_errors": frequent_errors,
        "knowledge_node_count": len(kg.get("nodes") or []),
        "error_node_count": len(eg.get("nodes") or []),
    }
