"""用户记忆 REST（Mem0 库 + 学习图谱）。

记忆 CRUD/检索直接转发给 mem0ai（AsyncMemory），mem0 自管 memories 表；
学习图谱（知识点 / 错题）由 graph_memory 独立承担，差异化能力。

- GET    /memory                 -> 列出当前用户全部记忆
- POST   /memory                 -> 新增（mem0 自动提取事实 + 去重）
- PUT    /memory/{memory_id}     -> 编辑一条
- DELETE /memory/{memory_id}     -> 删除一条
- GET    /memory/search?q=       -> 语义检索
- GET    /memory/overview        -> 计数 + 图谱计数
- GET    /memory/graph           -> 知识/错题图谱
- POST   /memory/graph/delete    -> 删除图谱节点
- GET    /memory/dashboard       -> 仪表盘
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from core.db.database import get_db
from core.memory.mem0_client import get_memory
from core.memory.graph_memory import delete_graph_node, load_graphs

router = APIRouter(prefix="/memory")
logger = logging.getLogger(__name__)

Section = Literal["profile", "preference", "learning", "general"]


class MemoryCreate(BaseModel):
    content: str
    section: Section = "general"


class MemoryUpdate(BaseModel):
    content: str


class GraphDeleteRequest(BaseModel):
    node_id: str


def _results(resp) -> list[dict]:
    """mem0 的 get_all/search 返回可能是 {"results":[...]} 或 [...]，统一成 list。"""
    if isinstance(resp, dict):
        return resp.get("results", []) or []
    return resp or []


@router.get("")
async def get_memories(user: dict = Depends(get_current_user)):
    user_id = user["id"]
    logger.info("[mem0-api] GET /memory user_id=%s", user_id)
    m = get_memory()
    items = _results(await m.get_all(filters={"user_id": user_id}, top_k=200))
    logger.info("[mem0-api] GET /memory OK user_id=%s count=%d", user_id, len(items))
    return {"memories": items}


@router.post("")
async def create_memory(payload: MemoryCreate, user: dict = Depends(get_current_user)):
    user_id = user["id"]
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="内容不能为空")
    logger.info("[mem0-api] POST /memory user_id=%s content_len=%d", user_id, len(payload.content))
    m = get_memory()
    result = await m.add(payload.content, user_id=user_id)
    if result:
        items = result if isinstance(result, list) else result.get("results", [])
        logger.info("[mem0-api] POST /memory OK user_id=%s stored_count=%d", user_id, len(items) if items else 0)
    else:
        logger.info("[mem0-api] POST /memory OK user_id=%s stored_count=0", user_id)
    return {"saved": True}


@router.put("/{memory_id}")
async def edit_memory(memory_id: str, payload: MemoryUpdate, user: dict = Depends(get_current_user)):
    user_id = user["id"]
    logger.info("[mem0-api] PUT /memory/%s user_id=%s content_len=%d", memory_id, user_id, len(payload.content))
    m = get_memory()
    try:
        await m.update(memory_id, payload.content)
        logger.info("[mem0-api] PUT /memory/%s OK user_id=%s", memory_id, user_id)
    except Exception as exc:
        logger.warning("[mem0-api] PUT /memory/%s FAILED user_id=%s error=%s", memory_id, user_id, exc)
        raise HTTPException(status_code=404, detail=f"更新失败: {exc}")
    return {"saved": True, "id": memory_id}


@router.delete("/{memory_id}")
async def remove_memory(memory_id: str, user: dict = Depends(get_current_user)):
    user_id = user["id"]
    logger.info("[mem0-api] DELETE /memory/%s user_id=%s", memory_id, user_id)
    m = get_memory()
    try:
        await m.delete(memory_id)
        logger.info("[mem0-api] DELETE /memory/%s OK user_id=%s", memory_id, user_id)
    except Exception as exc:
        logger.warning("[mem0-api] DELETE /memory/%s FAILED user_id=%s error=%s", memory_id, user_id, exc)
        raise HTTPException(status_code=404, detail=f"删除失败: {exc}")
    return {"deleted": True, "id": memory_id}


@router.get("/search")
async def search_memory(q: str, user: dict = Depends(get_current_user)):
    user_id = user["id"]
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="q is required")
    logger.info("[mem0-api] GET /memory/search user_id=%s query=%s", user_id, q[:50])
    m = get_memory()
    items = _results(await m.search(q, filters={"user_id": user_id}, top_k=8))
    logger.info("[mem0-api] GET /memory/search OK user_id=%s count=%d", user_id, len(items))
    return {"query": q, "memories": items}


@router.get("/overview")
async def get_overview(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = user["id"]
    logger.info("[mem0-api] GET /memory/overview user_id=%s", user_id)
    m = get_memory()
    items = _results(await m.get_all(filters={"user_id": user_id}, top_k=500))
    kg, eg = await load_graphs(db, user_id)
    logger.info(
        "[mem0-api] GET /memory/overview OK user_id=%s mem_count=%d kg_nodes=%d eg_nodes=%d",
        user_id, len(items), len(kg.get("nodes") or []), len(eg.get("nodes") or [])
    )
    return {
        "total": len(items),
        "memories": items,
        "knowledge_node_count": len(kg.get("nodes") or []),
        "error_node_count": len(eg.get("nodes") or []),
    }


# ---------------------------------------------------------------------------
# 知识图谱 / 错题图谱（graph_memory，差异化能力）
# ---------------------------------------------------------------------------

@router.get("/graph")
async def get_graphs(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = user["id"]
    logger.info("[graph-api] GET /memory/graph user_id=%s", user_id)
    kg, eg = await load_graphs(db, user_id)
    logger.info(
        "[graph-api] GET /memory/graph OK user_id=%s kg_nodes=%d eg_nodes=%d",
        user_id, len(kg.get("nodes") or []), len(eg.get("nodes") or [])
    )
    return {"knowledge_graph": kg, "error_graph": eg}


@router.post("/graph/delete")
async def delete_node(
    payload: GraphDeleteRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = user["id"]
    if not payload.node_id:
        raise HTTPException(status_code=400, detail="node_id is required")
    logger.info("[graph-api] POST /memory/graph/delete user_id=%s node_id=%s", user_id, payload.node_id)
    result = await delete_graph_node(db, user_id, payload.node_id)
    logger.info("[graph-api] POST /memory/graph/delete OK user_id=%s node_id=%s", user_id, payload.node_id)
    return result


@router.get("/dashboard")
async def get_dashboard(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = user["id"]
    logger.info("[mem0-api] GET /memory/dashboard user_id=%s", user_id)
    m = get_memory()
    items = _results(await m.get_all(filters={"user_id": user_id}, top_k=50))
    kg, eg = await load_graphs(db, user_id)
    high_risk = sorted(
        [n for n in (kg.get("nodes") or []) if n.get("status") == "active"],
        key=lambda n: -(n.get("risk") or 0),
    )[:5]
    frequent_errors = sorted(
        [n for n in (eg.get("nodes") or []) if (n.get("error_count") or 0) > 1],
        key=lambda n: -(n.get("error_count") or 0),
    )[:5]
    logger.info(
        "[mem0-api] GET /memory/dashboard OK user_id=%s mem_count=%d kg_nodes=%d eg_nodes=%d high_risk=%d frequent_err=%d",
        user_id, len(items), len(kg.get("nodes") or []), len(eg.get("nodes") or []),
        len(high_risk), len(frequent_errors)
    )
    memories = [{"content": i.get("memory", "")} for i in items]
    return {
        "memories": memories,
        # 前端 DashboardData.summary（"学习轨迹"区块）：把记忆条目拼成纯文本展示
        "summary": "\n".join(f"- {m['content']}" for m in memories if m.get("content")),
        "high_risk_points": high_risk,
        "frequent_errors": frequent_errors,
        "knowledge_node_count": len(kg.get("nodes") or []),
        "error_node_count": len(eg.get("nodes") or []),
    }
