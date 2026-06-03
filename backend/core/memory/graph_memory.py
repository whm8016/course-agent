"""学习图谱记忆：知识点图谱 + 错题图谱。

参考 MathClaw 的双图谱架构，但存储改为 PostgreSQL JSON 列。
每次对话结束后可选触发 LLM 提取新节点，然后与已有图谱合并。

图谱格式：
{
  "nodes": [
    {
      "id": "kp:xxx",
      "type": "knowledge_point",
      "label": "...",
      "risk": 0.0-1.0,
      "mastery": 0.0-1.0,
      "importance": 0.0-1.0,
      "status": "active" | "candidate",
      "notes": "",
      "examples": [],
      "related_points": [],
      "updated_at": "ISO timestamp"
    }
  ],
  "edges": [
    {"source": "kp:a", "target": "kp:b", "relation": "prerequisite"}
  ]
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import TEXT_MODEL
from core.db.database import User
from core.llm.llm import client as async_openai_client

logger = logging.getLogger(__name__)

_ACTIVE_LIMIT = 20
_CANDIDATE_LIMIT = 12
_EXTRACT_EVERY_N_TURNS = 3

_EXTRACT_SYSTEM_PROMPT = """你是学习分析助手。根据下面的对话，提取学生涉及的知识点和错误模式。

返回严格 JSON（不要包含 markdown 围栏）：
{
  "knowledge_points": [
    {
      "label": "知识点名称",
      "risk": 0.0-1.0,
      "mastery": 0.0-1.0,
      "importance": 0.0-1.0,
      "notes": "简短说明",
      "examples": ["示例"],
      "related_points": ["相关知识点"]
    }
  ],
  "error_patterns": [
    {
      "label": "错误模式名称",
      "severity": 0.0-1.0,
      "repeated": false,
      "notes": "错因说明",
      "related_knowledge_points": ["对应知识点"],
      "correction_suggestions": ["建议"]
    }
  ]
}

规则：
- 只提取本次对话中实际出现的知识点和错误
- risk 越高表示学生越薄弱；mastery 越高表示越熟练
- 如果对话中没有新知识点或错误，返回空数组
- 不要编造不存在的内容"""


def _node_id(prefix: str, label: str) -> str:
    normalized = re.sub(r"\s+", "-", label.strip().lower())
    slug = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "-", normalized).strip("-")[:24]
    digest = hashlib.sha1(label.strip().encode("utf-8")).hexdigest()[:8]
    return f"{prefix}:{slug}-{digest}" if slug else f"{prefix}:item-{digest}"


def _clamp(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _blend(old: Any, new: Any, default: float) -> float:
    a = _clamp(old, default)
    b = _clamp(new, a)
    return round((a + b) / 2, 3)


def _merge_knowledge_graph(existing: dict, new_points: list[dict]) -> dict:
    """将新提取的知识点合并进已有图谱。"""
    nodes: list[dict] = list(existing.get("nodes") or [])
    edges: list[dict] = list(existing.get("edges") or [])
    node_map: dict[str, dict] = {n["id"]: n for n in nodes}

    for point in new_points:
        label = str(point.get("label") or "").strip()
        if not label:
            continue
        nid = _node_id("kp", label)
        now = datetime.now().isoformat()

        if nid in node_map:
            node = node_map[nid]
            node["risk"] = _blend(node.get("risk"), point.get("risk"), 0.5)
            node["mastery"] = _blend(node.get("mastery"), point.get("mastery"), 0.5)
            node["importance"] = _blend(node.get("importance"), point.get("importance"), 0.5)
            if point.get("notes"):
                node["notes"] = point["notes"]
            examples = list(node.get("examples") or [])
            for ex in (point.get("examples") or []):
                if ex and ex not in examples:
                    examples.append(ex)
            node["examples"] = examples[-8:]
            node["status"] = "active"
            node["updated_at"] = now
        else:
            node = {
                "id": nid,
                "type": "knowledge_point",
                "label": label,
                "risk": _clamp(point.get("risk"), 0.5),
                "mastery": _clamp(point.get("mastery"), 0.5),
                "importance": _clamp(point.get("importance"), 0.5),
                "status": "active" if len(node_map) < _ACTIVE_LIMIT else "candidate",
                "notes": str(point.get("notes") or ""),
                "examples": list(point.get("examples") or [])[:8],
                "related_points": list(point.get("related_points") or []),
                "updated_at": now,
            }
            node_map[nid] = node
            nodes.append(node)

        for rel in (point.get("related_points") or []):
            rel_id = _node_id("kp", rel)
            edge = {"source": nid, "target": rel_id, "relation": "related"}
            if edge not in edges:
                edges.append(edge)

    active = [n for n in nodes if n.get("status") == "active"]
    if len(active) > _ACTIVE_LIMIT:
        active.sort(key=lambda n: n.get("updated_at", ""))
        for n in active[: len(active) - _ACTIVE_LIMIT]:
            n["status"] = "candidate"

    candidates = [n for n in nodes if n.get("status") == "candidate"]
    if len(candidates) > _CANDIDATE_LIMIT:
        candidates.sort(key=lambda n: n.get("updated_at", ""))
        nodes = [n for n in nodes if n not in candidates[: len(candidates) - _CANDIDATE_LIMIT]]

    return {"nodes": nodes, "edges": edges}


def _merge_error_graph(existing: dict, new_errors: list[dict]) -> dict:
    """将新提取的错误模式合并进已有图谱。"""
    nodes: list[dict] = list(existing.get("nodes") or [])
    edges: list[dict] = list(existing.get("edges") or [])
    node_map: dict[str, dict] = {n["id"]: n for n in nodes}

    for error in new_errors:
        label = str(error.get("label") or "").strip()
        if not label:
            continue
        nid = _node_id("err", label)
        now = datetime.now().isoformat()

        if nid in node_map:
            node = node_map[nid]
            node["severity"] = _blend(node.get("severity"), error.get("severity"), 0.5)
            node["error_count"] = (node.get("error_count") or 0) + 1
            node["repeated"] = True
            if error.get("notes"):
                node["notes"] = error["notes"]
            suggestions = list(node.get("correction_suggestions") or [])
            for s in (error.get("correction_suggestions") or []):
                if s and s not in suggestions:
                    suggestions.append(s)
            node["correction_suggestions"] = suggestions[-5:]
            node["updated_at"] = now
        else:
            node = {
                "id": nid,
                "type": "error_pattern",
                "label": label,
                "severity": _clamp(error.get("severity"), 0.5),
                "repeated": bool(error.get("repeated")),
                "error_count": 1,
                "notes": str(error.get("notes") or ""),
                "related_knowledge_points": list(error.get("related_knowledge_points") or []),
                "correction_suggestions": list(error.get("correction_suggestions") or [])[:5],
                "updated_at": now,
            }
            node_map[nid] = node
            nodes.append(node)

        for kp_label in (error.get("related_knowledge_points") or []):
            kp_id = _node_id("kp", kp_label)
            edge = {"source": nid, "target": kp_id, "relation": "caused_by"}
            if edge not in edges:
                edges.append(edge)

    return {"nodes": nodes, "edges": edges}


async def _extract_from_conversation(
    course_id: str, user_message: str, assistant_answer: str
) -> dict[str, list] | None:
    """用 LLM 从一轮对话中提取知识点和错误模式。"""
    source = (
        f"课程: {course_id}\n\n"
        f"学生提问:\n{user_message[:1500]}\n\n"
        f"教师回答:\n{assistant_answer[:1500]}"
    )
    try:
        resp = await async_openai_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": source},
            ],
            temperature=0.1,
            max_tokens=1200,
            stream=False,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("graph extraction failed: %s", e)
    return None


async def load_graphs(db: AsyncSession, user_id: str) -> tuple[dict, dict]:
    """读取用户的两份图谱。"""
    row = (await db.execute(
        select(User.knowledge_graph, User.error_graph).where(User.id == user_id)
    )).first()
    if not row:
        return {"nodes": [], "edges": []}, {"nodes": [], "edges": []}
    kg = row.knowledge_graph if isinstance(row.knowledge_graph, dict) else {"nodes": [], "edges": []}
    eg = row.error_graph if isinstance(row.error_graph, dict) else {"nodes": [], "edges": []}
    return kg, eg


async def save_graphs(
    db: AsyncSession, user_id: str, *, knowledge: dict | None = None, error: dict | None = None
) -> None:
    """保存图谱到 DB。"""
    values: dict = {}
    if knowledge is not None:
        values["knowledge_graph"] = knowledge
    if error is not None:
        values["error_graph"] = error
    if values:
        await db.execute(update(User).where(User.id == user_id).values(**values))


async def update_graphs_from_conversation(
    db: AsyncSession,
    user_id: str,
    *,
    course_id: str,
    user_message: str,
    assistant_answer: str,
) -> bool:
    """对话结束后提取并合并图谱（仅在有实质内容时触发）。"""
    if not user_message.strip() or not assistant_answer.strip():
        return False

    extracted = await _extract_from_conversation(course_id, user_message, assistant_answer)
    if not extracted:
        return False

    kp_list = extracted.get("knowledge_points") or []
    err_list = extracted.get("error_patterns") or []
    if not kp_list and not err_list:
        return False

    kg, eg = await load_graphs(db, user_id)
    new_kg = _merge_knowledge_graph(kg, kp_list) if kp_list else kg
    new_eg = _merge_error_graph(eg, err_list) if err_list else eg
    await save_graphs(db, user_id, knowledge=new_kg, error=new_eg)
    logger.info("graphs updated user=%s kp_new=%d err_new=%d", user_id, len(kp_list), len(err_list))
    return True


async def delete_graph_node(db: AsyncSession, user_id: str, node_id: str) -> dict:
    """删除图谱中指定节点及其相关边。"""
    kg, eg = await load_graphs(db, user_id)

    if node_id.startswith("kp:"):
        kg["nodes"] = [n for n in kg["nodes"] if n["id"] != node_id]
        kg["edges"] = [e for e in kg["edges"] if e["source"] != node_id and e["target"] != node_id]
        await save_graphs(db, user_id, knowledge=kg)
    elif node_id.startswith("err:"):
        eg["nodes"] = [n for n in eg["nodes"] if n["id"] != node_id]
        eg["edges"] = [e for e in eg["edges"] if e["source"] != node_id and e["target"] != node_id]
        await save_graphs(db, user_id, error=eg)

    return {"knowledge_graph": kg, "error_graph": eg}
