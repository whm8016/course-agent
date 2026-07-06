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
from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from settings import get_settings
TEXT_MODEL = get_settings().llm.text_model
from core.db.database import User
from core.llm.llm import client as async_openai_client

logger = logging.getLogger(__name__)

_ACTIVE_LIMIT = 200  # 放宽限制，一门课知识点可能很多
_CANDIDATE_LIMIT = 100
_EXTRACT_EVERY_N_TURNS = 6

# 全局计数器，用于节流 LLM 提取频率
_turn_counter: dict[str, int] = {}  # user_id -> turn count

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
      "relations": [
        {"target": "另一知识点", "type": "prerequisite|related|part_of|requires"}
      ]
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
- relations.type:
  - prerequisite: 学这个知识点之前必须先掌握的目标知识点
  - related: 相关但非前置依赖
  - part_of: 当前知识点是目标知识点的子知识点
  - requires: 当前知识点需要用到目标知识点
- 如果对话中没有新知识点或错误，返回空数组
- 不要编造不存在的内容"""

# 基于 LightRAG 实体目录的匹配 Prompt（新路径）
_MATCH_SYSTEM_PROMPT = """你是学习分析助手。下面是该课程的知识点目录和一轮对话。

知识点目录（来自课程文档提取）：
{entity_catalog}

对话内容：
学生提问: {user_message}
教师回答: {assistant_answer}

任务：
1. 从知识点目录中选出本轮对话实际涉及的知识点（只选，不要新增）
2. 评估学生在每个涉及知识点上的表现变化
3. 识别错误模式（如果有）

返回严格 JSON（不要包含 markdown 围栏）：
{
  "matched_points": [
    {
      "entity_id": "目录中的实体ID",
      "label": "知识点名称",
      "mastery_delta": -0.1到0.1,
      "risk_delta": -0.1到0.1,
      "notes": "简短说明"
    }
  ],
  "error_patterns": [
    {
      "label": "错误模式名称",
      "severity": 0.0-1.0,
      "related_entity_ids": ["对应知识点ID"]
    }
  ]
}

规则：
- matched_points 只能包含目录中已有的 entity_id
- mastery_delta > 0 表示进步，< 0 表示退步
- risk_delta > 0 表示薄弱加剧，< 0 表示改善
- 如果对话中没有涉及任何知识点，matched_points 返回空数组
- 不要编造不存在的 entity_id"""


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
    """将新提取的知识点合并进已有图谱。

    支持两种格式：
    1. 旧格式：{label, risk, mastery, importance, ...}
    2. 新格式（基于目录匹配）：{entity_id, label, mastery_delta, risk_delta, notes}
    """
    nodes: list[dict] = list(existing.get("nodes") or [])
    edges: list[dict] = list(existing.get("edges") or [])
    node_map: dict[str, dict] = {n["id"]: n for n in nodes}

    for point in new_points:
        label = str(point.get("label") or "").strip()
        if not label:
            continue
        nid = _node_id("kp", label)
        now = datetime.now().isoformat()

        # 新格式：基于 entity_id 的 delta 更新
        entity_id = point.get("entity_id")
        is_delta_update = "mastery_delta" in point or "risk_delta" in point

        if nid in node_map:
            node = node_map[nid]
            if is_delta_update:
                # delta 更新：增量调整 mastery/risk
                mastery_delta = _clamp(point.get("mastery_delta"), 0)
                risk_delta = _clamp(point.get("risk_delta"), 0)
                node["mastery"] = _clamp(node.get("mastery", 0.5) + mastery_delta, 0.5)
                node["risk"] = _clamp(node.get("risk", 0.5) + risk_delta, 0.5)
                logger.debug(
                    "[graph] merge delta nid=%s mastery %.2f -> %.2f risk %.2f -> %.2f",
                    nid, node.get("mastery", 0.5) - mastery_delta, node["mastery"],
                    node.get("risk", 0.5) - risk_delta, node["risk"]
                )
            else:
                # 旧格式：平均值融合
                node["risk"] = _blend(node.get("risk"), point.get("risk"), 0.5)
                node["mastery"] = _blend(node.get("mastery"), point.get("mastery"), 0.5)
                node["importance"] = _blend(node.get("importance"), point.get("importance"), 0.5)

            if point.get("notes"):
                node["notes"] = point["notes"]
            if entity_id:
                node["entity_id"] = entity_id  # 记录 LightRAG 实体 ID
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
                "risk": _clamp(point.get("risk"), 0.5) if not is_delta_update else _clamp(0.5 + point.get("risk_delta", 0), 0.5),
                "mastery": _clamp(point.get("mastery"), 0.5) if not is_delta_update else _clamp(0.5 + point.get("mastery_delta", 0), 0.5),
                "importance": _clamp(point.get("importance"), 0.5),
                "status": "active" if len(node_map) < _ACTIVE_LIMIT else "candidate",
                "notes": str(point.get("notes") or ""),
                "examples": list(point.get("examples") or [])[:8],
                "related_points": list(point.get("related_points") or []),
                "entity_id": entity_id or "",
                "updated_at": now,
            }
            node_map[nid] = node
            nodes.append(node)

        # 处理边关系：支持多种边类型
        relations = point.get("relations") or []
        # 兼容旧格式 related_points
        if not relations and point.get("related_points"):
            relations = [{"target": r, "type": "related"} for r in point.get("related_points") or []]

        for rel in relations:
            target_label = rel.get("target") if isinstance(rel, dict) else rel
            rel_type = rel.get("type", "related") if isinstance(rel, dict) else "related"
            if not target_label:
                continue
            rel_id = _node_id("kp", target_label)
            edge = {"source": nid, "target": rel_id, "relation": rel_type}
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


async def _get_entity_catalog(course_id: str, max_entities: int = 100) -> str:
    """从 LightRAG 获取实体目录，格式化为 prompt 可用的文本。"""
    from settings import get_settings
    LIGHTRAG_ENABLED = get_settings().lightrag.enabled

    if not LIGHTRAG_ENABLED:
        return ""

    try:
        from core.rag import get_course_entities
        entities = await get_course_entities(course_id)
        if not entities:
            return ""

        # 格式化为 "ID: 名称 (类型)" 列表
        lines = []
        for i, e in enumerate(entities[:max_entities]):
            lines.append(f"{e['id']}: {e['label']} ({e['type']})")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("[graph] get_entity_catalog failed course_id=%s error=%s", course_id, e)
        return ""


async def _extract_from_conversation_with_catalog(
    course_id: str, user_message: str, assistant_answer: str
) -> dict[str, list] | None:
    """基于 LightRAG 实体目录的提取（新路径）：LLM 做选择题而非开放提取。"""
    entity_catalog = await _get_entity_catalog(course_id)
    if not entity_catalog:
        logger.info("[graph] extract_with_catalog NO_CATALOG course_id=%s, fallback to old", course_id)
        return None

    prompt = _MATCH_SYSTEM_PROMPT.format(
        entity_catalog=entity_catalog,
        user_message=user_message[:1500],
        assistant_answer=assistant_answer[:1500],
    )
    try:
        resp = await async_openai_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200,
            stream=False,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        if isinstance(data, dict):
            # 转换 matched_points 为 knowledge_points 格式（兼容下游）
            matched = data.get("matched_points") or []
            knowledge_points = []
            for mp in matched:
                knowledge_points.append({
                    "label": mp.get("label", ""),
                    "entity_id": mp.get("entity_id", ""),
                    "mastery_delta": mp.get("mastery_delta", 0),
                    "risk_delta": mp.get("risk_delta", 0),
                    "notes": mp.get("notes", ""),
                })
            # 保留错误模式
            error_patterns = data.get("error_patterns") or []
            logger.info(
                "[graph] extract_with_catalog OK course_id=%s matched=%d errors=%d",
                course_id, len(knowledge_points), len(error_patterns)
            )
            return {
                "knowledge_points": knowledge_points,
                "error_patterns": error_patterns,
                "source": "catalog_match",  # 标记来源
            }
    except Exception as e:
        logger.warning("[graph] extract_with_catalog FAILED course_id=%s error=%s", course_id, e)
    return None


async def _extract_from_conversation(
    course_id: str, user_message: str, assistant_answer: str
) -> dict[str, list] | None:
    """用 LLM 从一轮对话中提取知识点和错误模式。

    优先使用 LightRAG 实体目录（新路径），降级时走旧的开放提取。
    """
    from settings import get_settings
    LIGHTRAG_ENABLED = get_settings().lightrag.enabled

    logger.info(
        "[graph] extract START course_id=%s user_msg_len=%d assistant_len=%d lightrag=%s",
        course_id, len(user_message), len(assistant_answer), LIGHTRAG_ENABLED
    )

    # 新路径：基于 LightRAG 实体目录
    if LIGHTRAG_ENABLED:
        result = await _extract_from_conversation_with_catalog(course_id, user_message, assistant_answer)
        if result:
            return result
        # 否则降级到旧路径

    # 旧路径：开放提取
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
            kp_count = len(data.get("knowledge_points") or [])
            err_count = len(data.get("error_patterns") or [])
            logger.info("[graph] extract OK (fallback) course_id=%s kp=%d err=%d", course_id, kp_count, err_count)
            return data
    except Exception as e:
        logger.warning("[graph] extract FAILED course_id=%s error=%s", course_id, e)
    return None


async def load_graphs(db: AsyncSession, user_id: str) -> tuple[dict, dict]:
    """读取用户的两份图谱。"""
    logger.info("[graph] load_graphs FETCH user_id=%s", user_id)
    row = (await db.execute(
        select(User.knowledge_graph, User.error_graph).where(User.id == user_id)
    )).first()
    if not row:
        logger.info("[graph] load_graphs NO_DATA user_id=%s", user_id)
        return {"nodes": [], "edges": []}, {"nodes": [], "edges": []}
    kg = row.knowledge_graph if isinstance(row.knowledge_graph, dict) else {"nodes": [], "edges": []}
    eg = row.error_graph if isinstance(row.error_graph, dict) else {"nodes": [], "edges": []}
    logger.info(
        "[graph] load_graphs OK user_id=%s kg_nodes=%d kg_edges=%d eg_nodes=%d eg_edges=%d",
        user_id, len(kg.get("nodes") or []), len(kg.get("edges") or []),
        len(eg.get("nodes") or []), len(eg.get("edges") or [])
    )
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
        logger.info(
            "[graph] save_graphs SAVED user_id=%s saved_kg=%s saved_eg=%s kg_nodes=%d eg_nodes=%d",
            user_id, knowledge is not None, error is not None,
            len(knowledge.get("nodes") or []) if knowledge else 0,
            len(error.get("nodes") or []) if error else 0
        )


async def update_graphs_from_conversation(
    db: AsyncSession,
    user_id: str,
    *,
    course_id: str,
    user_message: str,
    assistant_answer: str,
    force: bool = False,  # 强制提取，忽略节流
) -> bool:
    """对话结束后提取并合并图谱（仅在有实质内容时触发）。

    Args:
        db: 数据库会话
        user_id: 用户 ID
        course_id: 课程 ID
        user_message: 用户消息
        assistant_answer: 助手回复
        force: 是否强制提取（忽略 _EXTRACT_EVERY_N_TURNS 节流）

    Returns:
        是否成功更新图谱
    """
    logger.info("[graph] update_graphs START user_id=%s course_id=%s", user_id, course_id)
    if not user_message.strip() or not assistant_answer.strip():
        logger.info("[graph] update_graphs SKIP user_id=%s (empty msg/answer)", user_id)
        return False

    # 节流：每 N 轮对话才触发一次 LLM 提取
    global _turn_counter
    if not force:
        _turn_counter[user_id] = _turn_counter.get(user_id, 0) + 1
        if _turn_counter[user_id] % _EXTRACT_EVERY_N_TURNS != 0:
            logger.info(
                "[graph] update_graphs THROTTLED user_id=%s turn=%d (every %d turns)",
                user_id, _turn_counter[user_id], _EXTRACT_EVERY_N_TURNS
            )
            return False

    extracted = await _extract_from_conversation(course_id, user_message, assistant_answer)
    if not extracted:
        logger.warning("[graph] update_graphs EXTRACT_FAIL user_id=%s", user_id)
        return False

    kp_list = extracted.get("knowledge_points") or []
    err_list = extracted.get("error_patterns") or []
    if not kp_list and not err_list:
        logger.info("[graph] update_graphs SKIP user_id=%s (no kp/err extracted)", user_id)
        return False

    kg, eg = await load_graphs(db, user_id)
    logger.info(
        "[graph] update_graphs LOADED user_id=%s old_kg_nodes=%d old_eg_nodes=%d new_kp=%d new_err=%d",
        user_id, len(kg.get("nodes") or []), len(eg.get("nodes") or []), len(kp_list), len(err_list)
    )
    new_kg = _merge_knowledge_graph(kg, kp_list) if kp_list else kg
    new_eg = _merge_error_graph(eg, err_list) if err_list else eg
    await save_graphs(db, user_id, knowledge=new_kg, error=new_eg)
    logger.info(
        "[graph] update_graphs DONE user_id=%s new_kg_nodes=%d new_eg_nodes=%d",
        user_id, len(new_kg.get("nodes") or []), len(new_eg.get("nodes") or [])
    )
    return True


async def delete_graph_node(db: AsyncSession, user_id: str, node_id: str) -> dict:
    """删除图谱中指定节点及其相关边。"""
    logger.info("[graph] delete_graph_node START user_id=%s node_id=%s", user_id, node_id)
    kg, eg = await load_graphs(db, user_id)

    if node_id.startswith("kp:"):
        before = len(kg["nodes"])
        kg["nodes"] = [n for n in kg["nodes"] if n["id"] != node_id]
        kg["edges"] = [e for e in kg["edges"] if e["source"] != node_id and e["target"] != node_id]
        await save_graphs(db, user_id, knowledge=kg)
        logger.info("[graph] delete_graph_node KP_DELETED user_id=%s node_id=%s removed_nodes=%d", user_id, node_id, before - len(kg["nodes"]))
    elif node_id.startswith("err:"):
        before = len(eg["nodes"])
        eg["nodes"] = [n for n in eg["nodes"] if n["id"] != node_id]
        eg["edges"] = [e for e in eg["edges"] if e["source"] != node_id and e["target"] != node_id]
        await save_graphs(db, user_id, error=eg)
        logger.info("[graph] delete_graph_node ERR_DELETED user_id=%s node_id=%s removed_nodes=%d", user_id, node_id, before - len(eg["nodes"]))

    return {"knowledge_graph": kg, "error_graph": eg}
