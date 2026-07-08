"""LightRAG 图谱导出功能。

从 lightrag_engine.py 提取的实体/关系导出逻辑，
用于 graph_memory.py 的知识点目录映射。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def get_course_entities(course_id: str) -> list[dict]:
    """返回该课程 LightRAG 图谱中的所有实体节点。

    用于 graph_memory.py 的知识点目录映射，避免重复 LLM 提取。

    Args:
        course_id: 课程 ID

    Returns:
        实体列表，每个实体包含 {"id": str, "label": str, "type": str}
    """
    from core.rag.lightrag.llm_adapter import is_lightrag_available
    from core.rag.lightrag.instance_pool import lease_instance

    ok, reason = is_lightrag_available()
    if not ok:
        logger.warning("get_course_entities skipped: %s", reason)
        return []

    try:
        async with lease_instance(course_id) as rag:
            return await _collect_entities(rag, course_id)
    except RuntimeError as e:
        logger.warning("get_course_entities failed to get instance: %s", e)
        return []


async def _collect_entities(rag, course_id: str) -> list[dict]:
    """从 LightRAG 图谱收集实体节点（lease_instance 已保证引用计数配对）。"""
    # LightRAG 内部存储结构：chunk_entity_relation_graph 是 NetworkX 图
    entities: list[dict] = []
    if hasattr(rag, "chunk_entity_relation_graph"):
        graph = rag.chunk_entity_relation_graph
        if hasattr(graph, "get_all_nodes"):
            try:
                nodes = await graph.get_all_nodes()
                for node in nodes or []:
                    # 节点格式可能是字符串或 dict
                    if isinstance(node, dict):
                        entity_id = node.get("id") or node.get("entity_name") or str(node)
                        label = node.get("label") or node.get("entity_name") or str(node)
                        entity_type = node.get("type", "unknown")
                    else:
                        entity_id = str(node)
                        label = str(node)
                        entity_type = "unknown"
                    entities.append({
                        "id": entity_id,
                        "label": label,
                        "type": entity_type,
                    })
                logger.info(
                    "get_course_entities course=%s count=%d",
                    course_id, len(entities)
                )
            except Exception as e:
                logger.warning("get_course_entities get_all_nodes failed: %s", e)
        else:
            # 降级：直接读取 NetworkX 图的节点
            try:
                import networkx as nx
                if isinstance(graph, nx.Graph):
                    for node_id, node_data in graph.nodes(data=True):
                        label = node_data.get("entity_name") or node_data.get("label") or str(node_id)
                        entity_type = node_data.get("type", "unknown")
                        entities.append({
                            "id": str(node_id),
                            "label": label,
                            "type": entity_type,
                        })
                    logger.info(
                        "get_course_entities (NetworkX fallback) course=%s count=%d",
                        course_id, len(entities)
                    )
            except Exception as e:
                logger.warning("get_course_entities NetworkX fallback failed: %s", e)

    return entities


async def get_course_relations(course_id: str) -> list[dict]:
    """返回该课程 LightRAG 图谱中的所有关系（边）。

    用于 graph_memory.py 继承先修/相关边关系。

    Args:
        course_id: 课程 ID

    Returns:
        关系列表，每个关系包含 {"source": str, "target": str, "relation": str}
    """
    from core.rag.lightrag.llm_adapter import is_lightrag_available
    from core.rag.lightrag.instance_pool import lease_instance

    ok, reason = is_lightrag_available()
    if not ok:
        logger.warning("get_course_relations skipped: %s", reason)
        return []

    try:
        async with lease_instance(course_id) as rag:
            return await _collect_relations(rag, course_id)
    except RuntimeError as e:
        logger.warning("get_course_relations failed to get instance: %s", e)
        return []


async def _collect_relations(rag, course_id: str) -> list[dict]:
    """从 LightRAG 图谱收集关系边（lease_instance 已保证引用计数配对）。"""
    relations: list[dict] = []
    if hasattr(rag, "chunk_entity_relation_graph"):
        graph = rag.chunk_entity_relation_graph
        if hasattr(graph, "get_all_edges"):
            try:
                edges = await graph.get_all_edges()
                for edge in edges or []:
                    if isinstance(edge, dict):
                        source = edge.get("source") or edge.get("src_id") or ""
                        target = edge.get("target") or edge.get("tgt_id") or ""
                        relation = edge.get("relation") or edge.get("edge_type") or "related"
                        if source and target:
                            relations.append({
                                "source": source,
                                "target": target,
                                "relation": relation,
                            })
                logger.info(
                    "get_course_relations course=%s count=%d",
                    course_id, len(relations)
                )
            except Exception as e:
                logger.warning("get_course_relations get_all_edges failed: %s", e)
        else:
            # 降级：直接读取 NetworkX 图的边
            try:
                import networkx as nx
                if isinstance(graph, nx.Graph):
                    for src, tgt, edge_data in graph.edges(data=True):
                        relation = edge_data.get("relation") or edge_data.get("edge_type") or "related"
                        relations.append({
                            "source": str(src),
                            "target": str(tgt),
                            "relation": relation,
                        })
                    logger.info(
                        "get_course_relations (NetworkX fallback) course=%s count=%d",
                        course_id, len(relations)
                    )
            except Exception as e:
                logger.warning("get_course_relations NetworkX fallback failed: %s", e)

    return relations


__all__ = [
    "get_course_entities",
    "get_course_relations",
]