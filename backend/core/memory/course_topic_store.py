"""course_topic / course_topic_edge 表的查询层（运行时只读，零 LLM）。

id-align：把 ``knowledge_mastery`` 的 label/kp_id 对齐到 ``course_topic.topic_id``——
新数据写入应直接用 topic_id，本模块的 label 归一化映射兜底历史 mastery 数据（其 kp_id
曾是 entity_id 或 label 哈希，与 topic_id 不一致）。

门控（``core.memory.proactive.decide_stitch``）经本层读主题图：前置闭包回溯用
``get_prereq_predecessors``，运行时版与 ``scripts.eval_memory.stitch_cases.prereq_predecessors``
（内存图版）同语义，保证 CI 与运行时口径一致。
"""
from __future__ import annotations


def _norm_label(s: str) -> str:
    """label 归一化：小写 + 去非字母数字（与 build_course_topics.merge_synonymous 同口径）。"""
    return "".join(c for c in (s or "").lower() if c.isalnum())


async def get_course_topic_map(course_id: str, db) -> dict[str, str]:
    """course_id 的 ``label_norm -> topic_id`` 映射。门控用此把 mastery 的 label 对齐到 topic_id。"""
    from core.db.database import CourseTopic
    from sqlalchemy import select

    rows = (
        await db.execute(
            select(CourseTopic.topic_id, CourseTopic.label).where(
                CourseTopic.course_id == course_id
            )
        )
    ).all()
    return {_norm_label(label): tid for tid, label in rows if label}


async def resolve_topic_id(label: str, course_id: str, db) -> str | None:
    """label -> topic_id（归一化兜底）。找不到返回 None（门控按 unknown 处理）。

    id-align 的兜底路径：历史 mastery 行 kp_id 非 topic_id 时，靠其 label 经归一化映射回 topic_id。
    新数据应在写入时直接落 topic_id，逐步淘汰对本函数的依赖。
    """
    m = await get_course_topic_map(course_id, db)
    return m.get(_norm_label(label))


async def get_prereq_predecessors(topic_id: str, course_id: str, db) -> set[str]:
    """从 ``course_topic_edge`` 表递归回溯 topic_id 的全部（直接+间接）前置主题。

    运行时版，与 ``stitch_cases.prereq_predecessors``（内存图版）同语义：沿 prerequisite 边
    从 dst 回溯到 src，直到边界。门控用前置闭包去掉 matched 后 ∩ 高 risk 掌握度找未问的前置缺口。
    """
    from core.db.database import CourseTopicEdge
    from sqlalchemy import select

    preds: set[str] = set()
    frontier = [topic_id]
    while frontier:
        cur = frontier.pop()
        rows = (
            await db.execute(
                select(CourseTopicEdge.src_topic_id).where(
                    CourseTopicEdge.course_id == course_id,
                    CourseTopicEdge.dst_topic_id == cur,
                )
            )
        ).scalars().all()
        for src in rows:
            if src not in preds:
                preds.add(src)
                frontier.append(src)
    return preds


__all__ = ["get_course_topic_map", "resolve_topic_id", "get_prereq_predecessors"]
