"""course_topic / course_topic_edge 表的查询层（运行时只读，零 LLM）。

id-align：把 ``knowledge_mastery`` 的 label/kp_id 对齐到 ``course_topic.topic_id``——
新数据写入应直接用 topic_id，本模块的 label 归一化映射兜底历史 mastery 数据（其 kp_id
曾是 entity_id 或 label 哈希，与 topic_id 不一致）。

门控（``core.memory.proactive.decide_stitch``）经本层读主题图：前置闭包回溯用
``get_prereq_predecessors``，运行时版与 ``scripts.eval_memory.stitch_cases.prereq_predecessors``
（内存图版）同语义，保证 CI 与运行时口径一致。
"""
from __future__ import annotations

import math

# 主题匹配阈值（余弦）：**写侧**（topic_match.match_topics 的命中过滤）专用。
# 门控读侧（proactive._match_topic 的 unknown 判定）有独立常量
# proactive.UNKNOWN_THRESHOLD——两者刻意分开：写侧判定召回放宽（plan §4a-2）
# 不得连带读侧降阈（会把无关问句错认成主题，放大门控误报）。当前数值相等，
# 但无任何机制保证相等，改动任一侧时只改各自模块。
TOPIC_MATCH_THRESHOLD = 0.55


def _norm_label(s: str) -> str:
    """label 归一化：小写 + 去非字母数字。

    灌入侧（build_course_topics）与门控读出侧（proactive）共用，保证 id-align 的 label→topic_id
    映射在写入/读出两侧同口径——改这里即双侧同步，避免漂移。
    """
    return "".join(c for c in (s or "").lower() if c.isalnum())


def cosine(a: list[float], b: list[float]) -> float:
    """JSON-stored embedding 的余弦相似度（Python 算，避 pgvector 的 SQLite 不兼容）。

    集中供门控匹配（proactive._match_topic）与建库去重（course_topic_builder.merge_synonymous）
    复用，避免余弦算法在两侧各落一份。与 analytics.faq_cluster._cosine 同模式。
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def topic_map_from_rows(rows) -> dict[str, str]:
    """从已取的 topic 行构建 ``label_norm -> topic_id``（get_course_topic_map 的零查询版）。

    供已持有 topic 行的调用方（如学生 payload 的 fetch_course_topics 结果）本地构建，
    免为归一口径再发一次同表查询；归一化仍走 _norm_label（单一真相源）。
    """
    return {_norm_label(r.label): r.topic_id for r in rows if r.label}


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
    return topic_map_from_rows(rows)


async def resolve_topic_id(label: str, course_id: str, db) -> str | None:
    """label -> topic_id（归一化兜底）。找不到返回 None（门控按 unknown 处理）。

    id-align 的兜底路径：历史 mastery 行 kp_id 非 topic_id 时，靠其 label 经归一化映射回 topic_id。
    新数据应在写入时直接落 topic_id，逐步淘汰对本函数的依赖。
    """
    m = await get_course_topic_map(course_id, db)
    return m.get(_norm_label(label))


async def get_prereq_predecessors(topic_id: str, course_id: str, db) -> set[str]:
    """从 ``course_topic_edge`` 表递归回溯 topic_id 的全部（直接+间接）前置主题。

    一次拉全边内存 BFS（非每节点一次 round-trip），与 ``stitch_cases.prereq_predecessors``
    （内存图版）同算法：沿 prerequisite / prerequisite_inferred 边从 dst 回溯到 src（显式
    relation 过滤——防未来往同表加 related 等非先修边后被当成「必须先学」）。门控用前置
    闭包去掉 matched 后 ∩ 高 risk 掌握度找未问的前置缺口。
    """
    from core.db.database import CourseTopicEdge
    from sqlalchemy import select

    rows = (
        await db.execute(
            select(CourseTopicEdge.src_topic_id, CourseTopicEdge.dst_topic_id).where(
                CourseTopicEdge.course_id == course_id,
                CourseTopicEdge.relation.in_(("prerequisite", "prerequisite_inferred")),
            )
        )
    ).all()
    adj: dict[str, list[str]] = {}
    for src, dst in rows:
        adj.setdefault(dst, []).append(src)
    preds: set[str] = set()
    frontier = [topic_id]
    while frontier:
        for src in adj.get(frontier.pop(), []):
            if src not in preds:
                preds.add(src)
                frontier.append(src)
    return preds


async def fetch_course_topics(db, course_id: str) -> list:
    """主题行查询（教师/学生两个 GET payload 的单一真相源）。

    显式列查询不含 ``embedding``——1024 维 JSON 列视图层无用，全列取回白拉数百 KB。
    返回 Row 对象（属性访问与 ORM 实体一致）。
    """
    from core.db.database import CourseTopic
    from sqlalchemy import select

    return (
        await db.execute(
            select(
                CourseTopic.topic_id, CourseTopic.label, CourseTopic.definition,
                CourseTopic.source_section, CourseTopic.order_idx,
                CourseTopic.safety_weight, CourseTopic.difficulty_b,
            )
            .where(CourseTopic.course_id == course_id)
            .order_by(CourseTopic.order_idx)
        )
    ).all()


async def fetch_course_edges(db, course_id: str) -> list:
    """先修边行查询（同上，两 payload 共用）。"""
    from core.db.database import CourseTopicEdge
    from sqlalchemy import select

    return (
        await db.execute(
            select(
                CourseTopicEdge.src_topic_id, CourseTopicEdge.dst_topic_id,
                CourseTopicEdge.relation, CourseTopicEdge.confidence,
                CourseTopicEdge.verified_by,
            ).where(CourseTopicEdge.course_id == course_id)
        )
    ).all()


def align_mastery_rows(topic_map: dict[str, str], rows) -> dict[str, object]:
    """mastery 行对齐到 topic_id：kp_id 直接命中，legacy 哈希 id 走 label 归一兜底。

    id-align 的单一真相源——门控读侧（proactive._read_mastery）与学生端着色
    （api.kb_indexing 学生 payload）共调，保证"门控判 weak / 图谱标红"两侧同口径；
    对齐规则演进只改此处。返回 {topic_id: mastery行}，同 tid 多行时后行覆盖（last-wins，
    与两调用方原先的循环覆盖语义一致）。
    """
    topic_ids = set(topic_map.values())
    aligned: dict[str, object] = {}
    for r in rows:
        tid = r.kp_id if r.kp_id in topic_ids else topic_map.get(_norm_label(r.label))
        if tid:
            aligned[tid] = r
    return aligned


__all__ = [
    "get_course_topic_map", "topic_map_from_rows", "resolve_topic_id",
    "get_prereq_predecessors", "fetch_course_topics", "fetch_course_edges",
    "align_mastery_rows",
]
