"""建课程级主题图：灌种子（人工标注）或从讲义 chunk 抽取（LLM）——CLI 薄壳。

抽取逻辑（from-chunks）已下沉到 ``core.memory.course_topic_builder``（单一真相源，生产
worker/API 亦调它）；本脚本仅保留 CLI 入口 + seed 模式（``eval_memory.run_stitch_e2e`` 用
``seed_course_topics``）。

两种模式：
  --seed        用 eval_memory.stitch_cases 的人工电路实验主题+先修边灌入
                course_topic/course_topic_edge 并导出 JSON。让门控与评测立即可跑——
                主题与边等同教师审计过的标注。无 embedding API 时 embedding 留 NULL。
  --from-chunks 调 core.build_topic_graph：自动定位摄入审计 latest.json + 解析章节前缀 +
                强制算 embedding（门控必需）。--chunks/--model/--no-embed 在本模式下已由
                core 统管，不再生效（向后兼容忽略 + warning）；--force 删旧重建。

表是真相源，导出的 JSON 是其投影（评测复现、教师离线核对、论文附录）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

# 仿 eval_memory/run.py：连真实库前先确保环境变量（脚本可独立跑）
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

logger = logging.getLogger(__name__)


async def seed_course_topics(course_id: str, db, embed_model=None) -> dict:
    """种子模式：灌 eval_memory.stitch_cases 的人工电路主题+先修边。

    主题按 stitch_cases.COURSE_TOPICS 的列表序灌 order_idx；边按 PREREQ_EDGES。
    embedding 可选（label+definition 拼接算）。重复灌用 topic_id 去重（已存在则跳过）。
    embed/导出复用 core.memory.course_topic_builder（与 build_topic_graph 同源）。
    """
    from core.db.database import CourseTopic, CourseTopicEdge
    from sqlalchemy import select

    from core.memory.course_topic_builder import _aget_embed, export_json
    from scripts.eval_memory.stitch_cases import COURSE_TOPICS, PREREQ_EDGES

    # 已存在的 topic_id（去重，不重复灌）。select(单列) + .scalars() 已直接产出该列的值
    # （非 CourseTopic 对象），此前误加 .topic_id 取值在表非空时必抛 AttributeError。
    existing = set(
        (
            await db.execute(
                select(CourseTopic.topic_id).where(CourseTopic.course_id == course_id)
            )
        ).scalars()
    )

    n_topic = 0
    for i, t in enumerate(COURSE_TOPICS):
        if t["topic_id"] in existing:
            continue
        emb = await _aget_embed(embed_model, f"{t['label']}：{t['definition']}")
        db.add(
            CourseTopic(
                course_id=course_id,
                topic_id=t["topic_id"],
                label=t["label"],
                definition=t["definition"],
                source_section="人工标注(seed)",
                order_idx=i,
                embedding=emb,
            )
        )
        n_topic += 1

    n_edge = 0
    for src, dst in PREREQ_EDGES:
        if src in existing or dst in existing:
            # 已灌过则跳过边（粗粒度去重，重灌场景保守）
            continue
        db.add(
            CourseTopicEdge(
                course_id=course_id,
                src_topic_id=src,
                dst_topic_id=dst,
                relation="prerequisite",
                confidence=1.0,
                verified_by="seed",
            )
        )
        n_edge += 1
    await db.commit()

    # 读回导出 JSON
    topics = [
        {"topic_id": r.topic_id, "label": r.label, "definition": r.definition, "order_idx": r.order_idx}
        for r in (
            await db.execute(
                select(CourseTopic).where(CourseTopic.course_id == course_id).order_by(CourseTopic.order_idx)
            )
        ).scalars()
    ]
    edges = [
        {"src": r.src_topic_id, "dst": r.dst_topic_id, "relation": r.relation}
        for r in (
            await db.execute(
                select(CourseTopicEdge).where(CourseTopicEdge.course_id == course_id)
            )
        ).scalars()
    ]
    export_json(course_id, topics, edges)
    logger.info("[course_topics] seed done course=%s +%d主题 +%d边", course_id, n_topic, n_edge)
    return {"topics_added": n_topic, "edges_added": n_edge, "total_topics": len(topics), "total_edges": len(edges)}


async def _run(args) -> int:
    from core.db.database import AsyncSessionLocal, close_db, init_db

    embed_model = None
    # --no-embed 仅 seed 生效；from-chunks 强制算 embedding（build_topic_graph 内部，门控硬依赖）
    if args.mode == "seed" and not args.no_embed:
        try:
            from core.rag.llamaindex.pg_store import get_embed_model
            embed_model = get_embed_model()
        except Exception as exc:
            logger.warning("[course_topics] embed_model 不可用（seed --no_embed 或配 API 后重试）：%s", exc)

    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            if args.mode == "seed":
                result = await seed_course_topics(args.course_id, db, embed_model)
            else:
                # from-chunks：抽取逻辑下沉 core，自动定位 latest.json + 解析章节前缀 + 强制算 embedding。
                # --chunks/--model/--no-embed 在本模式下已由 core 统管，不再生效（向后兼容忽略）。
                if args.chunks:
                    logger.warning("[course_topics] --chunks 在 from-chunks 下已由 core 自动定位 latest.json，忽略")
                from core.memory.course_topic_builder import build_topic_graph
                result = await build_topic_graph(args.course_id, db, force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        await close_db()


def main() -> None:
    p = argparse.ArgumentParser(description="建课程级主题图（seed / from-chunks）")
    p.add_argument("--course-id", required=True)
    p.add_argument("--mode", choices=["seed", "from-chunks"], default="seed")
    p.add_argument("--chunks", help="[已废弃] from-chunks 现由 core 自动定位 latest.json，忽略")
    p.add_argument("--no-embed", action="store_true", help="跳过 embedding（仅 seed 生效；from-chunks 强制算）")
    p.add_argument("--force", action="store_true", help="from-chunks：删旧主题图重建（避开唯一约束）")
    p.add_argument(
        "--model",
        help="[已废弃] from-chunks 现走 COURSE_TOPIC__EXTRACT_MODEL（settings），忽略",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
