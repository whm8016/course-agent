"""导出 mycourse 课程图快照（表E评测的确定性图源）。

从 DB ``course_topic`` / ``course_topic_edge`` 导出主题与先修边（**不含** embedding），
写 ``graph_mycourse_v2.json``。评测 runner（``run_stitch_table_e``）只读快照——零 DB
依赖、可复现、零 LLM 臂可进 CI。抽取逻辑换版后重跑本脚本刷新快照即可。

快照确定性排序（topics 按 topic_id、edges 按 src,dst），不含时间戳，git diff 友好。
``_v2`` 对应课文锚定抽取（193 主题 / 225 边，prerequisite + prerequisite_inferred）。
与 ``data/course_topics/course_mycourse_topics.json`` 的关系：那是 builder 重建时自动
刷新的全量产物（含 embedding/definition/evidence），本快照是评测用的精简投影（仅
topic_id/label/src/dst/relation），手动重跑刷新——评测绑定提交内的快照保证可复现。

Usage（backend 目录）:
    python -m scripts.eval_memory.export_graph_snapshot
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

COURSE_ID = "mycourse"
SNAPSHOT_PATH = "graph_mycourse_v2.json"  # 与本脚本同目录（scripts/eval_memory/）


async def _export() -> dict:
    from core.db.database import AsyncSessionLocal, close_db
    from core.memory.course_topic_store import fetch_course_edges, fetch_course_topics

    async with AsyncSessionLocal() as db:
        # fetch_* 是两查询的单一真相源（显式列、不拉 embedding 大列）
        topics = await fetch_course_topics(db, COURSE_ID)
        edges = await fetch_course_edges(db, COURSE_ID)
    await close_db()

    snapshot = {
        "course_id": COURSE_ID,
        "note": "表E评测图源快照；runner 不连 DB，重抽取后重跑 export 刷新",
        "topics": sorted(
            ({"topic_id": t.topic_id, "label": t.label} for t in topics),
            key=lambda x: x["topic_id"],
        ),
        "edges": sorted(
            (
                {"src": e.src_topic_id, "dst": e.dst_topic_id, "relation": e.relation}
                for e in edges
            ),
            key=lambda x: (x["src"], x["dst"]),
        ),
    }
    return snapshot


def main() -> None:
    snapshot = asyncio.run(_export())
    out = Path(__file__).resolve().parent / SNAPSHOT_PATH
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(
        "[export_graph] %s: %d topics / %d edges -> %s",
        snapshot["course_id"], len(snapshot["topics"]), len(snapshot["edges"]), out,
    )


if __name__ == "__main__":
    main()
