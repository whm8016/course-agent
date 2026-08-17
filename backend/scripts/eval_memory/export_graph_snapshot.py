"""导出表E评测图快照（确定性图源，三份、三个层级）。

- ``graph_mycourse_v2.json``：精简投影（topic_id/label + 边），决策层离线 runner
  （``run_stitch_table_e``）与 CI 闭包断言用。从 DB 导出，**只在 ``--refresh-v2``
  时重写**——默认覆盖会把库漂移带进提交，CI 的 193 主题锚会红。
- ``graph_mycourse_v3_emb.json``：**v2 图 + 真实向量**，e2e 层（``run_stitch_e2e``）
  种进内存库用。默认导出，**不连 DB**——主题/边直接取提交内 v2 快照（保证与 30
  案例/教师金标同一世界），embedding 用真实 embedder 对 label 重算。
- ``graph_mycourse_live.json``：**现网库图直出**（含现网 difficulty_b/safety_weight），
  live 全栈层（``run_stitch_e2e_live``）的案例锚定图——第三层跟「现在学生上课用的
  图」走，与 v2/v3 冻结世界互不覆写（``--live`` 只写这一个文件，不动 v2/v3）。

背景（2026-08-17）：库图已被新抽版重写（176 主题/298 边），与 v2（193/225）不同源，
案例锚点主题（欧姆定律等 6 个）已不在库图——v3 必须锚定 v2。生产建库 embedding
口径是 ``label：definition``（course_topic_builder._embed_one），v2 同代 definition
已随 builder 产物覆盖丢失，故降为 label-only 重算：向量真实、口径偏简，实际命中
质量由 e2e 的 match_ok 列度量（是评测的一部分，不是要修的 bug）。
difficulty_b=0 / safety_weight=1：v2 同代值不可考，且库现状就是全 0/全 1（全班
重折未跑）；换图重审时再 revisit。

向量唯一性：round 后两行 embedding 完全相同时 ``_match_topic`` 的 max 平局按遍历
顺序走，离线接线自检的确定性命中会被无故打破——导出期 fail-fast。

Usage（backend 目录）:
    python -m scripts.eval_memory.export_graph_snapshot                # 写 v3（需 EMBEDDING__API_KEY）
    python -m scripts.eval_memory.export_graph_snapshot --refresh-v2  # 额外重写 v2（需 DB）
    python -m scripts.eval_memory.export_graph_snapshot --live        # 只写 live（需 DB，不动 v2/v3）
"""
from __future__ import annotations

import argparse
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
SNAPSHOT_PATH = "graph_mycourse_v2.json"        # 与本脚本同目录（scripts/eval_memory/）
SNAPSHOT_PATH_V3 = "graph_mycourse_v3_emb.json"
SNAPSHOT_PATH_LIVE = "graph_mycourse_live.json"

_EMBED_CONCURRENCY = 8  # 照 course_topic_builder._EMBED_CONCURRENCY：裸 gather 会 429 风暴


def _load_v2() -> dict:
    path = Path(__file__).resolve().parent / SNAPSHOT_PATH
    if not path.exists():
        raise SystemExit(f"缺 {path.name}（评测锚定图）——先提交 v2 快照再导 v3")
    return json.loads(path.read_text(encoding="utf-8"))


async def _embed_labels(labels: list[str]) -> list[list[float]]:
    """label → 真实向量（round 前）。单条失败重试一次，仍失败 fail-fast——v3 缺向量
    会让对应主题在匹配里静默不可达，宁可让操作员重跑。"""
    from core.rag.llamaindex.pg_store import get_embed_model

    try:
        embed_model = get_embed_model()
    except Exception as exc:
        raise SystemExit(f"embed_model 构造失败（缺 EMBEDDING__API_KEY？）：{exc}") from exc

    sem = asyncio.Semaphore(_EMBED_CONCURRENCY)

    async def _one(label: str) -> list[float]:
        for attempt in (1, 2):
            async with sem:
                try:
                    return await embed_model._aget_text_embedding(label)
                except Exception as exc:
                    if attempt == 2:
                        raise SystemExit(f"label embedding 失败（重跑本脚本）：{label!r} -> {exc}") from exc
                    logger.warning("[export_graph] 重试 %r：%s", label, exc)
        return []  # 不可达（上面 raise 兜底）

    return list(await asyncio.gather(*(_one(l) for l in labels)))


def _assert_unique_embeddings(topics: list[dict]) -> None:
    """round 后向量两两不重（平局会让最近邻按遍历序走，评测确定性破坏）。"""
    seen: dict[tuple, str] = {}
    for t in topics:
        key = tuple(t["embedding"])
        if key in seen:
            raise SystemExit(
                f"embedding round 后重复：{t['topic_id']!r} 与 {seen[key]!r} 向量完全相同，"
                "最近邻匹配存在平局歧义，请先排查 embedding"
            )
        seen[key] = t["topic_id"]


async def _export_v3() -> dict:
    """v3 = 提交内 v2（主题/边）+ label 重算向量。零 DB 依赖。"""
    v2 = _load_v2()
    topics_v2 = sorted(v2["topics"], key=lambda x: x["topic_id"])
    embs = await _embed_labels([t["label"] for t in topics_v2])
    topics = [
        {
            "topic_id": t["topic_id"],
            "label": t["label"],
            # v2 同代值不可考：b=0（库现状全 0，全班重折未跑）/ sw=1（默认即行为不变）
            "difficulty_b": 0.0,
            "safety_weight": 1.0,
            "embedding": [round(x, 6) for x in emb],
        }
        for t, emb in zip(topics_v2, embs)
    ]
    _assert_unique_embeddings(topics)
    return {
        "course_id": v2["course_id"],
        "note": "表E e2e 图源快照 = v2 图 + label 重算向量（真实 embedding，label-only 口径；"
                "v2 同代 definition 已失传，命中质量由 e2e match_ok 列度量）；"
                "b=0/sw=1 为库现状口径；重跑本脚本刷新",
        "topics": topics,
        "edges": sorted(
            v2["edges"], key=lambda x: (x["src"], x["dst"])
        ),
    }


async def _fetch_course_graph() -> tuple[list, list]:
    """DB → (topics, edges)。v2 刷新与 live 直出共用同一查询源（单一真相源在
    ``course_topic_store.fetch_*``），两个 DB 快照的取数口径永不漂移。"""
    from core.db.database import AsyncSessionLocal, close_db
    from core.memory.course_topic_store import fetch_course_edges, fetch_course_topics

    async with AsyncSessionLocal() as db:
        topics = await fetch_course_topics(db, COURSE_ID)
        edges = await fetch_course_edges(db, COURSE_ID)
    await close_db()
    return topics, edges


def _edges_payload(edges) -> list[dict]:
    """边投影 + 确定性排序（v2 与 live 同一份，防两快照边口径各自漂移）。"""
    return sorted(
        (
            {"src": e.src_topic_id, "dst": e.dst_topic_id, "relation": e.relation}
            for e in edges
        ),
        key=lambda x: (x["src"], x["dst"]),
    )


async def _export_v2() -> dict:
    """v2 刷新（DB → 精简投影）。默认不跑——动它等于换评测世界，需连带重审案例。"""
    topics, edges = await _fetch_course_graph()
    return {
        "course_id": COURSE_ID,
        "note": "表E评测图源快照；runner 不连 DB，重抽取后重跑 export --refresh-v2 刷新",
        "topics": sorted(
            ({"topic_id": t.topic_id, "label": t.label} for t in topics),
            key=lambda x: x["topic_id"],
        ),
        "edges": _edges_payload(edges),
    }


async def _export_live() -> dict:
    """现网库图直出（live 全栈层的案例锚定图），含现网 difficulty_b / safety_weight
    （neg_class_hard 用现网真 b，全班 b=0 则如实记）。"""
    topics, edges = await _fetch_course_graph()
    return {
        "course_id": COURSE_ID,
        "note": "现网库图快照（live 全栈评测锚定图）；重抽取后重跑 export --live 刷新",
        "topics": sorted(
            (
                {
                    "topic_id": t.topic_id, "label": t.label,
                    "difficulty_b": float(t.difficulty_b or 0.0),
                    "safety_weight": float(t.safety_weight if t.safety_weight is not None else 1.0),
                }
                for t in topics
            ),
            key=lambda x: x["topic_id"],
        ),
        "edges": _edges_payload(edges),
    }


def _write_snapshot(snapshot: dict, name: str, tag: str) -> None:
    """确定性落盘（indent=1 + ensure_ascii=False，git diff 友好）+ 一行日志。"""
    out = Path(__file__).resolve().parent / name
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(
        "[export_graph] %s: %d topics / %d edges -> %s",
        tag, len(snapshot["topics"]), len(snapshot["edges"]), out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="导出表E评测图快照（默认写 v3：v2 图 + label 重算向量）")
    parser.add_argument(
        "--refresh-v2", action="store_true",
        help="额外重写 graph_mycourse_v2.json（需 DB；默认不动——防库漂移弄脏 CI 的 193 主题锚）",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="只写 graph_mycourse_live.json（现网库图直出，需 DB；不动 v2/v3）",
    )
    args = parser.parse_args()

    if args.live:
        if args.refresh_v2:
            raise SystemExit("--live 与 --refresh-v2 不组合：live 只导现网图快照（不动 v2/v3）")
        _write_snapshot(asyncio.run(_export_live()), SNAPSHOT_PATH_LIVE, "live")
        return

    if args.refresh_v2:
        # 先刷 v2 再导 v3：v3 从磁盘上的 v2 构建，顺序反了会让 v3 停在旧图
        # （v2 新、v3 旧——离线层与 e2e 层各读一份不同源的图）
        _write_snapshot(asyncio.run(_export_v2()), SNAPSHOT_PATH, "v2")
    _write_snapshot(asyncio.run(_export_v3()), SNAPSHOT_PATH_V3, "v3")


if __name__ == "__main__":
    main()
