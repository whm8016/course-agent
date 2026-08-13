"""建课程级主题图：灌种子（人工标注）或从讲义 chunk 离线抽取（LLM）。

两种模式：
  --seed        用 eval_memory.stitch_cases 的人工电路实验主题+先修边灌入
                course_topic/course_topic_edge 并导出 JSON。让门控（proactive.decide_stitch）
                与评测立即可跑——主题与边等同教师审计过的标注。无 embedding API 时
                embedding 留 NULL（CI 用人工 matched_topic 不依赖 embedding；真实运行时
                门控算 S_t 时遇 NULL 降级）。
  --from-chunks 读讲义切块审计 JSON（chunk 开头带【章节: ...】前缀），LLM 按章抽主题+
                定义，embedding 去重合并同义，仅在「讲义顺序在前 + 同章或相邻章」候选对
                上抽前置边（含糊/双向丢弃，Goel 协议）。有课程库数据时跑，产物供教师核对。

表是真相源，导出的 JSON 是其投影（评测复现、教师离线核对、论文附录）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# 仿 eval_memory/run.py：连真实库前先确保环境变量（脚本可独立跑）
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")

logger = logging.getLogger(__name__)

JSON_DIR = Path("data/course_topics")


async def _aget_embed(embed_model, text: str) -> list[float] | None:
    if embed_model is None:
        return None
    try:
        return await embed_model._aget_text_embedding(text)
    except Exception as exc:
        logger.warning("[course_topics] embedding 失败（置 NULL）：%s", exc)
        return None


def export_json(course_id: str, topics: list[dict], edges: list[dict]) -> Path:
    """导出课程级 JSON artifact（表真相源的投影）。"""
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    out = JSON_DIR / f"course_{course_id}_topics.json"
    payload = {
        "course_id": course_id,
        "topics": topics,
        "edges": edges,
        "note": "表为真相源，本 JSON 为投影（评测复现/教师核对/论文附录用）",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[course_topics] JSON 导出 → %s（%d 主题 %d 边）", out, len(topics), len(edges))
    return out


async def seed_course_topics(course_id: str, db, embed_model=None) -> dict:
    """种子模式：灌 eval_memory.stitch_cases 的人工电路主题+先修边。

    主题按 stitch_cases.COURSE_TOPICS 的列表序灌 order_idx；边按 PREREQ_EDGES。
    embedding 可选（label+definition 拼接算）。重复灌用 topic_id 去重（已存在则跳过）。
    """
    from core.db.database import CourseTopic, CourseTopicEdge
    from sqlalchemy import select

    from scripts.eval_memory.stitch_cases import COURSE_TOPICS, PREREQ_EDGES

    # 已存在的 topic_id（去重，不重复灌）
    existing = {
        r.topic_id
        for r in (
            await db.execute(
                select(CourseTopic.topic_id).where(CourseTopic.course_id == course_id)
            )
        ).scalars()
    }

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


# -------------------- from-chunks 离线抽取（有课程库数据时跑）--------------------

_EXTRACT_TOPICS_PROMPT = """你是课程结构分析助手。下面是一节实验讲义的文本（开头标注了章节）。
请抽出本节实际教学的主题，每个主题给一句话定义。返回严格 JSON（不要 markdown 围栏）：
{"topics": [{"label": "主题名", "definition": "一句话定义"}]}
规则：只抽本节确实教的内容；主题粒度对齐一个可考核的知识点（如「戴维南等效」而非「电路」）；
没有明确主题返回空数组；不编造。"""

_EXTRACT_PREREQ_PROMPT = """你是课程结构分析助手。判断 A 是否是 B 的直接前置（学 B 前应先掌握 A）。
A：{a}
B：{b}
返回严格 JSON：{{"is_prereq": true|false, "confidence": 0.0-1.0}}
规则：只判「A 是 B 的直接前置」这一方向；不确定或双向都依赖时 is_prereq=false（宁可漏连不乱连）。"""


async def _llm_json(llm, model: str, prompt: str) -> dict | None:
    import re

    try:
        resp = await llm.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=800, stream=False,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("[course_topics] LLM 抽取失败：%s", exc)
        return None


async def extract_topics_from_chunks(
    course_id: str, chunks: list[dict], llm, model: str, embed_model=None
) -> list[dict]:
    """从讲义 chunk 按「章节」聚合抽主题。chunk: {text, section, order_idx}。

    同一 section 的 chunk 合并后再问一次 LLM（一章一次调用，省成本）。
    """
    by_section: dict[str, list[dict]] = {}
    for ck in chunks:
        sec = ck.get("section") or ck.get("source") or "未知章节"
        by_section.setdefault(sec, []).append(ck)

    topics: list[dict] = []
    order = 0
    for sec, group in by_section.items():
        text = "\n".join(ck["text"][:1500] for ck in group)[:4000]
        data = await _llm_json(llm, model, f"章节：{sec}\n\n{text}\n\n{_EXTRACT_TOPICS_PROMPT}")
        for t in (data or {}).get("topics", []) if isinstance(data, dict) else []:
            label = str(t.get("label") or "").strip()
            if not label:
                continue
            emb = await _aget_embed(embed_model, f"{label}：{t.get('definition','')}")
            topics.append({
                "topic_id": "",  # 合并去重后定（label 归一 + embedding 相似）
                "label": label,
                "definition": str(t.get("definition") or "").strip(),
                "source_section": sec,
                "order_idx": order,
                "embedding": emb,
            })
            order += 1
    return topics


def merge_synonymous(topics: list[dict], sim_threshold: float = 0.88) -> list[dict]:
    """合并同义主题：label 字符串归一 + embedding 余弦相似度去重。

    保留 order_idx 最小者，定义取较长者。纯函数、可单测。归一化与余弦复用
    course_topic_store（与门控读出侧同口径，避免漂移）。
    """
    from core.memory.course_topic_store import _norm_label, cosine

    merged: list[dict] = []
    seen: list[tuple[str, list[float]]] = []
    for t in sorted(topics, key=lambda x: x["order_idx"]):
        n = _norm_label(t["label"])
        dup = None
        for i, (sn, se) in enumerate(seen):
            if n == sn or (t["embedding"] and se and cosine(t["embedding"], se) >= sim_threshold):
                dup = i
                break
        if dup is not None:
            if len(t["definition"]) > len(merged[dup]["definition"]):
                merged[dup]["definition"] = t["definition"]
            continue
        seen.append((n, t["embedding"] or []))
        merged.append(t)
    for i, t in enumerate(merged):
        t["order_idx"] = i
        t["topic_id"] = _norm_label(t["label"])[:32] or f"topic{i}"
    return merged


async def extract_prereq_edges(
    course_id: str, topics: list[dict], llm, model: str
) -> list[dict]:
    """抽前置边：仅在 order_idx 在前 + 同章或相邻章候选对上判定（Goel 协议）。

    不做 O(n²) 全配对——讲义顺序在前 + 章节邻近才作候选，含糊/双向丢弃。
    """
    edges: list[dict] = []
    for i, dst in enumerate(topics):
        for src in topics[:i]:  # 只看顺序在前的
            # 章节邻近约束：同章或相邻（order 差小）
            if abs(src["order_idx"] - dst["order_idx"]) > 3:
                continue
            data = await _llm_json(
                llm, model,
                _EXTRACT_PREREQ_PROMPT.format(a=src["label"], b=dst["label"]),
            )
            if isinstance(data, dict) and data.get("is_prereq") and (data.get("confidence") or 0) >= 0.6:
                edges.append({
                    "src": src["topic_id"], "dst": dst["topic_id"],
                    "relation": "prerequisite", "confidence": float(data.get("confidence")),
                })
    return edges


async def build_from_chunks(
    course_id: str, chunks_path: Path, db, llm, model: str, embed_model=None
) -> dict:
    """from-chunks 完整流程：读 JSON → 按章抽主题 → 去重 → 抽边 → 灌表 → 导出 JSON。"""
    from core.db.database import CourseTopic, CourseTopicEdge

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = raw.get("chunks") if isinstance(raw, dict) else raw
    if not chunks:
        logger.warning("[course_topics] chunks 文件无内容：%s", chunks_path)
        return {"error": "no chunks"}

    topics = await extract_topics_from_chunks(course_id, chunks, llm, model, embed_model)
    topics = merge_synonymous(topics)
    edges = await extract_prereq_edges(course_id, topics, llm, model)

    for t in topics:
        db.add(CourseTopic(
            course_id=course_id, topic_id=t["topic_id"], label=t["label"],
            definition=t["definition"], source_section=t["source_section"],
            order_idx=t["order_idx"], embedding=t["embedding"],
        ))
    for e in edges:
        db.add(CourseTopicEdge(
            course_id=course_id, src_topic_id=e["src"], dst_topic_id=e["dst"],
            relation=e["relation"], confidence=e["confidence"],
        ))
    await db.commit()
    export_json(course_id, topics, edges)
    logger.info("[course_topics] from-chunks done course=%s %d主题 %d边", course_id, len(topics), len(edges))
    return {"topics": len(topics), "edges": len(edges)}


async def _run(args) -> int:
    from core.db.database import AsyncSessionLocal, close_db, init_db

    embed_model = None
    if not args.no_embed:
        try:
            from core.rag.llamaindex.pg_store import get_embed_model
            embed_model = get_embed_model()
        except Exception as exc:
            logger.warning("[course_topics] embed_model 不可用（--no_embed 或配 API 后重试）：%s", exc)

    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            if args.mode == "seed":
                result = await seed_course_topics(args.course_id, db, embed_model)
            else:
                from core.llm.llm import client as llm
                from settings import get_settings
                result = await build_from_chunks(
                    args.course_id, Path(args.chunks), db, llm, get_settings().llm.text_model, embed_model,
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        await close_db()


def main() -> None:
    p = argparse.ArgumentParser(description="建课程级主题图（seed / from-chunks）")
    p.add_argument("--course-id", required=True)
    p.add_argument("--mode", choices=["seed", "from-chunks"], default="seed")
    p.add_argument("--chunks", help="from-chunks 模式的切块审计 JSON 路径")
    p.add_argument("--no-embed", action="store_true", help="跳过 embedding（置 NULL）")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
