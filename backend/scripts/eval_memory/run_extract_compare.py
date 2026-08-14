"""抽取三臂对比：开放 LLM vs catalog 选择题 vs 零 LLM 闭集。

Usage:
    python -m scripts.eval_memory.run_extract_compare [--db-url sqlite+aiosqlite:///:memory:]

三臂（plan 阶段5a）：
- arm=open_llm      graph_memory._extract_from_conversation（开放抽取，需 LLM key）
- arm=catalog       graph_memory._extract_from_conversation_with_catalog（选择题，需 LLM key）
- arm=closed_set    topic_match.match_topics（零 LLM，embedding 最近邻闭集）

Gold 用 stitch_cases 的课程主题图与每条 case 的 matched_topic。主指标：主题一致率
（抽取主 topic == gold matched_topic）。副指标：LLM token/回合（closed_set 恒 0）、
embedding 调用次数。不跑 LongMemEval（域不匹配，且我们评的是闭集对齐率不是长程 QA）。

LLM 两臂离线跑（要真实 key/网络），CI 只跑 closed_set 臂 + gold 一致性框架自检。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

# 与 run.py 同款的环境前置（必须在 import backend 模块前设置）
os.environ.setdefault("DB__URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECURITY__JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DB__REDIS_URL", "memory://")
os.environ.setdefault("LLM__API_KEY", "sk-test")
os.environ.setdefault("SECURITY__ALLOWED_ORIGINS", "*")
os.environ.setdefault("TESTING", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


async def _seed_topics(db) -> str:
    """stitch_cases 主题图灌入 course_topic，embedding 用真实 embedder（与生产
    build_course_topics 同口径）——闭集臂的最近邻才有意义。embedder 不可用则抛错
    （调用方按臂跳过，不用玩具向量凑数——凑出来的一致率是假数字）。"""
    from core.db.database import CourseTopic
    from core.rag.llamaindex.pg_store import get_embed_model

    from .stitch_cases import COURSE_TOPICS

    course_id = "eval_extract"
    embed = get_embed_model()
    for t in COURSE_TOPICS:
        emb = await embed._aget_text_embedding(f"{t['label']}：{t['definition']}")
        db.add(CourseTopic(
            course_id=course_id, topic_id=t["topic_id"], label=t["label"],
            definition=t["definition"], embedding=emb,
        ))
    await db.commit()
    return course_id


async def _arm_closed_set(db, course_id: str, case: dict) -> dict:
    """零 LLM 闭集臂：match_topics 最近邻。embedding mock 见 main（离线跑）。"""
    from core.memory.topic_match import match_topics

    hits = await match_topics(case["query"], course_id, db)
    main_topic = hits[0][0] if hits else None
    return {"main_topic": main_topic, "llm_tokens": 0, "embed_calls": 1}


async def _arm_open_llm(db, course_id: str, case: dict) -> dict:
    """开放 LLM 臂：_extract_from_conversation 的 knowledge_points 标签 → 最近 topic。"""
    from core.memory import graph_memory

    extracted = await graph_memory._extract_from_conversation(
        course_id, case["query"], "（assistant 回复略）"
    )
    kps = (extracted or {}).get("knowledge_points") or []
    labels = [kp.get("label") or "" for kp in kps]
    return {"labels": labels, "llm_tokens": 1, "embed_calls": 0, "main_topic": labels[0] if labels else None}


async def _arm_catalog(db, course_id: str, case: dict) -> dict:
    """catalog 选择题臂：_extract_from_conversation_with_catalog。"""
    from core.memory import graph_memory

    extracted = await graph_memory._extract_from_conversation_with_catalog(
        course_id, case["query"], "（assistant 回复略）"
    )
    kps = (extracted or {}).get("knowledge_points") or []
    labels = [kp.get("label") or "" for kp in kps]
    return {"labels": labels, "llm_tokens": 1, "embed_calls": 0, "main_topic": labels[0] if labels else None}


ARMS = {
    "closed_set": _arm_closed_set,
    "open_llm": _arm_open_llm,
    "catalog": _arm_catalog,
}


async def _run(arm_names: list[str]) -> dict:
    from core.db.database import AsyncSessionLocal, close_db, init_db

    from .stitch_cases import STITCH_CASES

    await init_db()
    out: dict = {}
    try:
        async with AsyncSessionLocal() as db:
            try:
                course_id = await _seed_topics(db)
            except Exception as exc:
                # embedder/网络不可用 → 如实报 None，不用玩具向量凑数
                logger.warning("[extract_compare] seed topics 失败（embedding 不可用？）：%s", exc)
                for arm in arm_names:
                    out[arm] = {"topic_accuracy": None, "reason": "embedder_unavailable"}
                return out
            for arm in arm_names:
                fn = ARMS[arm]
                hit = 0
                total = 0
                details: list[str] = []
                for c in STITCH_CASES:
                    gold = c["matched_topic"]
                    if gold is None:
                        continue  # unknown 例不评主题一致率
                    total += 1
                    try:
                        r = await fn(db, course_id, c)
                    except Exception as exc:
                        details.append(f"{c['id']}: error {exc!r}")
                        continue
                    if r.get("main_topic") == gold:
                        hit += 1
                    else:
                        details.append(f"{c['id']}: got={r.get('main_topic')!r} gold={gold}")
                out[arm] = {
                    "topic_accuracy": round(hit / total, 4) if total else 0.0,
                    "hit": hit, "total": total, "details": details,
                }
                logger.info("[extract_compare] %s: %d/%d = %.3f", arm, hit, total, hit / total if total else 0)
    finally:
        await close_db()
    return out


def main() -> None:
    from . import config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms", default="closed_set",
        help="逗号分隔：closed_set / open_llm / catalog（LLM 两臂需真实 key）",
    )
    args = parser.parse_args()
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]

    summary = asyncio.run(_run(arm_names))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "extract_compare.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[extract_compare] → %s", out)


if __name__ == "__main__":
    main()
