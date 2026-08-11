"""高频问题语义聚类（学情分析四模块设计 §模块一 p2-faq-cluster）。

从 learning_events(verb=asked) 取近 N 天提问，embedding + 余弦阈值贪心聚类，落 course_faq
读模型。删后重算（幂等）。取代 P1-c 的 Redis 精确文本匹配--"这题怎么算"与"这个怎么算"
原文不等、但语义同簇。

embedding 存 JSON、cosine 在 Python 算：避 pgvector 在 SQLite 测试不可用（同 func.left 坑）。
embed_fn 可注入：默认走 get_embed_model（DashScope text-embedding-v3），测试传假函数确定性聚类。
"""
from __future__ import annotations

import logging
import math
import time

from sqlalchemy import delete, select

from core.db.database import AsyncSessionLocal, CourseFaq, LearningEvent

logger = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    """两向量余弦相似度；零向量返回 0。"""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


async def cluster_course_faqs(
    course_id: str,
    *,
    embed_fn=None,
    threshold: float = 0.86,
    days: int = 30,
) -> int:
    """重算一门课的 course_faq 聚类簇（先删后聚，幂等）。返回簇数；失败返回 0。

    embed_fn: ``async (texts: list[str]) -> list[list[float]]``，默认 get_embed_model。
    贪心：按时间升序遍历提问，cosine 最高的已有簇 >= threshold 则归入，否则新建簇。
    簇代表 = 种子提问原文（首条），centroid = 种子 embedding（固定，确定性）。
    """
    if embed_fn is None:
        from core.rag.llamaindex.pg_store import get_embed_model
        embed_fn = get_embed_model()._aget_text_embeddings

    try:
        async with AsyncSessionLocal() as db:
            cutoff = time.time() - days * 86400
            events = (await db.execute(
                select(LearningEvent.object_text, LearningEvent.created_at)
                .where(
                    LearningEvent.course_id == course_id,
                    LearningEvent.verb == "asked",
                    LearningEvent.created_at >= cutoff,
                    LearningEvent.object_text != "",
                )
                .order_by(LearningEvent.created_at.asc())
            )).all()

            # 幂等：先删旧簇
            await db.execute(delete(CourseFaq).where(CourseFaq.course_id == course_id))
            if not events:
                await db.commit()
                return 0

            # 去重后再 embedding：FAQ 数据高重复（同一问题被多人问），相同文本只需一次
            # API 调用（向量确定性相同），按 text->vec 映射回每条事件，聚类语义不变。
            unique_texts = list(dict.fromkeys(e.object_text for e in events))
            unique_vecs = await embed_fn(unique_texts)
            vec_by_text = dict(zip(unique_texts, unique_vecs))

            clusters: list[dict] = []  # {vec, question, count, last}
            for e in events:
                vec = vec_by_text[e.object_text]
                best_idx, best_sim = -1, 0.0
                for i, c in enumerate(clusters):
                    sim = _cosine(vec, c["vec"])
                    if sim > best_sim:
                        best_sim, best_idx = sim, i
                if best_idx >= 0 and best_sim >= threshold:
                    c = clusters[best_idx]
                    c["count"] += 1
                    c["last"] = e.created_at   # events 按 created_at 升序，e 必为簇内最新
                else:
                    clusters.append({
                        "vec": list(vec), "question": e.object_text,
                        "count": 1, "last": e.created_at,
                    })

            now = time.time()
            for c in clusters:
                db.add(CourseFaq(
                    course_id=course_id, question=c["question"], count=c["count"],
                    embedding=c["vec"], last_asked_at=c["last"], updated_at=now,
                ))
            await db.commit()
            return len(clusters)
    except Exception:
        logger.warning("cluster_course_faqs failed course=%s", course_id, exc_info=True)
        return 0
