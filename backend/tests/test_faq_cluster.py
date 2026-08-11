"""高频问题语义聚类测试（学情分析四模块设计 §模块一 p2-c）。

验证 cluster_course_faqs 的 embedding+阈值贪心聚类（注入假 embed_fn 确定性聚类）+
frequent_questions_merged 切读 course_faq。假 embed_fn 避开真实 API（测试 sk-test key 不可用）。
"""
from __future__ import annotations

import time

import pytest
from sqlalchemy import select


async def _fake_embed(texts: list[str]) -> list[list[float]]:
    """按关键词给 3 维正交向量：同关键词 cosine=1（同簇），不同=0（异簇）。"""
    out: list[list[float]] = []
    for t in texts:
        if "基尔霍夫" in t:
            out.append([1.0, 0.0, 0.0])
        elif "欧姆" in t:
            out.append([0.0, 1.0, 0.0])
        else:
            out.append([0.0, 0.0, 1.0])
    return out


@pytest.mark.asyncio
async def test_cluster_course_faqs_groups_semantic_duplicates(client):
    from core.analytics.faq_cluster import cluster_course_faqs
    from core.db.database import AsyncSessionLocal, CourseFaq, LearningEvent, User

    now = time.time()
    async with AsyncSessionLocal() as db:
        u = User(username="fc_u", password_hash="x")
        db.add(u)
        await db.flush()
        for i, q in enumerate(["基尔霍夫定律怎么用", "基尔霍夫怎么列方程", "欧姆定律是什么"]):
            db.add(LearningEvent(
                actor_user_id=u.id, course_id="c_fc", verb="asked",
                object_text=q, created_at=now + i,
            ))
        await db.commit()

    assert await cluster_course_faqs("c_fc", embed_fn=_fake_embed, threshold=0.86) == 2

    async with AsyncSessionLocal() as db:
        rows = {r.question: r.count for r in (await db.execute(
            select(CourseFaq).where(CourseFaq.course_id == "c_fc")
        )).scalars().all()}
    # 两个基尔霍夫提问合为一簇（count=2），欧姆单独一簇（count=1）
    kcl_q = next(q for q in rows if "基尔霍夫" in q)
    assert rows[kcl_q] == 2
    assert next(q for q in rows if "欧姆" in q) and rows[next(q for q in rows if "欧姆" in q)] == 1


@pytest.mark.asyncio
async def test_frequent_questions_merged_reads_course_faq(client):
    """聚类后 frequent_questions_merged 读 course_faq（按 count 降序），不走 SQL 兜底。

    课程只有 learning_events、无 Message -> 若走 SQL 兜底必为空；能返回 2 项即证读 course_faq。
    """
    from core.analytics.faq import frequent_questions_merged
    from core.analytics.faq_cluster import cluster_course_faqs
    from core.db.database import AsyncSessionLocal, LearningEvent, User

    now = time.time()
    async with AsyncSessionLocal() as db:
        u = User(username="fq_u", password_hash="x")
        db.add(u)
        await db.flush()
        for i, q in enumerate(["基尔霍夫怎么用", "基尔霍夫列方程", "欧姆定律"]):
            db.add(LearningEvent(
                actor_user_id=u.id, course_id="c_fq", verb="asked",
                object_text=q, created_at=now + i,
            ))
        await db.commit()

    await cluster_course_faqs("c_fq", embed_fn=_fake_embed)

    async with AsyncSessionLocal() as db:
        items = await frequent_questions_merged(db, "c_fq", 20)
    assert len(items) == 2
    assert items[0]["count"] == 2          # count 降序：基尔霍夫簇在前
    assert "基尔霍夫" in items[0]["question"]
    assert items[1]["count"] == 1
