"""FAQ 高频问题测试。

p2-c 后 course_faq 语义聚类表为真相源、近 30 天重复提问 SQL 为冷启动兜底
（course_faq 未聚簇时）。本文件验证冷启动兜底口径；聚类本身见 test_faq_cluster.py。
（P1-c 的 faq_record Redis 写入已在 p2-c 退役，原 wiring 测试随之移除。）
"""
from __future__ import annotations

import os
import time

import pytest


@pytest.mark.asyncio
async def test_admin_faq_sql_fallback_when_course_faq_cold(client, admin_headers, course_with_code):
    """course_faq 未聚簇（cron 未跑/无 asked 事件）时，/admin/faq 用近 30 天重复提问兜底。"""
    from core.db.database import AsyncSessionLocal, Message, Session, User

    course_id = course_with_code["course_id"]

    async with AsyncSessionLocal() as db:
        stu = User(username=f"faq_{os.urandom(3).hex()}", password_hash="x")
        db.add(stu)
        await db.flush()
        sess = Session(id=f"s_{os.urandom(2).hex()}", course_id=course_id, user_id=stu.id)
        db.add(sess)
        await db.flush()

        now = time.time()
        # 同一前缀提问 2 次 -> 计数 ≥2，应出现在看板
        for _ in range(2):
            db.add(Message(session_id=sess.id, role="user", content="怎么列基尔霍夫电压方程", created_at=now))
        # 只出现 1 次 -> 被 having(count>=2) 过滤掉
        db.add(Message(session_id=sess.id, role="user", content="这是一条只问过一次的问题", created_at=now))
        await db.commit()

    r = await client.get(f"/api/admin/faq?course_id={course_id}", headers=admin_headers)
    assert r.status_code == 200, r.text
    questions = r.json()["questions"]
    texts = [q["question"] for q in questions]
    assert any("基尔霍夫" in t for t in texts), questions   # 重复提问兜底命中
    assert not any("只问过一次" in t for t in texts), questions  # 单次提问被过滤
