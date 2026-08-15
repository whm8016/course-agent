"""学生端课程主题图接口（GET /courses/{id}/topic-graph/me）测试。

覆盖：
- 权限：未选课学生 403（check_course_access 复用，Enrollment 校验）
- payload 形状：topics 带 my 掌握度着色数据；weak 与门控 C1 同口径（theta <= WEAK_MARGIN）
- id-align：kp_id 直接命中 + label 归一兜底
- 未构建课程：空列表 200（学生端空态，不 404）
"""
from __future__ import annotations

import os

import pytest


async def _register_student(client) -> tuple[dict, str]:
    """注册一名学生，返回 (auth_headers, user_id)。"""
    from core.db.database import AsyncSessionLocal, User
    from sqlalchemy import select

    username = f"st_{os.urandom(3).hex()}"
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "testpass123", "display_name": "S"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    async with AsyncSessionLocal() as db:
        row = (await db.execute(select(User).where(User.username == username))).scalar_one()
        uid = row.id
    return {"Authorization": f"Bearer {token}"}, uid


@pytest.mark.asyncio
async def test_student_topic_graph_denied_without_enrollment(client, course_with_code):
    course_id = course_with_code["course_id"]
    headers, _ = await _register_student(client)
    r = await client.get(f"/api/courses/{course_id}/topic-graph/me", headers=headers)
    assert r.status_code == 403, r.text


@pytest.mark.asyncio
async def test_student_topic_graph_payload_and_weak_flag(client, course_with_code):
    from core.db.database import (
        AsyncSessionLocal,
        CourseTopic,
        CourseTopicEdge,
        Enrollment,
        KnowledgeMastery,
    )
    from core.memory.proactive import WEAK_MARGIN

    course_id = course_with_code["course_id"]
    headers, uid = await _register_student(client)

    async with AsyncSessionLocal() as db:
        db.add(Enrollment(student_id=uid, course_id=course_id))
        db.add(CourseTopic(
            course_id=course_id, topic_id="戴维南定理", label="戴维南定理",
            definition="等效电源定理", source_section="第5章", order_idx=0,
        ))
        db.add(CourseTopic(
            course_id=course_id, topic_id="滤波器", label="滤波器",
            definition="频率选择电路", source_section="第11章", order_idx=1,
        ))
        db.add(CourseTopicEdge(
            course_id=course_id, src_topic_id="戴维南定理", dst_topic_id="滤波器",
        ))
        # 戴维南定理：薄弱（theta 远低于 WEAK_MARGIN）；滤波器：无记录（my=null）
        db.add(KnowledgeMastery(
            user_id=uid, course_id=course_id, kp_id="戴维南定理",
            label="戴维南定理", theta=WEAK_MARGIN - 0.5, risk=0.8, mastery=0.2,
            observation_count=3,
        ))
        await db.commit()

    r = await client.get(f"/api/courses/{course_id}/topic-graph/me", headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["topic_count"] == 2 and data["edge_count"] == 1
    by_tid = {t["topic_id"]: t for t in data["topics"]}
    # 薄弱主题：my.weak=True（门控 C1 同口径）；不暴露 safety_weight / embedding
    my = by_tid["戴维南定理"]["my"]
    assert my["weak"] is True and my["observation_count"] == 3
    assert "safety_weight" not in by_tid["戴维南定理"]
    # 无记录主题：my=null（前端显示灰色未开始）
    assert by_tid["滤波器"]["my"] is None
    assert data["edges"][0]["src"] == "戴维南定理"


@pytest.mark.asyncio
async def test_student_topic_graph_label_fallback_align(client, course_with_code):
    """历史 mastery 行 kp_id 非 topic_id 时，经 label 归一兜底对齐（id-align）。"""
    from core.db.database import AsyncSessionLocal, CourseTopic, Enrollment, KnowledgeMastery

    course_id = course_with_code["course_id"]
    headers, uid = await _register_student(client)
    async with AsyncSessionLocal() as db:
        db.add(Enrollment(student_id=uid, course_id=course_id))
        db.add(CourseTopic(
            course_id=course_id, topic_id="戴维南等效", label="戴维南等效",
            definition="d", source_section="第5章", order_idx=0,
        ))
        # kp_id 是历史 entity_id（非 topic_id），label 与主题一致 → 归一兜底应命中
        db.add(KnowledgeMastery(
            user_id=uid, course_id=course_id, kp_id="ent_9f2", label="戴维南等效",
            theta=0.5, risk=0.3, mastery=0.8, observation_count=1,
        ))
        await db.commit()

    r = await client.get(f"/api/courses/{course_id}/topic-graph/me", headers=headers)
    assert r.status_code == 200, r.text
    topic = r.json()["topics"][0]
    assert topic["my"] is not None and topic["my"]["weak"] is False


@pytest.mark.asyncio
async def test_student_topic_graph_unbuilt_empty(client, course_with_code):
    from core.db.database import AsyncSessionLocal, Enrollment

    course_id = course_with_code["course_id"]
    headers, uid = await _register_student(client)
    async with AsyncSessionLocal() as db:
        db.add(Enrollment(student_id=uid, course_id=course_id))
        await db.commit()

    r = await client.get(f"/api/courses/{course_id}/topic-graph/me", headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["topic_count"] == 0 and data["topics"] == [] and data["edges"] == []
