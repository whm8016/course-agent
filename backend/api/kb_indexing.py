"""知识库索引入队编排（admin / teacher 共用）。

把"触发索引"接口里重复的状态校验 / 文件查询 / DB 预置 / ARQ 入队逻辑收口到这里，
admin 与 teacher 入口只保留各自的权限校验（_get_kb_or_404 / _get_owned_kb），其余
全部委托给本模块。

放 api 层（而非 core/rag/）是因为要调 api.courses.invalidate_courses_cache——放 core
会形成 core→api 循环依赖。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.courses import invalidate_courses_cache
from core.db.database import CourseTopic, KBFile, KbBuild, KnowledgeBase

logger = logging.getLogger(__name__)

# 后端展示名（409/日志用）
_BACKEND_LABELS = {"llamaindex_pg": "pgvector"}


def _backend_label(backend: str) -> str:
    return _BACKEND_LABELS.get(backend, "LightRAG")


async def get_build(db: AsyncSession, kb_id: str, backend: str) -> KbBuild | None:
    """取 (kb_id, backend) 对应的 kb_builds 行；无则 None。"""
    r = await db.execute(
        select(KbBuild).where(KbBuild.kb_id == kb_id, KbBuild.backend == backend)
    )
    return r.scalar_one_or_none()


async def get_or_create_build(db: AsyncSession, kb_id: str, backend: str) -> KbBuild:
    """取或建 (kb_id, backend) 的 kb_builds 行（status=pending）。在调用方事务内 flush 取 id。"""
    build = await get_build(db, kb_id, backend)
    if build is None:
        build = KbBuild(kb_id=kb_id, backend=backend, status="pending")
        db.add(build)
        await db.flush()
    return build


async def trigger_kb_indexing(
    db: AsyncSession,
    kb: KnowledgeBase,
    course_id: str,
    backend: str = "lightrag",
    force: bool = False,
    resume: bool = False,
) -> dict:
    """触发知识库索引（入队 ARQ run_indexing），按 backend 写 kb_builds 行。

    kb 由调用方按各自权限校验后传入（admin 用 _get_kb_or_404，teacher 用 _get_owned_kb）。
    backend: lightrag | llamaindex_pg——每后端一条 kb_builds（get-or-create），状态/进度隔离，
    一课程可同时构建两后端。封装：状态校验 → 文件查询/存在性 → resume 计算 → DB 预置
    indexing → ARQ 入队 → 返回。

    ARQ/Redis 不可用时直接 503，由 get_db 回滚 build status——不降级 BackgroundTasks，避免
    索引跑到 gunicorn 多进程、跨进程写同一份 lightrag_store 导致重复文档刷屏/损坏/卡死。
    """
    build = await get_or_create_build(db, kb.id, backend)
    if build.status == "indexing" and not force:
        raise HTTPException(
            status_code=409,
            detail=f"{_backend_label(backend)} 正在索引中，请等待完成后再试",
        )

    files_result = await db.execute(select(KBFile).where(KBFile.kb_id == kb.id))
    files = files_result.scalars().all()
    if not files:
        raise HTTPException(status_code=400, detail="知识库中没有文件，请先上传文件")

    file_paths = [f.file_path for f in files if Path(f.file_path).exists()]
    if not file_paths:
        raise HTTPException(status_code=400, detail="文件在磁盘上不存在，请重新上传")

    # 断点续传：error / paused 状态且有进度记录时生效（按本 build 的进度）
    resume_from = 0
    if resume and build.status in ("error", "paused") and build.chunks_done > 0:
        resume_from = build.chunks_done
        logger.info(
            "断点续传 course_id=%s backend=%s 从 chunk %d 继续", course_id, backend, resume_from
        )

    # 提前置本 build 为 indexing：接口返回时 DB 已是 indexing，前端 loadCourses 立即拿到 →
    # hasIndexing 触发轮询 → 刷新页面也不丢"进行中"状态。worker(_run_indexing) 仍会
    # 再写一次，幂等兜底。
    build.status = "indexing"
    build.progress = 0
    build.error_msg = ""
    build.progress_msg = "准备中…" if resume_from == 0 else f"续传中（从第 {resume_from} 块）…"
    if resume_from == 0:
        build.chunks_done = 0
        build.chunks_total = 0
        build.token_estimate = 0
    build.updated_at = time.time()
    await db.flush()
    await invalidate_courses_cache()

    from core.arq_pool import get_arq_pool
    arq_pool = await get_arq_pool()
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="任务队列（ARQ/Redis）不可用，请稍后重试")
    await arq_pool.enqueue_job("run_indexing", kb.id, course_id, file_paths, resume_from, backend)
    logger.info(
        "ARQ 索引任务已入队 course_id=%s backend=%s files=%d", course_id, backend, len(file_paths)
    )

    return {
        "message": "索引任务已启动" if resume_from == 0 else f"续传任务已启动（从第 {resume_from} 个文本块）",
        "course_id": course_id,
        "backend": backend,
        "file_count": len(file_paths),
        "resume_from_chunk": resume_from,
    }


async def trigger_topic_graph_build(
    db: AsyncSession,
    kb: KnowledgeBase,
    course_id: str,
    force: bool = False,
) -> dict:
    """触发课程主题图构建（入队 ARQ build_course_topic_graph）。

    kb 由调用方按各自权限校验后传入（admin 用 _get_kb_or_404，teacher 用 _get_owned_kb）。
    三步：① 校验索引已 ready（有 chunks 的可用后端）——主题图依赖讲义切块审计 latest.json，
    未索引则抽取无原料；② skip-if-exists（表已有主题且非 force → 不入队，省 LLM、避锁竞争）；
    ③ 取 ARQ pool 入队（None → 503，与 trigger_kb_indexing 一致）。

    ready 口径与 ``aggregate_build_status`` / 检索层 _get_ready_backends 同——任一后端
    ``status=="ready" and chunks_total>0``，防空索引被误判就绪。
    """
    builds = (
        await db.execute(select(KbBuild).where(KbBuild.kb_id == kb.id))
    ).scalars().all()
    if not any(b.status == "ready" and (b.chunks_total or 0) > 0 for b in builds):
        raise HTTPException(
            status_code=409,
            detail="知识库尚未索引完成，请先建立索引（生成主题图依赖讲义切块审计）",
        )

    existing = (
        await db.execute(
            select(func.count()).select_from(CourseTopic).where(CourseTopic.course_id == course_id)
        )
    ).scalar_one()
    if existing and not force:
        return {
            "status": "already_exists",
            "course_id": course_id,
            "topics": int(existing),
            "message": "主题图已存在，force=true 可重建",
        }

    from core.arq_pool import get_arq_pool
    arq_pool = await get_arq_pool()
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="任务队列（ARQ/Redis）不可用，请稍后重试")
    await arq_pool.enqueue_job("build_course_topic_graph", course_id, force)
    logger.info("ARQ 主题图任务已入队 course_id=%s force=%s", course_id, force)
    return {
        "message": "主题图构建任务已启动",
        "course_id": course_id,
        "force": force,
    }


def _read_failed_sections(course_id: str) -> list[str]:
    """构建时落盘的抽取失败章节（读写收口在 builder.read_failed_sections——文件名
    契约单一真相源，mtime 缓存未变化的构建零重读）。缺失/解析失败降级 []。"""
    from core.memory.course_topic_builder import read_failed_sections

    return read_failed_sections(course_id)


async def get_topic_graph_payload(db: AsyncSession, course_id: str) -> dict:
    """读取课程主题图（主题/边列表）供 teacher/admin GET 端点核对。投影口径与 export_json 一致。

    权限校验由各端点自行完成（teacher 用 _get_owned_kb、admin 用 _get_kb_or_404）后调用本函数，
    故查询逻辑在此单一真相源，避免两端各写一份漂移。
    """
    from core.memory.course_topic_store import fetch_course_edges, fetch_course_topics

    topics = [
        {
            "topic_id": r.topic_id, "label": r.label, "definition": r.definition,
            "source_section": r.source_section, "order_idx": r.order_idx,
            "safety_weight": r.safety_weight,
        }
        for r in await fetch_course_topics(db, course_id)
    ]
    edges = [
        {
            "src": r.src_topic_id, "dst": r.dst_topic_id, "relation": r.relation,
            "confidence": r.confidence, "verified_by": r.verified_by,
        }
        for r in await fetch_course_edges(db, course_id)
    ]
    return {
        "course_id": course_id,
        "topic_count": len(topics),
        "edge_count": len(edges),
        "topics": topics,
        "edges": edges,
        "failed_sections": _read_failed_sections(course_id),
    }


async def get_student_topic_graph_payload(db: AsyncSession, course_id: str, user_id: str) -> dict:
    """学生视角课程主题图：主题 + 先修边 + 「我的掌握度」着色数据。

    - ``my.weak`` 与 risk 与门控**同口径**：weak = theta <= WEAK_MARGIN（C1 入闸判据），
      risk = 1 - sigmoid(theta - b_topic) 按折叠状态重算（不读存储列——旧行可能是
      delta 累加口径的历史值，见 ``proactive._read_mastery`` 同款注释）。
    - mastery 对齐走 ``course_topic_store.align_mastery_rows``（与门控同一单一真相源）。
    - 边投影带 relation（"prerequisite"=直接，"prerequisite_inferred"=闭包推断）。
    - 不投影 embedding / safety_weight（学生端无用且体积大）。
    - 未构建课程返回空列表而非 404——学生无法自行触发构建，前端显示空态即可。
    """
    from core.db.database import KnowledgeMastery
    from core.memory.course_topic_store import (
        align_mastery_rows, fetch_course_edges, fetch_course_topics, get_course_topic_map,
    )
    from core.memory.mastery_state import sigmoid
    from core.memory.proactive import WEAK_MARGIN

    topic_rows = await fetch_course_topics(db, course_id)
    edge_rows = await fetch_course_edges(db, course_id)
    mastery_rows = (
        await db.execute(
            select(KnowledgeMastery).where(
                KnowledgeMastery.user_id == user_id,
                KnowledgeMastery.course_id == course_id,
            )
        )
    ).scalars().all()

    # label_norm→topic_id 走 get_course_topic_map（归一口径单一真相源，免手动 import 私有 _norm_label）
    my_by_tid = {
        tid: r for tid, r in align_mastery_rows(
            await get_course_topic_map(course_id, db), mastery_rows,
        )
    }
    b_map = {r.topic_id: float(r.difficulty_b or 0.0) for r in topic_rows}

    topics = [
        {
            "topic_id": r.topic_id, "label": r.label, "definition": r.definition,
            "source_section": r.source_section, "order_idx": r.order_idx,
            "my": None if (m := my_by_tid.get(r.topic_id)) is None else {
                "theta": m.theta,
                "risk": round(1.0 - sigmoid((m.theta or 0.0) - b_map.get(r.topic_id, 0.0)), 4),
                "mastery": m.mastery,
                "observation_count": m.observation_count,
                "weak": bool((m.theta or 0.0) <= WEAK_MARGIN),
            },
        }
        for r in topic_rows
    ]
    edges = [
        {
            "src": r.src_topic_id, "dst": r.dst_topic_id, "relation": r.relation,
            "confidence": r.confidence,
        }
        for r in edge_rows
    ]
    return {
        "course_id": course_id,
        "topic_count": len(topics),
        "edge_count": len(edges),
        "topics": topics,
        "edges": edges,
    }
