"""Async database layer: SQLAlchemy 2.0 + asyncpg connection pool."""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from settings import get_settings
DATABASE_URL = get_settings().db.url.get_secret_value()

logger = logging.getLogger(__name__)

def _engine_kwargs() -> dict:
    """连接池参数：SQLite 单连接；Postgres 接入 settings 并按 worker 缩放。

    H-17/M-46：旧实现裸读 os.getenv("DB_POOL_SIZE")，既没接 settings，也不按
    backend_workers 缩放——4 worker × (10+15) = 100 连接，轻易打爆 Postgres
    max_connections（默认 100）。现从 settings.db.pool_size/max_overflow 取基准值，
    再按 settings.backend_workers 整除（下限 2），使每 worker 自适应：

        pool_size_per_worker   = max(2, pool_size   // workers)
        max_overflow_per_worker= max(1, max_overflow // workers)
        总连接上限 = workers × (pool_size_per_worker + max_overflow_per_worker)

    例：pool=10, overflow=15, workers=4 → 每 worker (2+3)=5 → 4 worker 共 20 连接。
    SQLite 不走池（StaticPool/NullPool 由方言默认），保持 check_same_thread=False。
    """
    if DATABASE_URL.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    s = get_settings()
    workers = max(1, s.backend_workers)
    pool_size = max(2, s.db.pool_size // workers)
    max_overflow = max(1, s.db.max_overflow // workers)
    return {
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }


engine = create_async_engine(DATABASE_URL, **_engine_kwargs())

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def _short_uuid(length: int = 12) -> str:
    return uuid.uuid4().hex[:length]


class User(Base):
    __tablename__ = "users"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    username = Column(String(32), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    display_name = Column(String(64), nullable=False, default="")
    knowledge_graph = Column(JSON, nullable=False, default=lambda: {"nodes": [], "edges": []})
    error_graph = Column(JSON, nullable=False, default=lambda: {"nodes": [], "edges": []})
    # OCC 版本号：graph_memory 整列 rewrite 的并发保护（条件 UPDATE + 冲突重试，宪法原则 5）
    graph_version = Column(Integer, nullable=False, default=0, server_default="0")
    role = Column(String(16), nullable=False, default="student")
    is_admin = Column(Boolean, nullable=False, default=False)
    created_at = Column(Float, nullable=False, default=time.time)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    course_id = Column(String(64), nullable=False)
    user_id = Column(String(32), nullable=False, default="")
    title = Column(String(256), nullable=False, default="新对话")
    mode = Column(String(32), nullable=False, default="chat")
    created_at = Column(Float, nullable=False, default=time.time)
    updated_at = Column(Float, nullable=False, default=time.time)

    # L2 摘要层字段
    summary = Column(Text, nullable=False, default="")  # 压缩后的摘要文本
    summary_up_to_msg_id = Column(String(32), nullable=True)  # 摘要覆盖到哪条消息（tie-break）
    summary_updated_at = Column(Float, nullable=True)  # 摘要最后更新时间
    # OCC 版本号：写回条件 UPDATE WHERE summary_version = old，多 worker 并发不互相覆盖
    summary_version = Column(Integer, nullable=False, default=0, server_default="0")
    # keyset 游标时间分量：配合 summary_up_to_msg_id 把增量区间改成 SQL 范围查询
    # （存量行 NULL，首次压缩走 msg_id 兼容路径后回填）
    summary_up_to_created_at = Column(Float, nullable=True)

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_sessions_course", "course_id", "updated_at"),
        Index("idx_sessions_user", "user_id", "updated_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(16))
    session_id = Column(String(32), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    # P1：写时落盘 course_id，课程级查询免 JOIN Session（宪法原则 3）
    course_id = Column(String(64), nullable=False, default="")
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False, default="")
    msg_type = Column(String(16), nullable=False, default="text")
    metadata_ = Column("metadata", Text, default="{}")
    created_at = Column(Float, nullable=False, default=time.time)

    session = relationship("Session", back_populates="messages")

    __table_args__ = (
        Index("idx_messages_session", "session_id", "created_at"),
        Index("idx_messages_course", "course_id", "created_at"),
    )


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    course_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(256), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    icon = Column(String(32), nullable=False, default="📘")
    system_prompt = Column(Text, nullable=False, default="")
    sort_order = Column(Integer, nullable=False, default=0)
    file_count = Column(Integer, nullable=False, default=0)
    created_at = Column(Float, nullable=False, default=time.time)
    updated_at = Column(Float, nullable=False, default=time.time)
    is_visible = Column(Boolean, nullable=False, default=True)
    owner_id = Column(String(32), ForeignKey("users.id"), nullable=True)
    join_code = Column(String(16), unique=True, nullable=True, index=True)
    # 索引后端：lightrag（默认，知识图谱，慢但支持多跳关系推理）| llamaindex_pg（pgvector
    # 快速向量检索，embedding 批调用分钟级建索引）。建库时选，per-KB 二选一。
    index_backend = Column(String(32), nullable=False, default="lightrag")
    files = relationship("KBFile", back_populates="kb", cascade="all, delete-orphan")
    # 每后端一条构建状态（LightRAG / pgvector 可并存）；见 KbBuild。cascade 随 KB 删除连级清理。
    # lazy="selectin"：任何 KB 查询自动连 builds 一次取回（async 禁止 lazy 触发），_kb_to_dict /
    # _kb_to_course 直接读 kb.builds 聚合，无需各调用点手写 selectinload。
    builds = relationship("KbBuild", back_populates="kb", cascade="all, delete-orphan", lazy="selectin")


class KBFile(Base):
    __tablename__ = "kb_files"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(16))
    kb_id = Column(String(32), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    original_name = Column(String(512), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False, default=0)
    # status: uploaded | indexed | error
    status = Column(String(32), nullable=False, default="uploaded")
    error_msg = Column(Text, nullable=False, default="")
    # 解析引擎（mineru_api/docling/mupdf），事后归因检索质量问题；空=未解析或解析层未启用
    parser_engine = Column(String(32), nullable=True)
    created_at = Column(Float, nullable=False, default=time.time)

    kb = relationship("KnowledgeBase", back_populates="files")   #属性 some_kb_file.kb → 指向对应的 KnowledgeBase

    __table_args__ = (
        Index("idx_kb_files_kb", "kb_id", "created_at"),
    )


class KbBuild(Base):
    """单个索引后端的一次构建状态（一课程可同时建 LightRAG + pgvector 两套）。

    knowledge_bases 1 行/课程挂源文件；本表每 (kb_id, backend) 一行、status/progress 独立。
    UNIQUE(kb_id, backend) 保证每后端至多一条。构建/暂停/终止/续传都按本表行驱动，与 KB 行
    解耦——KB 行上的旧 status/progress 列保留但不再被索引流程写入（改由本表聚合，见
    _kb_to_dict / _kb_to_course）。
    """

    __tablename__ = "kb_builds"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    kb_id = Column(String(32), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    backend = Column(String(32), nullable=False)  # lightrag | llamaindex_pg
    # status: pending | indexing | ready | error | paused
    status = Column(String(32), nullable=False, default="pending")
    progress = Column(Integer, nullable=False, default=0)          # 0‑100
    progress_msg = Column(Text, nullable=False, default="")        # 当前步骤描述
    chunks_done = Column(Integer, nullable=False, default=0)       # 已处理 chunk 数
    chunks_total = Column(Integer, nullable=False, default=0)      # 总 chunk 数
    token_estimate = Column(Integer, nullable=False, default=0)    # 估算 token 消耗
    error_msg = Column(Text, nullable=False, default="")
    updated_at = Column(Float, nullable=False, default=time.time)

    kb = relationship("KnowledgeBase", back_populates="builds")

    __table_args__ = (
        UniqueConstraint("kb_id", "backend", name="uq_kb_builds_kb_backend"),
    )


def aggregate_build_status(builds: list[KbBuild]) -> str:
    """多后端 kb_builds → 单一展示状态（KB 列表徽标 / _kb_to_course.kb_status 用）。

    优先级：indexing（有在建）> error > paused > ready（至少一个真有 chunks 的可用后端）> pending。
    ready 需 status==ready 且 chunks_total>0——与 _kb_to_course.ready_backends / 检索层
    _get_ready_backends 同口径，防空索引（0 chunk）被误判就绪导致徽章绿而按钮却禁用。
    """
    if not builds:
        return "pending"
    statuses = {b.status for b in builds}
    for s in ("indexing", "error", "paused"):
        if s in statuses:
            return s
    if any(b.status == "ready" and (b.chunks_total or 0) > 0 for b in builds):
        return "ready"
    return "pending"


class NotebookCategory(Base):
    """User-defined categories for question notebook entries."""

    __tablename__ = "notebook_categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_notebook_category_user_name"),)


class NotebookEntry(Base):
    """Persisted quiz question rows (question notebook)."""

    __tablename__ = "notebook_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    # P1：写时落盘 course_id，课程级查询免 JOIN Session（宪法原则 3；修 teacher.py:779 跨课 bug）
    course_id = Column(String(64), nullable=False, default="")
    session_id = Column(String(32), nullable=False, default="")
    session_title = Column(String(256), nullable=False, default="")
    question_id = Column(String(64), nullable=False, default="")
    question = Column(Text, nullable=False, default="")
    question_type = Column(String(32), nullable=False, default="")
    options = Column(JSON, nullable=True)
    correct_answer = Column(Text, nullable=False, default="")
    explanation = Column(Text, nullable=False, default="")
    difficulty = Column(String(32), nullable=False, default="")
    user_answer = Column(Text, nullable=False, default="")
    is_correct = Column(Boolean, nullable=False, default=False)
    bookmarked = Column(Boolean, nullable=False, default=False)
    followup_session_id = Column(String(64), nullable=False, default="")
    created_at = Column(Float, nullable=False, default=time.time)
    updated_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        UniqueConstraint("user_id", "session_id", "question_id", name="uq_notebook_entry_natural"),
        Index("idx_notebook_entries_user", "user_id", "updated_at"),
        Index("idx_notebook_entries_course", "course_id", "user_id"),
    )


class NotebookEntryCategory(Base):
    """Many-to-many: notebook entry ↔ category."""

    __tablename__ = "notebook_entry_categories"

    entry_id = Column(Integer, ForeignKey("notebook_entries.id", ondelete="CASCADE"), primary_key=True)
    category_id = Column(
        Integer, ForeignKey("notebook_categories.id", ondelete="CASCADE"), primary_key=True
    )


class TeacherInvite(Base):
    """One-time invite codes for teacher registration."""

    __tablename__ = "teacher_invites"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    code = Column(String(16), unique=True, nullable=False, index=True)
    created_by = Column(String(32), ForeignKey("users.id"), nullable=False)
    used_by = Column(String(32), ForeignKey("users.id"), nullable=True)
    expires_at = Column(Float, nullable=True)
    created_at = Column(Float, nullable=False, default=time.time)


class ApplicationStatus(str, Enum):
    """教师申请状态机。DB 存 String(32)，值：pending/approved/rejected。

    符合项目既有约定（CronJobState/TopicStatus 同构）：业务层用 Enum，
    DB 列存裸字符串值，避免 SQLAlchemy Enum 类型的 schema 耦合。
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TeacherApplication(Base):
    """教师准入申请（申请-审批流，与邀请码即时升级并存）。

    状态机 pending → approved/rejected（终态不可逆，审批接口显式守卫）。
    approved 时审批事务把 users.role 升为 teacher。
    """

    __tablename__ = "teacher_applications"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    reason = Column(Text, nullable=False, default="")
    status = Column(String(32), nullable=False, default=ApplicationStatus.PENDING.value)
    reviewed_by = Column(String(32), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(Float, nullable=True)
    review_note = Column(Text, nullable=False, default="")
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        # 不加 user_id 全局 unique——允许 rejected 后重新申请，保留审计轨迹。
        # 并发防重靠部分唯一索引 uq_teacher_app_pending_user（迁移里建）。
        # 复合索引最左前缀已覆盖单列查询，故 user_id/status 不再单独 index。
        Index("ix_teacher_app_user_status", "user_id", "status"),
        Index("ix_teacher_app_status_created", "status", "created_at"),
    )


class Enrollment(Base):
    """Student ↔ Course enrollment (managed by teacher / admin)."""

    __tablename__ = "enrollments"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    student_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    course_id = Column(String(64), nullable=False)
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        UniqueConstraint("student_id", "course_id", name="uq_enrollment_student_course"),
        Index("idx_enrollment_student", "student_id"),
        Index("idx_enrollment_course", "course_id"),
    )


class CourseSchedule(Base):
    """课程课表行（课程级，不带 user_id）。

    由教师通过 REST API 录入；query_timetable 工具 JOIN Enrollment 限定「我选的课」。
    weekday 用整数 1-7（1=周一 … 7=周日）便于排序与过滤；weeks 用字符串存原始表达
    （如 "1-16" / "1,3,5,…"），不做结构化解析（业务简单、避免过度设计）。
    """

    __tablename__ = "course_schedules"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    course_id = Column(String(64), nullable=False, index=True)
    weekday = Column(Integer, nullable=False)            # 1=周一 … 7=周日
    start_time = Column(String(16), nullable=False, default="")   # "HH:MM"
    end_time = Column(String(16), nullable=False, default="")     # "HH:MM"
    location = Column(String(128), nullable=False, default="")
    teacher_name = Column(String(64), nullable=False, default="")
    weeks = Column(String(64), nullable=False, default="")
    note = Column(Text, nullable=False, default="")
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        Index("idx_course_schedule_course_weekday", "course_id", "weekday"),
    )


class Grade(Base):
    """学生成绩记录（学生级，student_id 强绑定登录身份）。

    由教师通过 REST API 批量 upsert；query_grades 工具强制 WHERE student_id==注入的
    user_id（身份只走注入，schema 不暴露身份参数）。UniqueConstraint 保证同学生同课程
    同条目唯一，upsert 靠 (student_id, course_id, item_name) 复合键。
    """

    __tablename__ = "grades"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    student_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    course_id = Column(String(64), nullable=False)
    item_name = Column(String(128), nullable=False)     # 期中考试 / 作业1 / …
    score = Column(Float, nullable=False, default=0.0)
    full_score = Column(Float, nullable=False, default=100.0)
    graded_at = Column(Float, nullable=True)             # 判分时间戳；null=未判分
    comment = Column(Text, nullable=False, default="")
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        UniqueConstraint("student_id", "course_id", "item_name", name="uq_grade_student_course_item"),
        Index("idx_grade_student_course", "student_id", "course_id"),
    )


class UserSocialBinding(Base):
    """User ↔ social platform binding (QQ / Feishu)."""

    __tablename__ = "user_social_bindings"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    platform = Column(String(16), nullable=False)  # "qq" | "feishu"
    platform_user_id = Column(String(128), nullable=False)
    chat_id = Column(String(128), nullable=False, default="")
    display_name = Column(String(128), nullable=False, default="")
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        UniqueConstraint("platform", "platform_user_id", name="uq_social_binding"),
        Index("idx_social_binding_user", "user_id"),
    )


class BotNotification(Base):
    """Bot 定时提醒/通知触达 web 端的离线存储（按 user_id 隔离）。

    cron 到点触发 bot 生成提醒内容后落库，前端轮询拉取——补齐 web 渠道「定时通知」
    的最后一公里（IM 渠道 QQ/飞书仍走 channel.send 实时推送，不落此表）。
    """

    __tablename__ = "bot_notifications"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    bot_id = Column(String(64), nullable=False, default="")
    content = Column(Text, nullable=False, default="")
    read = Column(Boolean, nullable=False, default=False)
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (Index("idx_bot_notif_user", "user_id", "read", "created_at"),)


class UserMCPEnrollment(Base):
    """用户 ↔ MCP server 启用关系（server 进程系统级共享，此表仅控用户级可见性）。

    对标 Enrollment 范式。无记录 = 未配置 → 运行时默认全部可用（向后兼容）；
    有记录 = 用户已自定义 → 仅启用集合内的 server 工具对该用户可见。
    """

    __tablename__ = "user_mcp_enrollments"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    server_name = Column(String(64), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        UniqueConstraint("user_id", "server_name", name="uq_user_mcp_enrollment"),
        Index("idx_user_mcp_enrollment_user", "user_id"),
    )


class UserSearchConfig(Base):
    """用户级联网搜索配置覆盖（admin 配全局默认 data/search_config.json，此表存用户自定义）。

    无记录 = 用 admin 默认（+env）；有记录 = 字段级覆盖（user 非空字段优先于 admin/env）。
    user_id 唯一（一个用户一条）。对标 UserMCPEnrollment 的 per-user 范式。
    """

    __tablename__ = "user_search_configs"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    provider = Column(String(32), nullable=False, default="")  # 空=不覆盖，用 admin/env
    api_key = Column(String(256), nullable=False, default="")
    base_url = Column(String(512), nullable=False, default="")
    max_results = Column(Integer, nullable=False, default=0)  # 0=不覆盖
    proxy = Column(String(512), nullable=False, default="")
    created_at = Column(Float, nullable=False, default=time.time)


class UserLLMProvider(Base):
    """用户级 LLM provider 配置（多租户：每个用户可配自己的 API key + 模型，key 加密存储）。

    无记录 = 用平台默认（model_catalog.json active profile / .env）。
    有记录 = 覆盖（用户自配 provider 优先于平台）。
    user_id 唯一（一个用户一条活跃配置）。
    对标 UserSearchConfig / UserMCPEnrollment 的 per-user 范式。
    """

    __tablename__ = "user_llm_providers"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    binding = Column(String(32), nullable=False, default="")  # 供应商标识，如 "deepseek" / "dashscope"
    api_key_encrypted = Column(String(512), nullable=False, default="")  # Fernet 加密后的密文
    base_url = Column(String(512), nullable=False, default="")
    api_version = Column(String(32), nullable=False, default="")
    text_model = Column(String(64), nullable=False, default="")
    fast_model = Column(String(64), nullable=False, default="")  # 遗留死字段（无消费方），保留避免迁移破坏
    vision_model = Column(String(64), nullable=False, default="")
    # 视觉模型独立供应商（可异于对话供应商：对话走 deepseek，视觉走 dashscope/qwen-vl）
    vision_binding = Column(String(32), nullable=False, default="")
    vision_api_key_encrypted = Column(String(512), nullable=False, default="")  # Fernet 密文
    vision_base_url = Column(String(512), nullable=False, default="")
    updated_at = Column(Float, nullable=False, default=time.time)
    created_at = Column(Float, nullable=False, default=time.time)


class MemoryEpisode(Base):
    """L3 episodic 层：原始对话 turn，永不删除，同时充当巩固 outbox。

    取代 Redis buffer——status（pending/processing/done/dead）天然表达重试与积压，
    幂等靠 (session_id, turn_id) 唯一索引。巩固 job 从 pending 批量取，升格 semantic/mastery。
    新表：非生产库由 init_db 的 create_all 自动建；生产库走 alembic 020。
    """

    __tablename__ = "memory_episodes"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(16))
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    course_id = Column(String(64), nullable=False, default="")
    session_id = Column(String(32), nullable=False, default="")
    turn_id = Column(String(32), nullable=False, default="")
    mode = Column(String(32), nullable=False, default="")
    user_msg = Column(Text, nullable=False, default="")
    assistant_msg = Column(Text, nullable=False, default="")
    importance = Column(Float, nullable=False, default=0.0)
    segment_id = Column(String(32), nullable=True)  # 巩固 job 按话题切分后回填
    # pending=待巩固 / processing=巩固中 / done=已升格 / dead=永久失败
    status = Column(String(16), nullable=False, default="pending")
    created_at = Column(Float, nullable=False, default=time.time)
    consolidated_at = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("session_id", "turn_id", name="uq_episodes_session_turn"),
        Index("idx_episodes_outbox", "status", "created_at"),
        Index("idx_episodes_user", "user_id", "course_id", "created_at"),
    )


class KnowledgeMastery(Base):
    """L3 mastery 层：知识点掌握度（course 维度隔离），追加观测不覆盖，读时指数衰减。

    与 users.knowledge_graph 的区别：带 course_id（修跨课程污染）+ 追加式观测历史
   （evidence_episode_ids）+ 读时时间衰减（旧错误概念随时间软降权，不物理删除——
    反复性错误的诊断证据）。参考 TASA 遗忘曲线 × 知识追踪、Graphiti bi-temporal。
    Phase 4 期间与 users.knowledge_graph 双写（后者供教师 dashboard 读）。
    """

    __tablename__ = "knowledge_mastery"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(16))
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    course_id = Column(String(64), nullable=False, default="")
    kp_id = Column(String(64), nullable=False)  # LightRAG entity_id 或 label 哈希
    label = Column(String(128), nullable=False, default="")
    mastery = Column(Float, nullable=False, default=0.5)   # 0-1，越高越熟练
    risk = Column(Float, nullable=False, default=0.5)      # 0-1，越高越薄弱
    observation_count = Column(Integer, nullable=False, default=1)
    first_observed_at = Column(Float, nullable=False, default=time.time)
    last_observed_at = Column(Float, nullable=False, default=time.time)
    evidence_episode_ids = Column(JSON, nullable=False, default=list)  # 贡献过观测的 episode id

    __table_args__ = (
        UniqueConstraint("user_id", "course_id", "kp_id", name="uq_mastery_user_course_kp"),
        Index("idx_mastery_user_course", "user_id", "course_id"),
    )


class CourseTopic(Base):
    """课程结构层（配置/引用层，读多写少）：课程级主题 + 定义 + embedding，全班共享。

    建库时一次性离线抽取（见建库脚本），运行时只读、零生成式调用。为学情拼接门控
    （``core.memory.proactive.decide_stitch``）提供「问句→主题」最近邻坐标：学生问句
    embedding 与本表 embedding 取最近邻，低于阈值 = unknown = 不拼。topic_id 同时作为
    ``knowledge_mastery.kp_id`` 的对齐键（id-align），使门控只读 mastery 表即可回溯前置缺口。
    embedding 存 JSON（非 pgvector）——与 course_faq 同口径，SQLite 测试可跑、Python 算 cosine。
    新表：非生产库由 init_db 的 create_all 自动建；生产库走 alembic。
    """

    __tablename__ = "course_topic"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(16))
    course_id = Column(String(64), nullable=False, default="")
    topic_id = Column(String(64), nullable=False)  # 课程内主题标识，对齐 mastery.kp_id
    label = Column(String(128), nullable=False, default="")
    definition = Column(Text, nullable=False, default="")
    source_section = Column(String(255), nullable=False, default="")  # 来源章节栈
    order_idx = Column(Integer, nullable=False, default=0)  # 讲义顺序（抽前置边的候选约束）
    embedding = Column(JSON, nullable=True)  # list[float]，问句最近邻用；JSON 非 pgvector

    __table_args__ = (
        UniqueConstraint("course_id", "topic_id", name="uq_course_topic"),
        Index("idx_course_topic_course", "course_id"),
    )


class CourseTopicEdge(Base):
    """课程结构层：主题间先修边（prerequisite 唯一事实源），全班共享。

    建库时离线抽取：仅在「讲义顺序在前 + 同章或相邻章」候选对上判定，含糊或双向判定的
    候选一律丢弃（Goel 协议，防 LLM 乱连边）。门控沿 prerequisite 边回溯前置闭包，找未掌握的
    前置缺口。与 ``users.knowledge_graph`` 的区别：那是每生对话现编的图（仪表盘展示 +
    error ``repeated`` 来源），本表是全班一致、可审计的课程结构。verified_by 标教师已核对。
    """

    __tablename__ = "course_topic_edge"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(16))
    course_id = Column(String(64), nullable=False, default="")
    src_topic_id = Column(String(64), nullable=False)  # 前置（学 dst 前应先掌握 src）
    dst_topic_id = Column(String(64), nullable=False)
    relation = Column(String(32), nullable=False, default="prerequisite")  # 当前仅 prerequisite
    confidence = Column(Float, nullable=False, default=1.0)
    verified_by = Column(String(64), nullable=True)  # 教师审计标记，NULL=未审

    __table_args__ = (
        UniqueConstraint(
            "course_id", "src_topic_id", "dst_topic_id", "relation",
            name="uq_course_topic_edge",
        ),
        Index("idx_course_topic_edge_dst", "course_id", "dst_topic_id"),  # 回溯前置按 dst 查
    )


class LearningEvent(Base):
    """学情事件层（L0）：actor-verb-object 三元组 + 时间戳 + 上下文。

    借鉴 xAPI（actor/verb/object + timestamp + context）结构，但**不实现完整规范**
    （无 LRS / IRI 词表 / JSON-LD——无跨系统互操作需求）。承接三类信号：
    - ``asked``：学生提问（对话 turn 完成，供 course_faq 语义聚类）
    - ``answered``：学生答题（quiz 作答，供 rollup 正确率/掌握度）
    - ``feedback``：用户反馈（点赞点踩，Phase 4）
    读模型层（rollup / course_faq）由 ARQ cron 从本表增量聚合，展示层只读 rollup，
    不再每次现算（学情分析四模块设计 §目标架构）。事件只追加、不修改（append-only 事实）。
    """

    __tablename__ = "learning_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    actor_user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    course_id = Column(String(64), nullable=False, default="")
    verb = Column(String(32), nullable=False)  # asked | answered | feedback
    object_id = Column(String(128), nullable=False, default="")  # question_id / turn_id / 反馈目标
    object_text = Column(Text, nullable=False, default="")  # 问题文本(供FAQ聚类)/答题内容/反馈内容
    session_id = Column(String(32), nullable=False, default="")
    metadata_ = Column("metadata", JSON, nullable=True)  # verb 专属：is_correct/difficulty/mode/tools_used/score
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        # FAQ 聚类 cron：按课程取 asked 事件的时间窗口
        Index("idx_events_course_verb_time", "course_id", "verb", "created_at"),
        # 学生 rollup cron：按 (学生, 课程) 取事件时间窗口
        Index("idx_events_actor_course_time", "actor_user_id", "course_id", "created_at"),
    )


class CourseDailyRollup(Base):
    """学情读模型（L1）：每 (课程, 日) 一行的活跃度聚合，供活跃趋势/概览只读。

    ARQ cron 从 learning_events 删后重算（幂等，同 llm_usage_daily 口径）。day 用
    "YYYYMMDD" UTC，字典序==日期序。展示层只读本表，不扫事件明细。
    """

    __tablename__ = "course_daily_rollup"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(String(64), nullable=False)
    day = Column(String(8), nullable=False)                    # "YYYYMMDD" UTC
    active_students = Column(Integer, nullable=False, default=0)   # 当日有事件的去重学生数
    questions = Column(Integer, nullable=False, default=0)     # verb='asked' 计数
    answers = Column(Integer, nullable=False, default=0)       # verb='answered' 计数
    updated_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        UniqueConstraint("course_id", "day", name="uq_course_daily_rollup"),
        Index("idx_course_daily_rollup_day", "day"),
    )


class StudentCourseRollup(Base):
    """学情读模型（L1）：每 (学生, 课程) 一行的累计聚合，供教师学情统计/仪表盘只读。

    ARQ cron 从 Session/Message/NotebookEntry 删后重算（幂等）。mastery_avg/risk 等
    Phase 4 BKT 落地后填，此前为 NULL 占位（读侧遇 NULL 回退旧公式，避免空窗）。
    """

    __tablename__ = "student_course_rollup"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    course_id = Column(String(64), nullable=False)
    sessions = Column(Integer, nullable=False, default=0)
    messages = Column(Integer, nullable=False, default=0)
    quiz_total = Column(Integer, nullable=False, default=0)
    quiz_correct = Column(Integer, nullable=False, default=0)
    last_active_at = Column(Float, nullable=True)
    mastery_avg = Column(Float, nullable=True)   # Phase 4 BKT
    risk = Column(Float, nullable=True)          # Phase 4 BKT
    updated_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        UniqueConstraint("user_id", "course_id", name="uq_student_course_rollup"),
        Index("idx_student_course_rollup_course", "course_id"),
    )


class CourseFaq(Base):
    """高频问题读模型（L1）：语义聚类簇。

    ARQ cron 从 learning_events(verb=asked) 用 embedding + 阈值贪心聚类，删后重算
    （幂等）。取代 P1-c 的 Redis 精确匹配（"这题怎么算"/"这个怎么算" 永远不合）。
    embedding 存 JSON（非 pgvector）--SQLite 测试可跑、Python 算 cosine（避 func.left 同款
    PG 专有坑）。学情分析四模块设计 §模块一 p2-faq-cluster。
    """

    __tablename__ = "course_faq"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_id = Column(String(64), nullable=False, index=True)
    question = Column(Text, nullable=False, default="")   # 簇代表问题（种子提问原文）
    count = Column(Integer, nullable=False, default=0)    # 簇内提问次数
    embedding = Column(JSON, nullable=True)               # 簇代表 embedding（list[float]）
    last_asked_at = Column(Float, nullable=True)
    updated_at = Column(Float, nullable=False, default=time.time)


class ResearchCheckpoint(Base):
    """深度研究阶段级 checkpoint（plan 阶段 2B）：worker 重启后按 resume_research_id 重放。

    每个 research_id 一行。phase=最后到达的阶段；state_json 存阶段产物（refined_topic /
    sub_topics / DynamicTopicQueue.to_dict），被中断阶段整段重放（接受该阶段内部 LLM/检索成本
    重付一次，与 LangGraph「node 从头重跑」同语义）。进 ask_user 暂停前写 status=awaiting_user +
    pending_question_json（问题卡片 payload），重连后据此恢复同一份卡片。best-effort：读写失败
    只记日志，绝不阻塞正在跑的研究（见 core/research/checkpoint.py）。
    """

    __tablename__ = "research_checkpoints"

    research_id = Column(String(64), primary_key=True)
    user_id = Column(String(32), nullable=False, default="")
    course_id = Column(String(64), nullable=False, default="")
    topic = Column(Text, nullable=False, default="")
    phase = Column(String(32), nullable=False, default="")  # rephrase/decompose/researching/reporting
    state_json = Column(Text, nullable=False, default="")   # 阶段产物 JSON（含 queue.to_dict）
    pending_question_json = Column(Text, nullable=False, default="")  # ask_user 卡片 payload
    # status: running | awaiting_user | done | error
    status = Column(String(32), nullable=False, default="running")
    updated_at = Column(Float, nullable=False, default=time.time)


class LlmUsageRecord(Base):
    """LLM 用量明细（append-only 账单行）：每个 run_agent_loop 一行。

    设计要点：
    - **无 FK，纯字符串列**：账单类数据须抗级联删除（删用户不能删旧账），故 user_id/course_id
      不挂外键，同 research_checkpoints（022）。CourseUser 删了历史账仍可查。
    - **cost_usd 落库即定档**：按当时 model_pricing.json 价目表快照，日后改价目表不篡改历史账
      （对齐 Langfuse token-and-cost-tracking：成本在摄取时算好存，不在查询时重算）。
    - **mode + rounds**：cost-of-pass 分析用（arXiv:2504.13359）。只记总 token 记不出「quiz 一次
      多少钱 / research 是否轮次失控」这类降本决策；mode 区分 chat/quiz/deep_solve/deep_research。
    - cache_read_tokens 是 input_tokens 的子集（OTel GenAI 语义），不重复累加，仅用于算命中率。
    """

    __tablename__ = "llm_usage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(32), nullable=False, default="")
    course_id = Column(String(64), nullable=False, default="")
    session_id = Column(String(32), nullable=False, default="")
    turn_id = Column(String(64), nullable=False, default="")
    mode = Column(String(32), nullable=False, default="")   # chat/quiz/deep_solve/deep_research
    model = Column(String(128), nullable=False, default="")
    input_tokens = Column(Integer, nullable=False, default=0)          # 含 cache_read（OTel：子集不重复加）
    output_tokens = Column(Integer, nullable=False, default=0)
    cache_read_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)              # 摄取时价目表快照
    rounds = Column(Integer, nullable=False, default=0)                # cost-of-pass 分析用
    created_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        # 按人/按课的时间窗口扫描 + rollup 增量扫描 + 保留期清理
        Index("idx_usage_user_time", "user_id", "created_at"),
        Index("idx_usage_course_time", "course_id", "created_at"),
        Index("idx_usage_created", "created_at"),
    )


class LlmUsageDaily(Base):
    """LLM 用量日汇总（读模型）：ARQ cron 从明细删后重算，展示层只读本表不扫明细。

    唯一键 (day, user_id, course_id, model)：rollup 用「删今日+昨日聚合行后重插」天然幂等，
    避免维护 PG/SQLite 双方言 ON CONFLICT 语法。day 用 "YYYYMMDD" UTC，与 cost_quota._day_key
    同口径。BigInteger：日聚合跨人多课，防 Integer 溢出。
    """

    __tablename__ = "llm_usage_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    day = Column(String(8), nullable=False)                  # "YYYYMMDD" UTC
    user_id = Column(String(32), nullable=False, default="")
    course_id = Column(String(64), nullable=False, default="")
    model = Column(String(128), nullable=False, default="")
    input_tokens = Column(BigInteger, nullable=False, default=0)
    output_tokens = Column(BigInteger, nullable=False, default=0)
    cache_read_tokens = Column(BigInteger, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    call_count = Column(Integer, nullable=False, default=0)  # 明细行数 = loop 次数
    updated_at = Column(Float, nullable=False, default=time.time)

    __table_args__ = (
        UniqueConstraint("day", "user_id", "course_id", "model", name="uq_usage_daily"),
        Index("idx_usage_daily_day", "day"),
    )


async def _ensure_column(conn, table_name: str, column_name: str, ddl: str):
    """Add a column only if it does not already exist (dialect-aware)."""
    dialect = conn.dialect.name
    if dialect == "sqlite":
        rows = await conn.execute(text(f"PRAGMA table_info({table_name})"))
        cols = {row[1] for row in rows}
        if column_name in cols:
            return
    else:
        result = await conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = :tn AND column_name = :cn LIMIT 1"
            ),
            {"tn": table_name, "cn": column_name},
        )
        if result.first() is not None:
            return
    await conn.execute(text(ddl))

# Serialize DDL on PostgreSQL so multiple uvicorn workers cannot race on create_all
# (each worker runs lifespan startup; without a lock several processes may emit CREATE TABLE).
_PG_INIT_LOCK_KEY1 = 842_061_437
_PG_INIT_LOCK_KEY2 = 3_291_021


async def init_db():
    """Create all tables if they don't exist (idempotent).

    Production relies on ``alembic upgrade head``; skip create_all / column patches.
    """
    from settings import get_settings
    ENVIRONMENT = get_settings().environment

    if ENVIRONMENT == "production":
        logger.info("ENVIRONMENT=production: skipping create_all (use Alembic migrations)")
        return

    async with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            await conn.execute(
                text("SELECT pg_advisory_xact_lock(:k1, :k2)"),
                {"k1": _PG_INIT_LOCK_KEY1, "k2": _PG_INIT_LOCK_KEY2},
            )
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=True)
        )
        await _ensure_column(
            conn,
            "sessions",
            "mode",
            "ALTER TABLE sessions ADD COLUMN mode VARCHAR(32) NOT NULL DEFAULT 'chat'",
        )
        # L2 OCC 版本号 + keyset 游标时间分量（迁移 019）：补建存量非生产库，保证升级前
        # 已存在的 sessions 表也能用（create_all checkfirst 不会给已存在的表加列）。
        await _ensure_column(
            conn,
            "sessions",
            "summary_version",
            "ALTER TABLE sessions ADD COLUMN summary_version INTEGER NOT NULL DEFAULT 0",
        )
        await _ensure_column(
            conn,
            "sessions",
            "summary_up_to_created_at",
            "ALTER TABLE sessions ADD COLUMN summary_up_to_created_at FLOAT",
        )
        await _ensure_column(
            conn,
            "users",
            "is_admin",
            "ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT FALSE",
        )
        await _ensure_column(
            conn,
            "users",
            "knowledge_graph",
            "ALTER TABLE users ADD COLUMN knowledge_graph JSON NOT NULL DEFAULT '{\"nodes\":[],\"edges\":[]}'",
        )
        await _ensure_column(
            conn,
            "users",
            "error_graph",
            "ALTER TABLE users ADD COLUMN error_graph JSON NOT NULL DEFAULT '{\"nodes\":[],\"edges\":[]}'",
        )
        await _ensure_column(
            conn,
            "users",
            "graph_version",
            "ALTER TABLE users ADD COLUMN graph_version INTEGER NOT NULL DEFAULT 0",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "icon",
            "ALTER TABLE knowledge_bases ADD COLUMN icon VARCHAR(32) NOT NULL DEFAULT '📘'",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "system_prompt",
            "ALTER TABLE knowledge_bases ADD COLUMN system_prompt TEXT NOT NULL DEFAULT ''",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "sort_order",
            "ALTER TABLE knowledge_bases ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "is_visible",
            "ALTER TABLE knowledge_bases ADD COLUMN is_visible BOOLEAN NOT NULL DEFAULT TRUE",
        )
        await _ensure_column(
            conn,
            "users",
            "role",
            "ALTER TABLE users ADD COLUMN role VARCHAR(16) NOT NULL DEFAULT 'student'",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "owner_id",
            "ALTER TABLE knowledge_bases ADD COLUMN owner_id VARCHAR(32) REFERENCES users(id)",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "join_code",
            "ALTER TABLE knowledge_bases ADD COLUMN join_code VARCHAR(16)",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "index_backend",
            "ALTER TABLE knowledge_bases ADD COLUMN index_backend VARCHAR(32) NOT NULL DEFAULT 'lightrag'",
        )
        await _ensure_column(
            conn,
            "kb_files",
            "parser_engine",
            "ALTER TABLE kb_files ADD COLUMN parser_engine VARCHAR(32)",
        )
        # 唯一索引（_ensure_column 只管列，索引单独建）
        await conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS ix_kb_join_code
            ON knowledge_bases (join_code)
            WHERE join_code IS NOT NULL
        """))
        # Sync role ← is_admin for pre-existing admin users
        await conn.execute(
            text("UPDATE users SET role = 'admin' WHERE is_admin = TRUE AND role = 'student'")
        )

    # 将硬编码课程一次性 seed 进数据库（幂等：已存在则跳过）
    await _seed_builtin_courses()


async def _seed_builtin_courses() -> None:
    """把原 COURSE_PROMPTS 硬编码课程迁移进 knowledge_bases 表（幂等）。"""
    from core.db._builtin_courses import BUILTIN_COURSES  # 避免循环导入

    async with AsyncSessionLocal() as db:
        async with db.begin():
            for order, course in enumerate(BUILTIN_COURSES):
                exists = await db.execute(
                    text("SELECT 1 FROM knowledge_bases WHERE course_id = :cid LIMIT 1"),
                    {"cid": course["id"]},
                )
                if exists.first() is not None:
                    continue
                db.add(KnowledgeBase(
                    course_id=course["id"],
                    name=course["name"],
                    description=course.get("description", ""),
                    icon=course.get("icon", "📘"),
                    system_prompt=course.get("system_prompt", ""),
                    sort_order=order,
                ))


async def close_db():
    """Dispose of the connection pool."""
    await engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：yield 一个 async session。

    M-42 风险：FastAPI 依赖的 yield 后半段（commit/关闭）延迟到响应**完全发送**后才执行。
    对于 StreamingResponse / SSE（如 ``/chat``），这意味着 session 在整个流式输出期间
    （数十秒～分钟）一直挂着连接，多并发可打满连接池。

    正确做法：SSE / 长流端点**不要**用 ``Depends(get_db)``，改用 :func:`session_scope`
    在需要查询的片段内开闭，查完立即归还连接，流式阶段不持有任何 DB 连接。
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


class _SessionScope:
    """短生命周期 session 上下文（M-42）。

    用法::

        async with session_scope() as db:
            await check_course_access(db, course_id, user)

    进入即开连接，退出即 commit/关闭（异常回滚）。专供 SSE / 长流端点用，避免
    ``Depends(get_db)`` 把连接挂到流结束。语义对齐项目既有 ``async with
    AsyncSessionLocal()`` 模式，只是给出语义化入口 + 统一异常处理。
    """

    def __init__(self) -> None:
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> AsyncSession:
        self._session = AsyncSessionLocal()
        await self._session.__aenter__()
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is None:
            return
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            await self._session.__aexit__(exc_type, exc, tb)
            self._session = None


def session_scope() -> _SessionScope:
    """返回短生命周期 session 上下文管理器（见 :class:`_SessionScope`）。"""
    return _SessionScope()
