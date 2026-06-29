"""Async database layer: SQLAlchemy 2.0 + asyncpg connection pool."""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncGenerator

from sqlalchemy import (
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

from config import DATABASE_URL

logger = logging.getLogger(__name__)

def _engine_kwargs() -> dict:
    if DATABASE_URL.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {
        "pool_size": int(__import__("os").getenv("DB_POOL_SIZE", "10")),
        "max_overflow": int(__import__("os").getenv("DB_MAX_OVERFLOW", "15")),
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
    summary_memory = Column(Text, nullable=False, default="")
    profile_memory = Column(Text, nullable=False, default="{}")
    scope_memory = Column(Text, nullable=False, default="")
    preferences_memory = Column(Text, nullable=False, default="")
    knowledge_graph = Column(JSON, nullable=False, default=lambda: {"nodes": [], "edges": []})
    error_graph = Column(JSON, nullable=False, default=lambda: {"nodes": [], "edges": []})
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
    summary_up_to_msg_id = Column(String(32), nullable=True)  # 摘要覆盖到哪条消息
    summary_updated_at = Column(Float, nullable=True)  # 摘要最后更新时间

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_sessions_course", "course_id", "updated_at"),
        Index("idx_sessions_user", "user_id", "updated_at"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(16))
    session_id = Column(String(32), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False, default="")
    msg_type = Column(String(16), nullable=False, default="text")
    metadata_ = Column("metadata", Text, default="{}")
    created_at = Column(Float, nullable=False, default=time.time)

    session = relationship("Session", back_populates="messages")

    __table_args__ = (
        Index("idx_messages_session", "session_id", "created_at"),
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
    # status: pending | indexing | ready | error
    status = Column(String(32), nullable=False, default="pending")
    file_count = Column(Integer, nullable=False, default=0)
    error_msg = Column(Text, nullable=False, default="")
    # 索引进度相关字段
    progress = Column(Integer, nullable=False, default=0)          # 0‑100
    progress_msg = Column(Text, nullable=False, default="")        # 当前步骤描述
    chunks_done = Column(Integer, nullable=False, default=0)       # 已处理 chunk 数
    chunks_total = Column(Integer, nullable=False, default=0)      # 总 chunk 数
    token_estimate = Column(Integer, nullable=False, default=0)    # 估算 token 消耗
    created_at = Column(Float, nullable=False, default=time.time)
    updated_at = Column(Float, nullable=False, default=time.time)
    is_visible = Column(Boolean, nullable=False, default=True)
    owner_id = Column(String(32), ForeignKey("users.id"), nullable=True)
    join_code = Column(String(16), unique=True, nullable=True, index=True)
    files = relationship("KBFile", back_populates="kb", cascade="all, delete-orphan")


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
    created_at = Column(Float, nullable=False, default=time.time)

    kb = relationship("KnowledgeBase", back_populates="files")   #属性 some_kb_file.kb → 指向对应的 KnowledgeBase

    __table_args__ = (
        Index("idx_kb_files_kb", "kb_id", "created_at"),
    )


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
    fast_model = Column(String(64), nullable=False, default="")
    vision_model = Column(String(64), nullable=False, default="")
    updated_at = Column(Float, nullable=False, default=time.time)
    created_at = Column(Float, nullable=False, default=time.time)


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
    from config import ENVIRONMENT

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
        await _ensure_column(
            conn,
            "users",
            "summary_memory",
            "ALTER TABLE users ADD COLUMN summary_memory TEXT NOT NULL DEFAULT ''",
        )
        await _ensure_column(
            conn,
            "users",
            "profile_memory",
            "ALTER TABLE users ADD COLUMN profile_memory TEXT NOT NULL DEFAULT '{}'",
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
            "scope_memory",
            "ALTER TABLE users ADD COLUMN scope_memory TEXT NOT NULL DEFAULT ''",
        )
        await _ensure_column(
            conn,
            "users",
            "preferences_memory",
            "ALTER TABLE users ADD COLUMN preferences_memory TEXT NOT NULL DEFAULT ''",
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
        # 知识库进度字段（向已有表追加）
        await _ensure_column(
            conn,
            "knowledge_bases",
            "progress",
            "ALTER TABLE knowledge_bases ADD COLUMN progress INTEGER NOT NULL DEFAULT 0",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "progress_msg",
            "ALTER TABLE knowledge_bases ADD COLUMN progress_msg TEXT NOT NULL DEFAULT ''",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "chunks_done",
            "ALTER TABLE knowledge_bases ADD COLUMN chunks_done INTEGER NOT NULL DEFAULT 0",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "chunks_total",
            "ALTER TABLE knowledge_bases ADD COLUMN chunks_total INTEGER NOT NULL DEFAULT 0",
        )
        await _ensure_column(
            conn,
            "knowledge_bases",
            "token_estimate",
            "ALTER TABLE knowledge_bases ADD COLUMN token_estimate INTEGER NOT NULL DEFAULT 0",
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
                    status="pending",
                ))


async def close_db():
    """Dispose of the connection pool."""
    await engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

