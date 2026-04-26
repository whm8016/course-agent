"""Async database layer: SQLAlchemy 2.0 + asyncpg connection pool."""
from __future__ import annotations

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
    String,
    Text,
    text,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from config import DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    pool_size=int(__import__("os").getenv("DB_POOL_SIZE", "10")),
    max_overflow=int(__import__("os").getenv("DB_MAX_OVERFLOW", "15")),
    pool_pre_ping=True,
    pool_recycle=1800,
)

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


async def _ensure_column(conn, table_name: str, column_name: str, ddl: str):
    exists_sql = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = :table_name AND column_name = :column_name
        LIMIT 1
        """
    )
    result = await conn.execute(
        exists_sql, {"table_name": table_name, "column_name": column_name}
    )
    if result.first() is None:
        await conn.execute(text(ddl))
# text(...)（SQLAlchemy）
#把多行字符串包成“可执行的 SQL 文本对象”，便于 conn.execute 使用。

# Serialize DDL on PostgreSQL so multiple uvicorn workers cannot race on create_all
# (each worker runs lifespan startup; without a lock several processes may emit CREATE TABLE).
_PG_INIT_LOCK_KEY1 = 842_061_437
_PG_INIT_LOCK_KEY2 = 3_291_021


async def init_db():
    """Create all tables if they don't exist (idempotent)."""
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

    # 将硬编码课程一次性 seed 进数据库（幂等：已存在则跳过）
    await _seed_builtin_courses()


async def _seed_builtin_courses() -> None:
    """把原 COURSE_PROMPTS 硬编码课程迁移进 knowledge_bases 表（幂等）。"""
    from core._builtin_courses import BUILTIN_COURSES  # 避免循环导入

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
