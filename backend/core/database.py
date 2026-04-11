"""Async database layer: SQLAlchemy 2.0 + asyncpg connection pool."""
from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
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
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
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
    created_at = Column(Float, nullable=False, default=time.time)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(32), primary_key=True, default=lambda: _short_uuid(12))
    course_id = Column(String(64), nullable=False)
    user_id = Column(String(32), nullable=False, default="")
    title = Column(String(256), nullable=False, default="新对话")
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


async def init_db():
    """Create all tables if they don't exist (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


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
