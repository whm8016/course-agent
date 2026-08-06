"""Add memory_episodes table (L3 episodic layer).

L3 记忆重构 Phase 2：新建 episodic 原始层，取代 Redis buffer（flush_manager）。
每轮对话完成后写一行，永不删除（保留 provenance，提取逻辑改进后可重算历史）；
status（pending/processing/done/dead）充当巩固 outbox，(session_id, turn_id) 唯一保证幂等。

索引：
- uq_episodes_session_turn(session_id, turn_id) — 幂等：同一 turn 重放只写一次
- idx_episodes_outbox(status, created_at)        — 巩固 job / safety net 扫 pending/processing
- idx_episodes_user(user_id, course_id, created_at) — 按 user+course 取待巩固段

新表（无存量行），SQLite 与 PG 均安全。

Revision ID: 020
Revises: 019
Create Date: 2026-08-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_episodes",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("course_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("session_id", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("turn_id", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("mode", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("user_msg", sa.Text(), nullable=False, server_default=""),
        sa.Column("assistant_msg", sa.Text(), nullable=False, server_default=""),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0"),
        sa.Column("segment_id", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("consolidated_at", sa.Float(), nullable=True),
        # 唯一约束必须内联进 CREATE TABLE：SQLite 不支持 ALTER TABLE ADD CONSTRAINT
        # （单独 op.create_unique_constraint 需 batch mode）。内联在 SQLite/PG 都生效。
        sa.UniqueConstraint("session_id", "turn_id", name="uq_episodes_session_turn"),
    )
    op.create_index("idx_episodes_outbox", "memory_episodes", ["status", "created_at"])
    op.create_index(
        "idx_episodes_user", "memory_episodes", ["user_id", "course_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_table("memory_episodes")
