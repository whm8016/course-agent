"""Add learning_events table (学情事件层 L0).

学情分析四模块设计 §第二期：事件 → 读模型 → 展示管道的源头。借鉴 xAPI
actor-verb-object + timestamp + context 结构（不实现完整规范）。承接 asked（对话）、
answered（答题）、feedback（反馈，Phase 4）三类信号；读模型层（rollup / course_faq）
由 ARQ cron 从本表增量聚合。

索引按 cron 访问模式：
- idx_events_course_verb_time(course_id, verb, created_at) — FAQ 聚类按课程取 asked 窗口
- idx_events_actor_course_time(actor_user_id, course_id, created_at) — 学生 rollup 按(学生,课程)取窗口

新表（无存量行），SQLite 与 PG 均安全。

Revision ID: 023
Revises: 022
Create Date: 2026-08-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "learning_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "actor_user_id",
            sa.String(length=32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("course_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("verb", sa.String(length=32), nullable=False),
        sa.Column("object_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("object_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("session_id", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index(
        "idx_events_course_verb_time",
        "learning_events",
        ["course_id", "verb", "created_at"],
    )
    op.create_index(
        "idx_events_actor_course_time",
        "learning_events",
        ["actor_user_id", "course_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("learning_events")
