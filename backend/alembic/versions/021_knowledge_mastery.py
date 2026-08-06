"""Add knowledge_mastery table (L3 mastery layer).

L3 记忆重构 Phase 4a：知识点掌握度层。与 users.knowledge_graph 的区别——
带 course_id（修跨课程污染）+ 追加式观测（evidence_episode_ids）+ 读时时间衰减。
Phase 4 期间与 users.knowledge_graph 双写（后者供教师 dashboard 读）。

约束/索引：
- uq_mastery_user_course_kp(user_id, course_id, kp_id) — 每用户每课程每知识点一行（追加观测靠 UPDATE，不新增行）
- idx_mastery_user_course(user_id, course_id) — 读时按 user+course 取掌握度

新表（无存量行），SQLite 与 PG 均安全（唯一约束内联进 CREATE TABLE）。

Revision ID: 021
Revises: 020
Create Date: 2026-08-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "knowledge_mastery",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(length=32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("course_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("kp_id", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("mastery", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("risk", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("observation_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("first_observed_at", sa.Float(), nullable=False),
        sa.Column("last_observed_at", sa.Float(), nullable=False),
        sa.Column("evidence_episode_ids", sa.JSON(), nullable=False),
        # 唯一约束内联（SQLite 不支持 ALTER ADD CONSTRAINT）；每 user+course+kp 一行
        sa.UniqueConstraint(
            "user_id", "course_id", "kp_id", name="uq_mastery_user_course_kp"
        ),
    )
    op.create_index("idx_mastery_user_course", "knowledge_mastery", ["user_id", "course_id"])


def downgrade() -> None:
    op.drop_table("knowledge_mastery")
