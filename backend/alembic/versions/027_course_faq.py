"""Add course_faq table (高频问题语义聚类读模型).

学情分析四模块设计 §模块一 p2-faq-cluster：ARQ cron 从 learning_events(verb=asked)
用 embedding + 阈值贪心聚类，替代 P1-c 的 Redis 精确文本匹配（"这题怎么算"/"这个怎么算"
按原文永远不合）。embedding 存 JSON 非 pgvector--SQLite 测试可跑、Python 算 cosine。

新表无存量行，SQLite 与 PG 均安全。

Revision ID: 027
Revises: 026
Create Date: 2026-08-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "027"
down_revision: Union[str, None] = "026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "course_faq",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("course_id", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False, server_default=""),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("last_asked_at", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_index("ix_course_faq_course_id", "course_faq", ["course_id"])


def downgrade() -> None:
    op.drop_table("course_faq")
