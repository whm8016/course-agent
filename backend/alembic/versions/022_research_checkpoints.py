"""Add research_checkpoints table (deep_research 阶段级 checkpoint).

深度研究 worker 重启恢复（plan 阶段 2B）：每个 research_id 一行，记录最后到达的 phase + 阶段产物
state_json；ask_user 暂停时写 status=awaiting_user + pending_question_json，重连后恢复卡片。
新表（无存量行、无 FK），SQLite 与 PG 均安全。

Revision ID: 022
Revises: 021
Create Date: 2026-08-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "022"
down_revision: Union[str, None] = "021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "research_checkpoints",
        sa.Column("research_id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(32), nullable=False, server_default=""),
        sa.Column("course_id", sa.String(64), nullable=False, server_default=""),
        sa.Column("topic", sa.Text, nullable=False, server_default=""),
        sa.Column("phase", sa.String(32), nullable=False, server_default=""),
        sa.Column("state_json", sa.Text, nullable=False, server_default=""),
        sa.Column("pending_question_json", sa.Text, nullable=False, server_default=""),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("updated_at", sa.Float, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("research_checkpoints")
