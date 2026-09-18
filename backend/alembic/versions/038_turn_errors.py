"""Add turn_errors table (错误处理与可观测性重整，第 32 张表).

失败 turn 的 append-only 运维流水：turn_runtime._run_turn finally 在 turn_error 非空时
best-effort 落一行（信封 code/ref/exc_type/exc_head），admin 端点
GET /admin/errors/recent 按学生报障 ref 精确查询。与 messages（学生叙事流）两概念两
真相源——「最近失败 turn」不退化为 LIKE 全表扫。无 FK 裸字符串（用户/课程删除后须
存活），同 llm_usage_records（024）。新表无存量行，SQLite 与 PG 均安全。

Revision ID: 038
Revises: 037
Create Date: 2026-09-18
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "038"
down_revision: Union[str, None] = "037"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "turn_errors",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("turn_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("ref", sa.String(length=16), nullable=False, server_default=""),
        sa.Column("session_id", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("user_id", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("course_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("mode", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("code", sa.String(length=32), nullable=False, server_default="internal"),
        sa.Column("message", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("exc_type", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("exc_head", sa.String(length=256), nullable=False, server_default=""),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_turn_errors_course_time", "turn_errors", ["course_id", "created_at"])
    op.create_index("idx_turn_errors_user_time", "turn_errors", ["user_id", "created_at"])
    op.create_index("idx_turn_errors_ref", "turn_errors", ["ref"])
    op.create_index("idx_turn_errors_created", "turn_errors", ["created_at"])


def downgrade() -> None:
    op.drop_table("turn_errors")
