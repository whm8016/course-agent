"""Add role system, teacher invites, enrollments, KB owner_id.

Revision ID: 002
Revises: 001
Create Date: 2026-05-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    cols = [c["name"] for c in sa.inspect(bind).get_columns(table)]
    return column in cols


def _table_exists(table: str) -> bool:
    bind = op.get_bind()     #获取当前数据库连接
    return table in sa.inspect(bind).get_table_names() 
    #inspect = 反射（reflection）：不看你 Python 里的 User 类，而是去问数据库：现在库里实际长什么样。


def upgrade() -> None:
    # -- users.role --
    if not _column_exists("users", "role"):
        op.add_column("users", sa.Column("role", sa.String(16), nullable=False, server_default="student"))
        # 001 未建 is_admin；仅当列已存在时再同步（旧库由 init_db 补过 is_admin）
        if _column_exists("users", "is_admin"):
            op.execute("UPDATE users SET role = 'admin' WHERE is_admin = TRUE")

    # -- knowledge_bases.owner_id --
    if _table_exists("knowledge_bases") and not _column_exists("knowledge_bases", "owner_id"):
        op.add_column(
            "knowledge_bases",
            sa.Column("owner_id", sa.String(32), sa.ForeignKey("users.id"), nullable=True),
        )

    # -- teacher_invites --
    if not _table_exists("teacher_invites"):
        op.create_table(
            "teacher_invites",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column("code", sa.String(16), unique=True, nullable=False),
            sa.Column("created_by", sa.String(32), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("used_by", sa.String(32), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("expires_at", sa.Float, nullable=True),
            sa.Column("created_at", sa.Float, nullable=False),
        )
        op.create_index("ix_teacher_invites_code", "teacher_invites", ["code"], unique=True)

    # -- enrollments --
    if not _table_exists("enrollments"):
        op.create_table(
            "enrollments",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "student_id",
                sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("course_id", sa.String(64), nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
        )
        op.create_index("idx_enrollment_student", "enrollments", ["student_id"])
        op.create_index("idx_enrollment_course", "enrollments", ["course_id"])
        op.create_unique_constraint("uq_enrollment_student_course", "enrollments", ["student_id", "course_id"])


def downgrade() -> None:
    op.drop_table("enrollments")
    op.drop_table("teacher_invites")
    if _column_exists("knowledge_bases", "owner_id"):
        op.drop_column("knowledge_bases", "owner_id")
    if _column_exists("users", "role"):
        op.drop_column("users", "role")
