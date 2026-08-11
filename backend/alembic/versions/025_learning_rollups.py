"""Add learning read-model tables: course_daily_rollup + student_course_rollup.

学情分析四模块设计 §第二期 p2-rollup：两张读模型表，ARQ cron 增量聚合（删后重算，
幂等，同 020/llm_usage_daily 口径）。展示层（教师学情统计/仪表盘）切读这两张表，
不再每次现算 8 路串行查询。

- course_daily_rollup(course_id, day)：每日课程活跃度（去重学生数 / asked / answered），
  从 learning_events 重算。
- student_course_rollup(user_id, course_id)：每学生每课程的累计（sessions/messages/
  quiz_total/quiz_correct/last_active_at），从 Session/Message/NotebookEntry 重算；
  mastery_avg/risk 留给 Phase 4 BKT，先建列占位（NULL）。

唯一约束内联（SQLite 不支持 ALTER ADD CONSTRAINT）。新表无存量行，SQLite 与 PG 均安全。

Revision ID: 025
Revises: 024
Create Date: 2026-08-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "025"
down_revision: Union[str, None] = "024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "course_daily_rollup",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("course_id", sa.String(length=64), nullable=False),
        sa.Column("day", sa.String(length=8), nullable=False),
        sa.Column("active_students", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("questions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("answers", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.UniqueConstraint("course_id", "day", name="uq_course_daily_rollup"),
    )
    op.create_index("idx_course_daily_rollup_day", "course_daily_rollup", ["day"])

    op.create_table(
        "student_course_rollup",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.String(length=32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("course_id", sa.String(length=64), nullable=False),
        sa.Column("sessions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("messages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quiz_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quiz_correct", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_active_at", sa.Float(), nullable=True),
        sa.Column("mastery_avg", sa.Float(), nullable=True),
        sa.Column("risk", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.UniqueConstraint("user_id", "course_id", name="uq_student_course_rollup"),
    )
    op.create_index("idx_student_course_rollup_course", "student_course_rollup", ["course_id"])


def downgrade() -> None:
    op.drop_table("student_course_rollup")
    op.drop_table("course_daily_rollup")
