"""Add course_id to messages + notebook_entries (P1 租主硬化).

按《表设计宪法》原则 3（多租户就地隔离）：课程级可查的行**写时落盘** course_id，
使 analytics/rollup 查询无需 JOIN Session 反查课程（已致一次跨租户 bug，teacher.py:779
注释：旧实现仅按 user_id 全局聚合，把他课答题算进本课）。

- messages.course_id + idx_messages_course(course_id, created_at)
- notebook_entries.course_id + idx_notebook_entries_course(course_id, user_id)

存量行从 sessions 关联子查询回填 course_id（PG/SQLite 双方言通用）；orphan session_id
（会话已删）回填为 ''。新写入由 add_message / upsert_notebook_entry 落盘。

NOT NULL + server_default=''：单步安全加列（无需两阶段可空性 Expand/Contract）。
BotNotification 不加--仅按 user_id 查的学生收件箱，非课程级可查（YAGNI）。

Revision ID: 026
Revises: 025
Create Date: 2026-08-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def upgrade() -> None:
    # messages.course_id
    if not _column_exists("messages", "course_id"):
        op.add_column(
            "messages",
            sa.Column("course_id", sa.String(length=64), nullable=False, server_default=""),
        )
        op.create_index("idx_messages_course", "messages", ["course_id", "created_at"])
    # notebook_entries.course_id
    if not _column_exists("notebook_entries", "course_id"):
        op.add_column(
            "notebook_entries",
            sa.Column("course_id", sa.String(length=64), nullable=False, server_default=""),
        )
        op.create_index("idx_notebook_entries_course", "notebook_entries", ["course_id", "user_id"])
    # 回填存量行：从 sessions 关联子查询取 course_id（PG/SQLite 通用，非 UPDATE...FROM）。
    # 加列后所有行 course_id='' -> 全量回填；WHERE course_id='' 兼容部分已回填的重跑。
    op.execute(sa.text(
        "UPDATE messages SET course_id = (SELECT s.course_id FROM sessions s "
        "WHERE s.id = messages.session_id) WHERE course_id = ''"
    ))
    op.execute(sa.text(
        "UPDATE notebook_entries SET course_id = (SELECT s.course_id FROM sessions s "
        "WHERE s.id = notebook_entries.session_id) WHERE course_id = ''"
    ))


def downgrade() -> None:
    if _column_exists("messages", "course_id"):
        op.drop_index("idx_messages_course", table_name="messages")
        op.drop_column("messages", "course_id")
    if _column_exists("notebook_entries", "course_id"):
        op.drop_index("idx_notebook_entries_course", table_name="notebook_entries")
        op.drop_column("notebook_entries", "course_id")
