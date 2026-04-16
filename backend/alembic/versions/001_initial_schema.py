"""Initial schema: users, sessions, messages.

Revision ID: 001
Revises: None
Create Date: 2026-04-10

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = sa.inspect(bind).get_table_names()

    if "users" not in existing:
        op.create_table(
            "users",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column("username", sa.String(32), unique=True, nullable=False),
            sa.Column("password_hash", sa.String(256), nullable=False),
            sa.Column("display_name", sa.String(64), nullable=False, server_default=""),
            sa.Column("created_at", sa.Float, nullable=False),
        )

    if "sessions" not in existing:
        op.create_table(
            "sessions",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column("course_id", sa.String(64), nullable=False),
            sa.Column("user_id", sa.String(32), nullable=False, server_default=""),
            sa.Column("title", sa.String(256), nullable=False, server_default="新对话"),
            sa.Column("created_at", sa.Float, nullable=False),
            sa.Column("updated_at", sa.Float, nullable=False),
        )
        op.create_index("idx_sessions_course", "sessions", ["course_id", "updated_at"])
        op.create_index("idx_sessions_user", "sessions", ["user_id", "updated_at"])

    if "messages" not in existing:
        op.create_table(
            "messages",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "session_id",
                sa.String(32),
                sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("role", sa.String(16), nullable=False),
            sa.Column("content", sa.Text, nullable=False, server_default=""),
            sa.Column("msg_type", sa.String(16), nullable=False, server_default="text"),
            sa.Column("metadata", sa.Text, server_default="{}"),
            sa.Column("created_at", sa.Float, nullable=False),
        )
        op.create_index("idx_messages_session", "messages", ["session_id", "created_at"])


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("users")
