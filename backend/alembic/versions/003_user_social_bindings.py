"""Add user_social_bindings table.

Revision ID: 003
Revises: 002
Create Date: 2026-06-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = sa.inspect(bind).get_table_names()

    if "user_social_bindings" not in existing:
        op.create_table(
            "user_social_bindings",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("platform", sa.String(16), nullable=False),
            sa.Column("platform_user_id", sa.String(128), nullable=False),
            sa.Column("chat_id", sa.String(128), nullable=False, server_default=""),
            sa.Column("display_name", sa.String(128), nullable=False, server_default=""),
            sa.Column("created_at", sa.Float, nullable=False),
        )
        op.create_unique_constraint(
            "uq_social_binding", "user_social_bindings", ["platform", "platform_user_id"]
        )
        op.create_index(
            "idx_social_binding_user", "user_social_bindings", ["user_id"]
        )


def downgrade() -> None:
    op.drop_table("user_social_bindings")
