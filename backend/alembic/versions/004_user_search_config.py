"""Add user_search_configs table.

Revision ID: 004
Revises: 003
Create Date: 2026-06-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = sa.inspect(bind).get_table_names()

    if "user_search_configs" not in existing:
        op.create_table(
            "user_search_configs",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("provider", sa.String(32), nullable=False, server_default=""),
            sa.Column("api_key", sa.String(256), nullable=False, server_default=""),
            sa.Column("base_url", sa.String(512), nullable=False, server_default=""),
            sa.Column("max_results", sa.Integer, nullable=False, server_default="0"),
            sa.Column("proxy", sa.String(512), nullable=False, server_default=""),
            sa.Column("created_at", sa.Float, nullable=False),
        )
        op.create_unique_constraint(
            "uq_user_search_config_user", "user_search_configs", ["user_id"]
        )


def downgrade() -> None:
    op.drop_table("user_search_configs")
