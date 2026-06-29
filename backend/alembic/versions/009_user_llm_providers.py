"""Add user_llm_providers table for user-level LLM provider configuration.

Revision ID: 009
Revises: 008
Create Date: 2026-06-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = sa.inspect(bind).get_table_names()

    if "user_llm_providers" not in existing:
        op.create_table(
            "user_llm_providers",
            sa.Column("id", sa.String(32), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(32),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("binding", sa.String(32), nullable=False, server_default=""),
            sa.Column("api_key_encrypted", sa.String(512), nullable=False, server_default=""),
            sa.Column("base_url", sa.String(512), nullable=False, server_default=""),
            sa.Column("api_version", sa.String(32), nullable=False, server_default=""),
            sa.Column("text_model", sa.String(64), nullable=False, server_default=""),
            sa.Column("fast_model", sa.String(64), nullable=False, server_default=""),
            sa.Column("vision_model", sa.String(64), nullable=False, server_default=""),
            sa.Column("updated_at", sa.Float, nullable=False),
            sa.Column("created_at", sa.Float, nullable=False),
        )
        op.create_unique_constraint(
            "uq_user_llm_provider_user", "user_llm_providers", ["user_id"]
        )


def downgrade() -> None:
    op.drop_table("user_llm_providers")
