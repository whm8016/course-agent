"""Add L2 session summary columns.

Revision ID: 010
Revises: 009
Create Date: 2026-06-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add summary column
    op.add_column(
        "sessions",
        sa.Column("summary", sa.Text(), nullable=False, server_default="")
    )

    # Add summary_up_to_msg_id column
    op.add_column(
        "sessions",
        sa.Column("summary_up_to_msg_id", sa.String(32), nullable=True)
    )

    # Add summary_updated_at column
    op.add_column(
        "sessions",
        sa.Column("summary_updated_at", sa.Float(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("sessions", "summary_updated_at")
    op.drop_column("sessions", "summary_up_to_msg_id")
    op.drop_column("sessions", "summary")
