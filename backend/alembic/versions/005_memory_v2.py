"""Add scope_memory and preferences_memory to users.

Revision ID: 005
Revises: 004
Create Date: 2026-06-26
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in sa.inspect(bind).get_columns("users")}
    if "scope_memory" not in cols:
        op.add_column("users", sa.Column("scope_memory", sa.Text, nullable=False, server_default=""))
    if "preferences_memory" not in cols:
        op.add_column("users", sa.Column("preferences_memory", sa.Text, nullable=False, server_default=""))


def downgrade() -> None:
    op.drop_column("users", "preferences_memory")
    op.drop_column("users", "scope_memory")