"""Memory v3 — structured document store + snapshot + consolidator meta.

Revision ID: 006
Revises: 005
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = sa.inspect(bind).get_table_names()

    # 替代 DeepTutor 的 L2/*.md + L3/*.md 文件
    if "memory_docs" not in existing:
        op.create_table(
            "memory_docs",
            sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("layer", sa.String(4), nullable=False),      # 'L2' | 'L3'
            sa.Column("doc_key", sa.String(32), nullable=False),    # 'chat'|'quiz'|'profile'|...
            sa.Column("content", sa.Text, nullable=False, server_default=""),
            sa.Column("updated_at", sa.Float, nullable=False, server_default="0"),
            sa.PrimaryKeyConstraint("user_id", "layer", "doc_key"),
        )

    # 替代 DeepTutor 的 snapshot/<surface>/state.json + changes.jsonl
    if "memory_snapshot_state" not in existing:
        op.create_table(
            "memory_snapshot_state",
            sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("surface", sa.String(32), nullable=False),
            sa.Column("fingerprints", sa.JSON, nullable=False, server_default="{}"),
            sa.Column("labels", sa.JSON, nullable=False, server_default="{}"),
            sa.Column("last_refresh", sa.String(64), nullable=True),
            sa.PrimaryKeyConstraint("user_id", "surface"),
        )

    # 替代 DeepTutor 的 L2/*.meta.json + L3/*.meta.json
    if "memory_consolidator_meta" not in existing:
        op.create_table(
            "memory_consolidator_meta",
            sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("layer", sa.String(4), nullable=False),
            sa.Column("doc_key", sa.String(32), nullable=False),
            sa.Column("last_update_at", sa.String(64), nullable=True),
            sa.Column("seen_ids", sa.JSON, nullable=False, server_default="{}"),
            sa.PrimaryKeyConstraint("user_id", "layer", "doc_key"),
        )


def downgrade() -> None:
    op.drop_table("memory_consolidator_meta")
    op.drop_table("memory_snapshot_state")
    op.drop_table("memory_docs")