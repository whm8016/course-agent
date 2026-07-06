"""Add independent vision provider columns to user_llm_providers.

对话模型与视觉模型可走不同供应商（对话 deepseek，视觉 dashscope/qwen-vl）。

Revision ID: 011
Revises: 010
Create Date: 2026-07-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if "user_llm_providers" not in sa.inspect(bind).get_table_names():
        return  # 表尚未创建（全新部署由建表迁移负责）；幂等跳过
    existing = {c["name"] for c in sa.inspect(bind).get_columns("user_llm_providers")}
    if "vision_binding" not in existing:
        op.add_column(
            "user_llm_providers",
            sa.Column("vision_binding", sa.String(32), nullable=False, server_default=""),
        )
    if "vision_api_key_encrypted" not in existing:
        op.add_column(
            "user_llm_providers",
            sa.Column("vision_api_key_encrypted", sa.String(512), nullable=False, server_default=""),
        )
    if "vision_base_url" not in existing:
        op.add_column(
            "user_llm_providers",
            sa.Column("vision_base_url", sa.String(512), nullable=False, server_default=""),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "user_llm_providers" not in sa.inspect(bind).get_table_names():
        return
    existing = {c["name"] for c in sa.inspect(bind).get_columns("user_llm_providers")}
    for col in ("vision_base_url", "vision_api_key_encrypted", "vision_binding"):
        if col in existing:
            op.drop_column("user_llm_providers", col)
