"""Add llm_usage_records + llm_usage_daily tables (LLM 用量统计).

两级存储（对标 LiteLLM SpendLogs/DailyUserSpend + Langfuse 摄取时算成本）：
- ``llm_usage_records``：append-only 明细，每个 run_agent_loop 一行，带 mode/rounds（cost-of-pass
  分析，arXiv:2504.13359）。无 FK，抗级联删（删用户不删旧账），同 research_checkpoints（022）。
- ``llm_usage_daily``：日聚合读模型，ARQ cron「删今日+昨日后重插」天然幂等；展示层只读本表。

新表（无存量行），SQLite 与 PG 均安全。

Revision ID: 024
Revises: 023
Create Date: 2026-08-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 明细表 ──
    op.create_table(
        "llm_usage_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("course_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("session_id", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("turn_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("mode", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("model", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("rounds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index("idx_usage_user_time", "llm_usage_records", ["user_id", "created_at"])
    op.create_index("idx_usage_course_time", "llm_usage_records", ["course_id", "created_at"])
    op.create_index("idx_usage_created", "llm_usage_records", ["created_at"])

    # ── 日汇总表 ──
    op.create_table(
        "llm_usage_daily",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("day", sa.String(length=8), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("course_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("model", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("cache_read_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("call_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.Float(), nullable=False),
    )
    op.create_index(
        "uq_usage_daily",
        "llm_usage_daily",
        ["day", "user_id", "course_id", "model"],
        unique=True,
    )
    op.create_index("idx_usage_daily_day", "llm_usage_daily", ["day"])


def downgrade() -> None:
    op.drop_table("llm_usage_daily")
    op.drop_table("llm_usage_records")
