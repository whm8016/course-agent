"""Add memory_episodes.claimed_at（巩固租约时刻）.

cron_consolidate_memory 旧口径用 created_at 判 processing 超时——created_at 是
对话发生时刻，被捞走的 episode 按定义已 pending ≥1800s，条件恒真：巩固 job 只要
在一次 cron 周期（300s）内没跑完（最坏 20 段 × 并发 4 × 120s 判定超时），就被翻回
pending 重领重跑——mem0 event 槽双写、persona 双抽取、mastery_observation 追加双份
（append-only 无法事后去重，fold_observations 全量重放时双计污染学情）。

claim_pending 把 claimed_at 与 status='processing' 在同一 UPDATE 原子写入；cron 按
claimed_at 判租约超时（NULL = 迁移前遗留行，视同超租约，回收一次后即有值）。

- nullable=True：存量行留 NULL，单步安全加列，无需回填。
- 判龄走 (status, claimed_at) 过滤，行数量级小（outbox 模式，done 即出队），
  复用 idx_episodes_outbox(status, created_at) 前缀即可，不另建索引。

Revision ID: 038
Revises: 037
Create Date: 2026-09-25
"""

# revision identifiers, used by Alembic.
revision = "038"
down_revision = "037"
branch_labels: None = None
depends_on: None = None

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


def _column_exists(table: str, column: str) -> bool:
    bind = op.get_bind()
    return column in [c["name"] for c in sa.inspect(bind).get_columns(table)]


def upgrade() -> None:
    if not _column_exists("memory_episodes", "claimed_at"):
        op.add_column(
            "memory_episodes",
            sa.Column("claimed_at", sa.Float(), nullable=True),
        )


def downgrade() -> None:
    if _column_exists("memory_episodes", "claimed_at"):
        op.drop_column("memory_episodes", "claimed_at")
