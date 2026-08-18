"""bot_notifications 清理 cron 的 purge 函数测试（audit 遗留项）。

口径：只删 N 天前**已读**通知；未读（学生还没消费）与新通知不删。
"""
from __future__ import annotations

import os
import time

import pytest


@pytest.mark.asyncio
async def test_purge_deletes_only_old_read(client):
    from core.bot.notification import purge_old_notifications
    from core.db.database import AsyncSessionLocal, BotNotification, User

    now = time.time()
    async with AsyncSessionLocal() as db:
        u = User(username=f"pn_{os.urandom(3).hex()}", password_hash="x")
        db.add(u)
        await db.flush()
        rows = [
            # (read, age_days, 应删)
            (True, 40, True),    # 旧已读 → 删
            (False, 40, False),  # 旧未读 → 留（还没消费）
            (True, 10, False),   # 新已读 → 留（窗口内）
        ]
        ids = []
        for read, age, _ in rows:
            n = BotNotification(
                user_id=u.id, bot_id="b1", content="x",
                read=read, created_at=now - age * 86400,
            )
            db.add(n)
            await db.flush()
            ids.append(n.id)
        await db.commit()

    removed = await purge_old_notifications(retention_days=30)
    assert removed >= 1

    async with AsyncSessionLocal() as db:
        from sqlalchemy import select

        left = set((await db.execute(select(BotNotification.id).where(
            BotNotification.user_id == u.id))).scalars().all())
    assert ids[0] not in left
    assert ids[1] in left
    assert ids[2] in left
