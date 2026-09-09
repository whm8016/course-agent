"""L2 目标不变式逐轮回归（L2 返修核心验收）。

不变式：任一条会话消息，任一时刻，要么在摘要卡里（游标之前），要么在本轮原文里
（游标之后）——不存在两边都没有的状态。读侧负责兜住这条，与压缩何时跑无关。

模拟 11 轮会话（服务端每轮落 user/assistant 两行 + 读侧触发压缩），逐轮断言：
  1) 全部消息按 id 构成「已覆盖 ∪ 在尾部」无缝分区；
  2) 游标单调前进（不回退）；
  3) 第 9、10 轮重点：滑出旧实现 [-10:] 条数窗口的消息全部已被卡覆盖——旧实现
     （读侧按条数砍 + 写侧 3 轮闸推游标）在此处存在 1-2 轮真空，新实现恒无缝。

不依赖真实 LLM：_do_compress 全程 patch 返回固定 v2 JSON。真实 in-memory SQLite。
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from core.db.database import AsyncSessionLocal, Message, Session
from core.memory import session_summary as ss
from core.memory.memory import add_message
from settings import get_settings

_SID = "sess-invariant-1"
_ROUNDS = 11


async def _all_msgs(db):
    return list((await db.execute(
        select(Message).where(Message.session_id == _SID)
        .order_by(Message.created_at, Message.id)
    )).scalars().all())


@pytest.mark.asyncio
async def test_eleven_round_invariant_no_gap(client, monkeypatch):
    """11 轮逐轮断言：已覆盖 ∪ 在尾部 = 全量，无缝、游标单调。"""
    # token 闸调低（50）：2-3 轮即触发一次压缩，让游标在 11 轮里前进多次
    monkeypatch.setattr(get_settings().summary, "compact_trigger_tokens", 50)

    db = AsyncSessionLocal()
    db.add(Session(id=_SID, course_id="c1", user_id="u1", title="t"))
    await db.commit()

    async def fake_compress(self, existing, msgs, track=None):
        return json.dumps({"v": 2, "items": [
            {"k": "topic", "key": f"r{i}", "t": f"第{i}轮讨论", "ts": 1.0, "n": 1}
            for i, m in enumerate(msgs)
        ]}, ensure_ascii=False)

    mgr = ss.get_summary_manager()
    prev_cursor_id: str | None = None
    gaps: list[tuple[int, str]] = []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ss.SessionSummaryManager, "_do_compress", fake_compress)
        for r in range(_ROUNDS):
            # -- 服务端回合落盘（turn_runtime 路径的等价简化）--
            content = f"第{r}轮问题，" + "上下文细节" * (3 + r)  # 逐轮变长，后期触发 token 闸
            await add_message(db, _SID, "user", content, turn_id=f"turn-r{r}")
            await add_message(db, _SID, "assistant", f"第{r}轮回答，" + "解答细节" * (3 + r),
                              metadata={"mode": "chat"}, turn_id=f"turn-r{r}")
            await db.commit()

            # -- 读侧触发压缩（读侧排队 -> 落定；token 闸内部决定压不压）--
            await mgr.maybe_compress(db, _SID)

            # -- 不变式断言 --
            db.expire_all()  # 清 ORM 身份映射，重读游标最新值
            session = await db.get(Session, _SID)
            cursor_ts, cursor_id = await ss._resolve_cursor(db, session)
            tail = await ss.load_uncompressed_tail(db, _SID)

            msgs = await _all_msgs(db)
            tail_ids = {m["content"] for m in tail}  # 尾部按 content 近似（id 不出端）
            for m in msgs:
                covered = (
                    cursor_ts is not None
                    and (m.created_at < cursor_ts
                         or (m.created_at == cursor_ts and m.id <= cursor_id))
                )
                in_tail = m.content in tail_ids
                if not covered and not in_tail:
                    gaps.append((r, m.id))

            # 游标单调前进（id 序即 created_at 序）
            if cursor_id is not None and prev_cursor_id is not None:
                cur_idx = [m.id for m in msgs].index(cursor_id)
                prev_idx = [m.id for m in msgs].index(prev_cursor_id)
                assert cur_idx >= prev_idx, f"游标回退 r={r}"
            if cursor_id is not None:
                prev_cursor_id = cursor_id

            # -- 第 9、10 轮重点（旧实现空窗位）：滑出 [-10:] 条数窗口的消息须全被卡覆盖 --
            if r in (8, 9, 10):
                old_window = msgs[-10:]
                slid_out = msgs[:-10] if len(msgs) > 10 else []
                assert len(old_window) == min(10, len(msgs))
                for m in slid_out:
                    covered = (
                        cursor_ts is not None
                        and (m.created_at < cursor_ts
                             or (m.created_at == cursor_ts and m.id <= cursor_id))
                    )
                    assert covered, (
                        f"r={r}: 消息 {m.id} 既不在旧条数窗口也不在卡里——旧实现空窗位未兜住"
                    )

    assert gaps == [], f"存在两边都没有的消息: {gaps}"
    # 11 轮后确实发生过压缩（游标非空、卡非空）——不变式是被真实分区考验过的
    db.expire_all()
    session = await db.get(Session, _SID)
    assert session.summary_up_to_msg_id, "11 轮长会话后游标应已推进"
    assert session.summary, "摘要卡应非空"
    await db.close()
