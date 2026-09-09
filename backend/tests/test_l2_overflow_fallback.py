"""溢出当轮同步兜底回归（L2 返修步骤 3，对齐 Letta 溢出先 summarize 再重试）。

场景：读侧只装游标之后的消息，plan_turn 裁剪超预算历史时被裁消息既不在摘要卡
（游标未及）也不在本轮原文——违反不变式。此时 turn_runtime 当轮同步强制压缩
（force=True 绕频率闸）推进游标，被裁消息进卡，再按新卡重新裁剪。

不依赖真实 LLM：_do_compress 全程 patch；compute_budgets patch 成小预算让裁剪
确定性触发（与 test_context_trim_observability 同手法）。
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from core.context import UnifiedContext
from core.db.database import AsyncSessionLocal, Message, Session
from core.stream import StreamEvent, StreamEventType
from services.session import turn_runtime as tr

_SID = "sess-overflow-1"


class _DoneOrch:
    """伪 orchestrator：直接吐 DONE 收尾。"""

    async def handle(self, ctx):
        yield StreamEvent(type=StreamEventType.DONE, payload={})


def _seed_msgs(n: int):
    """n 条消息 m00..m{n-1}，内容带序号（断言哪些被裁用）。

    单条 ~24 token：16 条总量 ~380 必超 soft(100) 触发裁剪，而最新 2 条 (~48) 装
    得进 carry 半预算 (50)——保证重裁后仍有原文保留（而非全裁光）。
    """
    return [
        (f"m{i:02d}", "user" if i % 2 == 0 else "assistant", f"长消息{i}。" * 4, 1000.0 + i)
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_overflow_triggers_sync_compress_and_replan(client):
    """超预算裁剪越游标 -> 当轮同步压缩推游标（被裁消息进卡）-> 新卡前插重新 plan。

    不变式断言：重裁后保留的原始消息全部 > 新游标（m05），即被裁的全在卡里；
    history 首位是 <session_summary> 续接块。
    """
    # seed：16 条长消息；压缩窗口 window_size=5 -> boundary=倒数第 11 条（m05）
    db = AsyncSessionLocal()
    db.add(Session(id=_SID, course_id="c1", user_id="u1", title="t"))
    for mid, role, content, ca in _seed_msgs(16):
        db.add(Message(id=mid, session_id=_SID, role=role, content=content, created_at=ca))
    await db.commit()
    await db.close()

    fake_settings = SimpleNamespace(
        research=SimpleNamespace(clarify_enabled=False, clarify_wait_timeout_s=0),
        context_budget=SimpleNamespace(coordinator_enabled=True, carry_forward_location="history_prefix"),
    )
    mgr = tr.TurnRuntimeManager()
    ctx = UnifiedContext(
        course_id="c1", user_id="u1", session_id=_SID, user_message="hi", mode="chat",
        conversation_history=[],  # 读侧尾部本应载入；此处直接构造超预算历史进 plan
    )
    # 直接喂超预算历史（模拟读侧已载入 16 条尾部）：塞进 conversation_history
    ctx.conversation_history = [
        {"role": r, "content": c} for _, r, c, _ in _seed_msgs(16)
    ]

    async def fake_compress(existing, msgs, track=None):
        return json.dumps({"v": 2, "items": [
            {"k": "topic", "key": "t1", "t": "早期讨论摘要", "ts": 1.0, "n": 1},
        ]}, ensure_ascii=False)

    from core.memory import session_summary as ss

    with patch("core.orchestrator.get_orchestrator", return_value=_DoneOrch()), \
         patch("services.session.turn_runtime.get_settings", return_value=fake_settings), \
         patch("core.agentic.context_budget.compute_budgets", lambda m: (100, 150)), \
         patch.object(ss.SessionSummaryManager, "_do_compress", new=AsyncMock(side_effect=fake_compress)) as mocked:
        turn_id = await mgr.start_turn(ctx)
        async for _ev in mgr.subscribe_turn(turn_id):
            pass
    _ex = mgr._executions.get(turn_id)
    if _ex and _ex.task:
        await _ex.task
    mgr._executions.pop(turn_id, None)

    # 同步兜底真的压了（force 路径绕频率闸，16 条消息若走异步轮数闸也够——此处断言
    # 压缩在回合内完成且被调用即可）
    assert mocked.await_count >= 1

    # 新游标推进到 boundary=m05（压缩窗口 10 条之外的第一条）
    async with AsyncSessionLocal() as db2:
        row = (await db2.execute(
            select(Session.summary_up_to_msg_id).where(Session.id == _SID)
        )).scalar_one_or_none()
    assert row == "m05"

    # 重裁结果：首位是 <session_summary> 续接块；保留的原始消息最旧 > m05（被裁全进卡）
    hist = ctx.conversation_history
    assert hist[0]["role"] == "system"
    assert hist[0]["content"].startswith("<session_summary>")
    assert "早期讨论摘要" in hist[0]["content"]
    # 剩余原始消息条数 < 16 且全部来自 m06..m15（内容以"长消息6".."长消息15"为界）
    originals = [m for m in hist[1:]]
    assert 0 < len(originals) < 16
    assert all("长消息" in m["content"] for m in originals)
    idxs = [int(m["content"].split("。")[0].replace("长消息", "")) for m in originals]
    assert min(idxs) >= 6  # m06 起——m00..m05 已进卡，不再出现在原文里
    assert max(idxs) == 15  # 最新消息永不裁
