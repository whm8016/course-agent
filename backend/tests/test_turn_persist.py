"""turn_runtime 回合落盘回归（L2 返修步骤 1：服务端权威回合日志）。

此前 messages 行只有前端流结束后 best-effort saveMessage：关标签页/出错/取消轮
永久丢失。现在 turn_runtime._run_turn finally 落 user/assistant 两行（幂等 by
(session_id, turn_id, role)，含取消/错误轮），前端不再双写。

不依赖真实 LLM/orchestrator：伪 orchestrator 按脚本吐 StreamEvent。
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from sqlalchemy import select

from core.attachment import Attachment
from core.context import UnifiedContext
from core.db.database import AsyncSessionLocal, Message, Session
from core.stream import StreamEvent, StreamEventType
from services.session import turn_runtime as tr

_SID = "sess-persist-1"
_TID = "turn0001persist0000000000ab"  # 32 位内、与 session_id 不同


class _SilentBus:
    """哑 EventBus：publish 即返，不派发任何订阅者。"""

    async def publish(self, event) -> None:
        return None


class _Orch:
    """伪 orchestrator：按脚本吐事件；可选拖尾抛 CancelledError / RuntimeError。"""

    def __init__(self, events, *, raises_cancel=False, raises_error=None):
        self._events = list(events)
        self._cancel = raises_cancel
        self._error = raises_error

    async def handle(self, ctx):
        for e in self._events:
            yield e
        if self._cancel:
            raise asyncio.CancelledError()
        if self._error:
            raise RuntimeError(self._error)


def _ev(t: StreamEventType, **payload) -> StreamEvent:
    return StreamEvent(type=t, payload=payload)


def _ctx(**kw) -> UnifiedContext:
    base = dict(
        course_id="c1", user_id="u1", session_id=_SID, user_message="问个问题",
        mode="chat", conversation_history=[],
    )
    base.update(kw)
    return UnifiedContext(**base)


async def _run_turn(events, ctx, *, turn_id=_TID, cancel=False, error=None):
    mgr = tr.TurnRuntimeManager()
    orch = _Orch(events, raises_cancel=cancel, raises_error=error)
    with patch("core.orchestrator.get_orchestrator", return_value=orch), patch(
        # 哑 EventBus：防全局订阅者（全量顺序下已注册）卡真 Redis 拖死 loop teardown
        "events.event_bus.get_event_bus", lambda: _SilentBus(),
    ):
        await mgr._run_turn(tr._TurnExecution(turn_id=turn_id, context=ctx))


async def _rows(sid=_SID):
    async with AsyncSessionLocal() as db:
        rs = (await db.execute(
            select(Message).where(Message.session_id == sid)
            .order_by(Message.created_at, Message.id)
        )).scalars().all()
        return [(r.role, r.turn_id, r.content, json.loads(r.metadata_ or "{}")) for r in rs]


@pytest.fixture(autouse=True)
def _no_langsmith(monkeypatch):
    """钉死 tracing：本机 .env 开着 LANGSMITH_TRACING=true 时 turn 根 run 真打
    LangSmith（Clash 代理下常卡死）——测试不打外网（test_context_trim_observability
    同款钉法）。"""
    from core.observability import langsmith_trace as lt

    monkeypatch.setattr(lt, "_TRACING_ON", False)


@pytest.fixture
async def seeded_session(client):
    """client fixture 触发 init_db 建表；插一行 Session 供落盘挂靠。"""
    db = AsyncSessionLocal()
    db.add(Session(id=_SID, course_id="c1", user_id="u1", title="t"))
    await db.commit()
    await db.close()
    return _SID


@pytest.mark.asyncio
async def test_normal_turn_persists_user_and_assistant(seeded_session):
    """正常轮：user + assistant 两行都带 turn_id；assistant metadata 对齐前端集合
    （mode/tools_used/usage/truncated）；user 行带 attachments（无 base64）。"""
    ctx = _ctx(attachments=[Attachment(
        type="image", url="/api/uploads/a.png", mime_type="image/png",
        filename="电路图.png", base64="SGVsbG8=",
    )])
    await _run_turn([
        _ev(StreamEventType.TOKEN, content="你"),
        _ev(StreamEventType.TOKEN, content="好！"),
        _ev(StreamEventType.ANSWER, content="你好！"),
        _ev(StreamEventType.DONE, metadata={
            "mode": "chat", "tools_used": ["rag"], "truncated": True,
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 0},
        }),
    ], ctx)

    rows = await _rows()
    assert len(rows) == 2
    (urole, utid, ucontent, umeta), (arole, atid, acontent, ameta) = rows
    assert (urole, arole) == ("user", "assistant")
    assert utid == atid == _TID
    assert ucontent == "问个问题"
    assert umeta["attachments"] == [{
        "type": "image", "url": "/api/uploads/a.png",
        "mime_type": "image/png", "filename": "电路图.png",
    }]  # base64/file_path 不落
    assert acontent == "你好！"
    assert ameta["mode"] == "chat"
    assert ameta["tools_used"] == ["rag"]
    assert ameta["truncated"] is True
    assert ameta["usage"]["input_tokens"] == 10


@pytest.mark.asyncio
async def test_same_turn_id_redelivery_is_idempotent(seeded_session):
    """幂等：同 (session_id, turn_id) 重复投递只产生两行，不重复插入。"""
    events = [
        _ev(StreamEventType.ANSWER, content="答"),
        _ev(StreamEventType.DONE, metadata={"mode": "chat"}),
    ]
    await _run_turn(events, _ctx())
    await _run_turn(events, _ctx())  # 重复投递（同 turn_id）

    rows = await _rows()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_cancelled_turn_persists_partial_output(seeded_session):
    """取消轮：token 部分输出 + 停止尾注（与前端展示格式一致）落盘，stopped=true。"""
    await _run_turn(
        [_ev(StreamEventType.TOKEN, content="部分回")],
        _ctx(), cancel=True,
    )
    rows = await _rows()
    assert len(rows) == 2
    _, _, acontent, ameta = rows[1]
    assert acontent == f"部分回\n\n{tr._STOP_NOTE}"
    assert ameta["stopped"] is True


@pytest.mark.asyncio
async def test_cancelled_before_any_output_persists_note(seeded_session):
    """取消且无任何输出：user/assistant 两行仍落（此前这轮整轮永久丢失）。"""
    await _run_turn([], _ctx(), cancel=True)
    rows = await _rows()
    assert len(rows) == 2
    assert rows[1][2] == tr._STOP_EMPTY_NOTE


@pytest.mark.asyncio
async def test_quiz_turn_persists_empty_content_with_quiz_meta(seeded_session):
    """quiz 轮：题目走 metadata.quiz（QuizCard 渲染口径），content 恒空。"""
    questions = [{"id": "q1", "type": "single", "question": "1+1=?", "options": {}}]
    await _run_turn([
        _ev(StreamEventType.RESULT, questions=questions, metadata={"count": 1}),
        _ev(StreamEventType.DONE, metadata={"mode": "quiz", "count": 1}),
    ], _ctx(mode="quiz", user_message="出一道题"))
    rows = await _rows()
    assert len(rows) == 2
    _, _, acontent, ameta = rows[1]
    assert acontent == ""
    assert ameta["quiz"] == {"questions": questions}
    assert ameta["count"] == 1  # RESULT metadata 一并并入


@pytest.mark.asyncio
async def test_error_turn_persists_error_text(seeded_session):
    """错误轮：assistant 行落错误文本（此前前端见「出错了」即整轮不落库）。"""
    await _run_turn([], _ctx(), error="boom")
    rows = await _rows()
    assert len(rows) == 2
    assert rows[1][2] == "出错了: boom"


@pytest.mark.asyncio
async def test_ephemeral_session_skips_persist(client):
    """session_id==turn_id（start_turn 对无会话 turn 的占位）：无 Session 行，跳过。"""
    ctx = _ctx(session_id=_TID)  # 与 turn_id 相同 → 占位会话
    await _run_turn(
        [_ev(StreamEventType.ANSWER, content="x"),
         _ev(StreamEventType.DONE, metadata={"mode": "chat"})],
        ctx,
    )
    assert await _rows(_TID) == []


@pytest.mark.asyncio
async def test_missing_session_row_skips_persist(client):
    """Session 行不存在：跳过（避免 FK 违约），不抛错不影响回合收尾。"""
    await _run_turn(
        [_ev(StreamEventType.ANSWER, content="x"),
         _ev(StreamEventType.DONE, metadata={"mode": "chat"})],
        _ctx(session_id="no-such-session"),
    )
    assert await _rows("no-such-session") == []


@pytest.mark.asyncio
async def test_persist_skips_foreign_session(client):
    """P1 回归：session 不属于本 turn 用户（session_id 来自请求体）→ 写侧拒落，
    与读侧 load_uncompressed_tail 的 user_id 过滤同口径——防 B 向 A 的会话写行。"""
    db = AsyncSessionLocal()
    db.add(Session(id="sess-foreign", course_id="c1", user_id="owner-u", title="t"))
    await db.commit()
    await db.close()

    await _run_turn(
        [_ev(StreamEventType.ANSWER, content="x"),
         _ev(StreamEventType.DONE, metadata={"mode": "chat"})],
        _ctx(session_id="sess-foreign"),  # ctx.user_id="u1" ≠ owner-u
    )
    assert await _rows("sess-foreign") == []


@pytest.mark.asyncio
async def test_same_turn_rows_created_at_strictly_increasing(seeded_session):
    """P1 回归：同回合 user/assistant 两行 created_at 严格递增——keyset 游标与
    展示排序按 (created_at, id)，同 ts + 随机 id 会乱序甚至被 keyset 永久排除。"""
    await _run_turn(
        [_ev(StreamEventType.ANSWER, content="答"),
         _ev(StreamEventType.DONE, metadata={"mode": "chat"})],
        _ctx(),
    )
    async with AsyncSessionLocal() as db2:
        rs = (await db2.execute(
            select(Message.created_at, Message.role)
            .where(Message.session_id == _SID)
            .order_by(Message.created_at, Message.id)
        )).all()
    assert [r.role for r in rs] == ["user", "assistant"]  # user 恒在前
    assert rs[1].created_at > rs[0].created_at  # 严格递增，非随机 id 决定


@pytest.mark.asyncio
async def test_persist_survives_api_layer_cancel(seeded_session):
    """P0 回归：API 层订阅流一结束就 cancel_turn（chat.py SSE finally / run.py
    尾部的真实行为）——persist 先于 bus.close() 且 shield 包裹，两行必须落库。

    此前 persist 在 close 之后：close 唤醒订阅者 → API cancel 注入 persist 的
    DB await（CancelledError 不被 except Exception 捕）→ 落盘整体失效。
    """
    mgr = tr.TurnRuntimeManager()
    events = [
        _ev(StreamEventType.ANSWER, content="正常回答"),
        _ev(StreamEventType.DONE, metadata={"mode": "chat"}),
    ]
    orch = _Orch(events)
    with patch("core.orchestrator.get_orchestrator", return_value=orch), patch(
        "events.event_bus.get_event_bus", lambda: _SilentBus(),
    ):
        turn_id = await mgr.start_turn(_ctx())
        async for _ev2 in mgr.subscribe_turn(turn_id):
            pass  # 流结束（bus.close() 投的终止信号）
        await mgr.cancel_turn(turn_id)  # API 层随后必做的 cancel

    # 不 await task（API 层同样不 await）——轮询等 shield 后台落盘完成
    rows: list = []
    for _ in range(200):
        rows = await _rows()
        if len(rows) >= 2:
            break
        await asyncio.sleep(0.05)
    assert len(rows) == 2
    assert rows[0][0] == "user" and rows[1][0] == "assistant"
    mgr._executions.pop(turn_id, None)
