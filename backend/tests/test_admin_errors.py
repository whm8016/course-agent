"""turn_errors 表 + GET /admin/errors/recent 规格测试（可观测性重整 Step 6）。

spec 先行：断言表列契约（ref 精确查）、过滤参数、admin 鉴权、失败 turn 落行。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import select

from core.context import UnifiedContext
from core.db.database import AsyncSessionLocal, Session, TurnErrorRecord
from core.stream import StreamEvent, StreamEventType
from services.session import turn_runtime as tr

_SID = "sess-err-1"

_TID = "turnerr1persist00000000xy"  # 前 8 位 = "turnerr1"


@pytest.fixture(autouse=True)
def _no_langsmith(monkeypatch):
    """钉死 tracing：本机 .env 开着 LANGSMITH_TRACING=true 时 turn 根 run 真打
    LangSmith——测试不打外网（test_turn_persist 同款钉法）。"""
    from core.observability import langsmith_trace as lt

    monkeypatch.setattr(lt, "_TRACING_ON", False)


class _SilentBus:
    async def publish(self, event) -> None:
        return None


class _Orch:
    def __init__(self, events):
        self._events = list(events)

    async def handle(self, ctx):
        for e in self._events:
            yield e


async def _run_failing_turn():
    """跑一个带 ERROR 事件的 turn（capability 级错误，带信封字段）。"""
    orch = _Orch([StreamEvent(type=StreamEventType.ERROR, payload={
        "message": "AI 服务暂时不可用，请稍后重试。", "source": "chat",
        "code": "llm_unavailable", "ref": _TID[:8],
        "exc_type": "LLMRetryError", "exc_head": "LLM call failed after 4 attempts",
    })])
    ctx = UnifiedContext(
        course_id="c1", user_id="u_err", session_id=_SID, user_message="hi",
        mode="chat", conversation_history=[],
    )
    mgr = tr.TurnRuntimeManager()
    with patch("core.orchestrator.get_orchestrator", return_value=orch), patch(
        "events.event_bus.get_event_bus", lambda: _SilentBus(),
    ):
        await mgr._run_turn(tr._TurnExecution(turn_id=_TID, context=ctx))


async def _seed_session():
    db = AsyncSessionLocal()
    db.add(Session(id=_SID, course_id="c1", user_id="u_err", title="t"))
    await db.commit()
    await db.close()


@pytest.mark.asyncio
async def test_failed_turn_writes_turn_errors_row(client):
    """失败 turn → turn_errors 落一行：ref/turn_id/code/exc_type 完整（best-effort 写）。"""
    await _seed_session()
    await _run_failing_turn()

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(TurnErrorRecord).where(TurnErrorRecord.turn_id == _TID)
        )).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.ref == "turnerr1"
    assert row.code == "llm_unavailable"
    assert row.exc_type == "LLMRetryError"
    assert "after 4 attempts" in row.exc_head
    assert row.session_id == _SID
    assert row.user_id == "u_err"
    assert row.course_id == "c1"
    assert row.mode == "chat"
    assert row.message


@pytest.mark.asyncio
async def test_admin_errors_recent_ref_lookup(client, admin_headers):
    """学生报障主路径：?ref= 精确匹配命中该行；错误 ref 查不到。"""
    await _seed_session()
    await _run_failing_turn()

    r = await client.get(
        f"/api/admin/errors/recent?ref={_TID[:8]}", headers=admin_headers
    )
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["turn_id"] == _TID
    assert items[0]["code"] == "llm_unavailable"
    assert items[0]["ref"] == _TID[:8]

    r2 = await client.get("/api/admin/errors/recent?ref=zzzzzzz", headers=admin_headers)
    assert r2.status_code == 200
    assert r2.json()["items"] == []


@pytest.mark.asyncio
async def test_admin_errors_recent_requires_admin(client):
    """非 admin 401/403——错误流水含用户粒度信息，学生不可读。"""
    r = await client.get("/api/admin/errors/recent")
    assert r.status_code in (401, 403)
