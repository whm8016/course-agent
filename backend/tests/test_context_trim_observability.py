"""压缩可观测性三连：LangSmith 上何时压缩 / 压成了什么 / 本轮发了什么。

此前三处各自漏东西：
1. 压缩 LLM 调用与普通聊天同名（course_agent_chat），翻不出哪条是压缩
   -> ls_extra / langsmith_name 给单次 create 打独立 run 名。
2. record_context_trim 只留计数/布尔，摘要正文与被删原文一个字没存
   -> EnforceReport.cleared_originals/summary_text + outputs 补截断正文。
3. 回合前 plan_turn 裁剪只有几个数字挂在 turn 根 run metadata，无独立 run
   -> start_turn 捕获被砍正文，_run_turn 建 context.trim.pretrim 子 run。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from core.context import UnifiedContext
from core.stream import StreamEvent, StreamEventType
from core.observability import langsmith_trace as lt
from settings import get_settings


# ---------------------------------------------------------------------------
# ls_extra：tracing 开关两态
# ---------------------------------------------------------------------------

def test_ls_extra_disabled_returns_empty(monkeypatch):
    """未启用追踪 -> 空 dict：create(**kwargs, **ls_extra(...)) 与原调用逐字节等价。

    原生 openai client 不认识 langsmith_extra（TypeError），这里就是防它混进去。
    状态显式钉住：本机 .env 可能开着 tracing，不能依赖环境默认。
    """
    monkeypatch.setattr(lt, "_TRACING_ON", False)
    assert lt.ls_extra("l2_session_summary", {"session_id": "s1"}) == {}


def test_ls_extra_enabled(monkeypatch):
    monkeypatch.setattr(lt, "_TRACING_ON", True)
    assert lt.ls_extra("l2_session_summary", {"session_id": "s1"}) == {
        "langsmith_extra": {"name": "l2_session_summary", "metadata": {"session_id": "s1"}},
    }
    # metadata 空/None 时不带 key
    assert lt.ls_extra("ctx_summary") == {"langsmith_extra": {"name": "ctx_summary"}}
    assert lt.ls_extra("ctx_summary", None) == {"langsmith_extra": {"name": "ctx_summary"}}


# ---------------------------------------------------------------------------
# record_context_trim：outputs 补正文且统一截断
# ---------------------------------------------------------------------------

def test_record_context_trim_outputs_body_and_truncation(monkeypatch):
    recorded: list[dict] = []

    def fake_record(**kw):
        recorded.append(kw)

    monkeypatch.setattr(lt, "_record", fake_record)
    lt.record_context_trim(
        stage="in_turn",
        coordinator_enabled=True,
        iteration=3,
        dropped_tool_results=2,
        summary_added=True,
        dropped_preview="x" * 3000,
        summary_text="y" * 3000,
    )
    assert len(recorded) == 1
    outputs = recorded[0]["outputs"]
    assert outputs["dropped_tool_results"] == 2
    assert len(outputs["dropped_preview"]) == 800   # 统一截断，调用方无需把关
    assert len(outputs["summary_text"]) == 800
    # 未传正文 -> 空串（老调用点行为兼容，不 KeyError）
    lt.record_context_trim(stage="in_turn", coordinator_enabled=True, iteration=0)
    assert recorded[1]["outputs"]["dropped_preview"] == ""
    assert recorded[1]["outputs"]["summary_text"] == ""


# ---------------------------------------------------------------------------
# chat_complete(langsmith_name=...) 透传到 _make_chat_completion
# ---------------------------------------------------------------------------

async def test_chat_complete_passes_langsmith_name(monkeypatch):
    from core.llm import llm as llm_mod

    seen: list[dict] = []

    class _FakeResp:
        model = "m"
        usage = None
        choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]

    async def fake_make(**kw):
        seen.append(kw)
        return _FakeResp()

    monkeypatch.setattr(llm_mod, "_make_chat_completion", fake_make)
    await llm_mod.chat_complete("sys", [], "hi", langsmith_name="ctx_summary")
    assert seen[0]["langsmith_name"] == "ctx_summary"
    # 不传 -> None（默认），既有调用方零影响
    await llm_mod.chat_complete("sys", [], "hi")
    assert seen[1]["langsmith_name"] is None


async def test_make_chat_completion_injects_extra_when_enabled(monkeypatch):
    """langsmith_name 非空 -> kwargs 塞 langsmith_extra；tracing 关时 ls_extra 返回空、不塞。"""
    from core.llm import llm as llm_mod

    seen: list[dict] = []

    async def fake_call(llm_client, kwargs, binding, model, circuit_breaker=None):
        seen.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(llm_mod, "_create_with_image_fallback", fake_call)
    # 关态：显式钉住（本机 .env 可能开着 tracing），kwargs 不混 langsmith_extra（不炸原生 SDK）
    monkeypatch.setattr(lt, "_TRACING_ON", False)
    await llm_mod._make_chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        langsmith_name="ctx_summary",
    )
    assert "langsmith_extra" not in seen[0]
    # 强制开 -> 注入
    monkeypatch.setattr(llm_mod, "ls_extra",
                        lambda name: {"langsmith_extra": {"name": name}})
    await llm_mod._make_chat_completion(
        model="m", messages=[{"role": "user", "content": "hi"}],
        langsmith_name="ctx_summary",
    )
    assert seen[1]["langsmith_extra"] == {"name": "ctx_summary"}


# ---------------------------------------------------------------------------
# turn_runtime：pretrim 被砍正文捕获 + context.trim.pretrim 子 run
# ---------------------------------------------------------------------------

class _DoneOrch:
    """伪 orchestrator：直接吐 DONE 收尾。"""

    async def handle(self, ctx):
        yield StreamEvent(type=StreamEventType.DONE, payload={})


async def _run_turn_with_history(history: list[dict], trims: list[dict]) -> tuple[str, dict]:
    """跑一个 turn（长历史触发回合前裁剪），trims 捕获 record_context_trim 调用。

    返回 (turn_id, ctx.metadata)。
    """
    from services.session.turn_runtime import TurnRuntimeManager

    mgr = TurnRuntimeManager()
    ctx = UnifiedContext(
        course_id="c1", user_id="u1", user_message="hi", mode="chat",
        session_id="sess-1", conversation_history=history,
    )
    fake_settings = type(
        "S", (),
        {
            "research": type(
                "R", (), {"clarify_enabled": False, "clarify_wait_timeout_s": 0}
            )(),
            "context_budget": type("CB", (), {"coordinator_enabled": False})(),
        },
    )()
    def spy_trim(**kw):
        trims.append(kw)
        # 只记录不上报：本机 .env 开着 tracing 时真函数会向 LangSmith 上传（测试不打外网）

    with patch("core.orchestrator.get_orchestrator", return_value=_DoneOrch()), \
         patch("services.session.turn_runtime.get_settings", return_value=fake_settings), \
         patch("services.session.turn_runtime.resolve_budget", lambda m: 100), \
         patch("core.observability.langsmith_trace.record_context_trim", spy_trim):
        turn_id = await mgr.start_turn(ctx)
        async for _ev in mgr.subscribe_turn(turn_id):
            pass
    await asyncio.sleep(0.05)
    mgr._executions.pop(turn_id, None)  # 免 TTL 清理任务悬挂
    return turn_id, ctx.metadata


async def test_pretrim_records_dropped_previews():
    """超预算历史 -> 回合前裁掉最旧消息：被砍正文进 metadata，且记 context.trim.pretrim。

    resolve_budget 被 patch 成 100 token：最后一条短消息留得下、再往前一条就超，
    裁剪确定性触发（不依赖 token 估算与真实窗口大小）。
    """
    long_msg = "这是一个很长的历史消息。" * 20  # 远超 100 token 预算，必被裁
    history = [{"role": "user", "content": long_msg} for _ in range(6)]
    history.append({"role": "user", "content": "最新的问题"})
    trims: list[dict] = []
    _tid, meta = await _run_turn_with_history(history, trims)

    pre = next(t for t in trims if t["stage"] == "pretrim")
    assert pre["dropped_messages"] > 0
    assert "这是一个很长的历史消息" in pre["dropped_preview"]  # 被砍正文可见
    assert "[user]" in pre["dropped_preview"]
    assert pre["summary_text"] in (None, "")  # 旧分支无摘要续接
    # metadata 里同样留了 previews（_run_turn 从这里读）
    assert meta["_ctx_pretrim"]["dropped"] == pre["dropped_messages"]
    assert len(meta["_ctx_pretrim_dropped_previews"]) == pre["dropped_messages"]


async def test_pretrim_no_run_when_nothing_dropped():
    """历史在预算内 -> 零裁剪回合不产生噪声 pretrim run。"""
    trims: list[dict] = []
    await _run_turn_with_history(
        [{"role": "user", "content": "short"}], trims
    )
    assert not [t for t in trims if t["stage"] == "pretrim"]


async def test_pretrim_carry_forward_captures_summary_preview(monkeypatch):
    """coordinator 开 + 有摘要兜底 + 超阈值 -> carry_forward 前插：摘要正文捕获进 metadata。

    session_summary 会被 start_turn 清空（避免 system 双付），preview 必须在清空前往
    metadata 存一份，否则 pretrim run 的 summary_text 永远为空。
    """
    from core.agentic import context_budget as cb
    from services.session import turn_runtime as trm_mod

    monkeypatch.setattr(cb, "compute_budgets", lambda m: (10, 20))  # 极小阈值必裁
    # plan_turn 内部读真 settings 的 carry_forward_location（turn 分支开关由上面的
    # fake_settings 控制，无需动真值）
    monkeypatch.setattr(
        get_settings().context_budget, "carry_forward_location", "history_prefix"
    )

    from services.session.turn_runtime import TurnRuntimeManager

    mgr = TurnRuntimeManager()
    ctx = UnifiedContext(
        course_id="c1", user_id="u1", user_message="hi", mode="chat",
        session_id="sess-2",
        conversation_history=[
            {"role": "user", "content": f"很长的历史消息 {i} " + "x" * 200}
            for i in range(4)
        ],
    )
    ctx.session_summary = "此前对话的摘要正文：学了导数定义。"
    fake_settings = type(
        "S", (),
        {
            "research": type(
                "R", (), {"clarify_enabled": False, "clarify_wait_timeout_s": 0}
            )(),
            "context_budget": type("CB", (), {"coordinator_enabled": True})(),
        },
    )()
    trims: list[dict] = []

    def spy_trim(**kw):
        trims.append(kw)

    with patch("core.orchestrator.get_orchestrator", return_value=_DoneOrch()), \
         patch.object(trm_mod, "get_settings", return_value=fake_settings), \
         patch.object(lt, "record_context_trim", spy_trim):
        turn_id = await mgr.start_turn(ctx)
        async for _ev in mgr.subscribe_turn(turn_id):
            pass
    await asyncio.sleep(0.05)
    mgr._executions.pop(turn_id, None)

    pre = next(t for t in trims if t["stage"] == "pretrim")
    assert pre["summary_added"] is True  # carry_forward 前插发生
    assert "学了导数定义" in pre["summary_text"]  # 摘要正文可见（session_summary 已清空）
    assert ctx.session_summary == ""  # 清空防双付的原有行为不变
