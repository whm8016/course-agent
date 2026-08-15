"""辅助 LLM 调用用量埋点测试（Fix 3：绕过 agent loop 的调用点纳入统计）。

覆盖：
- ``usage_from_response``：非流式 resp 归一（有/无 usage、cache 细节）
- ``track_llm_usage`` / ``track_llm_response``：mode 默认值、无 usage 跳过、
  record_llm_usage 失败不抛（best-effort 契约）
- ``chat_complete(track=...)``：传则埋点、不传零调用
- ``QuestionAgentBase.stream_llm``：流式 usage 末块捕获 + stream_options 携带，
  端点不支持时剥掉重试（复用 llm._is_stream_options_unsupported）

record_llm_usage 一律 monkeypatch 同模块名（绕开 usage_tracking 开关与
in-memory sqlite 耦合，断言 called-with 即可）。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _resp(prompt=120, completion=80, cached=40, model="deepseek-v4-pro"):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=details,
        ),
    )


# ---------------------------------------------------------------------------
# usage_from_response（非流式归一）
# ---------------------------------------------------------------------------

def test_usage_from_response_reads_usage():
    from core.observability.cost import usage_from_response

    u = usage_from_response(_resp())
    assert u.input_tokens == 120
    assert u.output_tokens == 80
    assert u.cache_read_tokens == 40


def test_usage_from_response_without_usage_returns_none():
    from core.observability.cost import usage_from_response

    assert usage_from_response(SimpleNamespace(choices=[])) is None
    assert usage_from_response(SimpleNamespace(choices=[], usage=None)) is None


# ---------------------------------------------------------------------------
# track_llm_usage / track_llm_response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_track_llm_usage_calls_record_with_defaults(monkeypatch):
    from core.analytics import token_usage as tu
    from core.observability.cost import TokenUsage

    calls: list[dict] = []

    async def fake_record(**kw):
        calls.append(kw)

    monkeypatch.setattr(tu, "record_llm_usage", fake_record)

    await tu.track_llm_usage(
        TokenUsage(100, 50, 20), model="m1",
        user_id="u1", course_id="c1", session_id="s1",
    )
    assert len(calls) == 1
    kw = calls[0]
    assert kw["mode"] == "aux"          # mode 缺省防 track dict 缺 key 静默丢账
    assert kw["model"] == "m1"
    assert kw["user_id"] == "u1"
    assert kw["rounds"] == 1


@pytest.mark.asyncio
async def test_track_llm_usage_none_usage_skips(monkeypatch):
    from core.analytics import token_usage as tu

    called = []

    async def fake_record(**kw):
        called.append(kw)

    monkeypatch.setattr(tu, "record_llm_usage", fake_record)
    await tu.track_llm_usage(None, model="m1")
    assert called == []


@pytest.mark.asyncio
async def test_track_llm_usage_record_failure_does_not_raise(monkeypatch):
    """best-effort 契约：record 抛异常只 warning，绝不向主链路抛。"""
    from core.analytics import token_usage as tu
    from core.observability.cost import TokenUsage

    async def boom(**kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(tu, "record_llm_usage", boom)
    # 不抛即通过
    await tu.track_llm_usage(TokenUsage(1, 1, 0), model="m1", mode="x")


@pytest.mark.asyncio
async def test_track_llm_response_prefers_resp_model(monkeypatch):
    """resp.model 优先（fallback 换模型时账按真实模型查价）。"""
    from core.analytics import token_usage as tu

    calls: list[dict] = []

    async def fake_record(**kw):
        calls.append(kw)

    monkeypatch.setattr(tu, "record_llm_usage", fake_record)
    await tu.track_llm_response(
        _resp(model="fallback-model"), model="requested-model", mode="rag_enrich",
    )
    assert calls[0]["model"] == "fallback-model"
    assert calls[0]["mode"] == "rag_enrich"


@pytest.mark.asyncio
async def test_track_llm_response_without_usage_skips(monkeypatch):
    from core.analytics import token_usage as tu

    called = []

    async def fake_record(**kw):
        called.append(kw)

    monkeypatch.setattr(tu, "record_llm_usage", fake_record)
    await tu.track_llm_response(SimpleNamespace(choices=[]), model="m")
    assert called == []


# ---------------------------------------------------------------------------
# chat_complete(track=...) 钩子
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_complete_track_hook(monkeypatch):
    from core.llm import llm as llm_mod

    tracked: list[tuple] = []

    class _FakeResp:
        model = "deepseek-v4-pro"
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
        )
        choices = [SimpleNamespace(message=SimpleNamespace(content="answer"))]

    async def fake_make(**kw):
        return _FakeResp()

    async def fake_track(resp, *, model, **kw):
        tracked.append((model, kw))

    monkeypatch.setattr(llm_mod, "_make_chat_completion", fake_make)

    from core.analytics import token_usage as tu
    monkeypatch.setattr(tu, "track_llm_response", fake_track)

    out = await llm_mod.chat_complete(
        "sys", [], "hi", model="deepseek-v4-pro", track={"mode": "ctx_summary"},
    )
    assert out == "answer"
    assert tracked == [("deepseek-v4-pro", {"mode": "ctx_summary"})]


@pytest.mark.asyncio
async def test_chat_complete_without_track_zero_calls(monkeypatch):
    from core.llm import llm as llm_mod

    class _FakeResp:
        model = "m"
        usage = None
        choices = [SimpleNamespace(message=SimpleNamespace(content="answer"))]

    async def fake_make(**kw):
        return _FakeResp()

    monkeypatch.setattr(llm_mod, "_make_chat_completion", fake_make)

    from core.analytics import token_usage as tu

    called = []

    async def fake_track(resp, *, model, **kw):
        called.append(model)

    monkeypatch.setattr(tu, "track_llm_response", fake_track)
    await llm_mod.chat_complete("sys", [], "hi")
    assert called == []


# ---------------------------------------------------------------------------
# QuestionAgentBase.stream_llm：流式捕获 + 降级重试
# ---------------------------------------------------------------------------

class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        raise StopAsyncIteration


def _content_chunk(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
    )


def _usage_chunk(prompt, completion, cached=0):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        ),
    )


@pytest.mark.asyncio
async def test_stream_llm_captures_usage_and_requests_stream_options(monkeypatch):
    from core.question import agent_base as ab

    creates: list[dict] = []

    async def fake_create(**kwargs):
        creates.append(kwargs)
        return _FakeStream([_content_chunk("答"), _usage_chunk(120, 80, cached=40)])

    monkeypatch.setattr(
        ab, "_openai_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))),
    )

    tracked: list[dict] = []

    async def fake_track(usage, *, model, **kw):
        tracked.append({"model": model, "usage": usage, **kw})

    from core.analytics import token_usage as tu
    monkeypatch.setattr(tu, "track_llm_usage", fake_track)

    agent = ab.QuestionAgentBase(module_name="q", agent_name="a")
    parts = [
        c async for c in agent.stream_llm(
            "user", "system",
            track={"mode": "quiz_followup", "user_id": "stu1"},
        )
    ]
    assert parts == ["答"]
    # 1) req 带 stream_options
    assert creates[0].get("stream_options") == {"include_usage": True}
    # 2) usage 从末块捕获并埋点（mode/user_id 透传）
    assert len(tracked) == 1
    assert tracked[0]["mode"] == "quiz_followup"
    assert tracked[0]["user_id"] == "stu1"
    assert tracked[0]["usage"].input_tokens == 120
    assert tracked[0]["usage"].cache_read_tokens == 40


@pytest.mark.asyncio
async def test_stream_llm_without_track_no_stream_options(monkeypatch):
    """不传 track：不带 stream_options、不埋点（评测/脚本路径零变化）。"""
    from core.question import agent_base as ab

    creates: list[dict] = []

    async def fake_create(**kwargs):
        creates.append(kwargs)
        return _FakeStream([_content_chunk("x")])

    monkeypatch.setattr(
        ab, "_openai_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))),
    )
    agent = ab.QuestionAgentBase(module_name="q", agent_name="a")
    parts = [c async for c in agent.stream_llm("user", "system")]
    assert parts == ["x"]
    assert "stream_options" not in creates[0]


@pytest.mark.asyncio
async def test_stream_llm_strips_stream_options_on_unsupported(monkeypatch):
    """端点不支持 stream_options（报 400）：剥掉重试一次，正文照常（丢 usage 保流）。"""
    from core.question import agent_base as ab

    creates: list[dict] = []

    async def fake_create(**kwargs):
        creates.append(kwargs)
        if "stream_options" in kwargs:
            raise Exception("Invalid parameter: stream_options is not supported")
        return _FakeStream([_content_chunk("ok")])

    monkeypatch.setattr(
        ab, "_openai_client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))),
    )
    agent = ab.QuestionAgentBase(module_name="q", agent_name="a")
    parts = [
        c async for c in agent.stream_llm(
            "user", "system", track={"mode": "quiz_followup"},
        )
    ]
    assert parts == ["ok"]
    assert len(creates) == 2                       # 重试了一次
    assert "stream_options" in creates[0]          # 首次带了
    assert "stream_options" not in creates[1]      # 重试剥掉
