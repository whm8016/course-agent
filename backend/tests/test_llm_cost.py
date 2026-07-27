"""LLM 成本可观测性测试（第一批）。

覆盖：
- usage_from_response_chunk：OpenAI 末块 / 无 usage / 部分 usage 归一
- 跨 provider 契约：anthropic_adapter._openai_usage_chunk 合成的块能被 usage_from_response_chunk 读回
- estimate_cost：精确命中 / 前缀模糊命中 / 未命中 / None usage
- TokenUsage.add
- _one_round 端到端：流式末块的 usage 被正确捕获（原 bug——空-choices 的 usage 块被
  `if not choices: continue` 跳掉），且 kwargs 携带 stream_options
- Anthropic _run 端到端：从 message_start/message_delta 攒 usage，message_stop 合成等价块
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agentic.loop import _one_round
from core.llm.providers.anthropic_adapter import (
    _AnthropicStream,
    _openai_usage_chunk,
)
from core.observability.cost import (
    TokenUsage,
    estimate_cost,
    usage_from_response_chunk,
)


# ---------------------------------------------------------------------------
# usage_from_response_chunk
# ---------------------------------------------------------------------------

def _openai_usage_obj(prompt, completion, cached=None):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=details,
    )


def test_usage_from_openai_chunk_with_cache():
    chunk = SimpleNamespace(choices=[], usage=_openai_usage_obj(120, 80, cached=40))
    u = usage_from_response_chunk(chunk)
    assert u == TokenUsage(input_tokens=120, output_tokens=80, cache_read_tokens=40)


def test_usage_from_chunk_without_usage_attr():
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="x"))])
    assert usage_from_response_chunk(chunk) is None


def test_usage_from_chunk_with_none_usage():
    chunk = SimpleNamespace(choices=[], usage=None)
    assert usage_from_response_chunk(chunk) is None


def test_usage_from_chunk_without_cache_details():
    chunk = SimpleNamespace(choices=[], usage=_openai_usage_obj(50, 10))
    u = usage_from_response_chunk(chunk)
    assert u == TokenUsage(input_tokens=50, output_tokens=10, cache_read_tokens=0)


def test_cache_hit_rate_computable():
    """验收项：cache 命中率 = cache_read / input 可从 usage 算出。"""
    u = usage_from_response_chunk(
        SimpleNamespace(choices=[], usage=_openai_usage_obj(1000, 200, cached=600))
    )
    assert u.cache_read_tokens / u.input_tokens == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 跨 provider 契约：Anthropic 合成块 ↔ usage_from_response_chunk
# ---------------------------------------------------------------------------

def test_anthropic_synthesized_chunk_read_back_by_normalizer():
    """Anthropic 适配器合成的 usage 块必须能被同一套 usage_from_response_chunk 读回——
    这是「loop 侧一套代码读两类 provider」的契约保证。"""
    chunk = _openai_usage_chunk(input_tokens=10, output_tokens=5, cache_read=3)
    assert usage_from_response_chunk(chunk) == TokenUsage(10, 5, 3)


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_prefix_match():
    # deepseek-v4-pro 未全等命中，应模糊命中家族前缀 deepseek-v4
    cost = estimate_cost("deepseek-v4-pro", TokenUsage(1_000_000, 500_000, 200_000))
    # input 0.5 + output 1.0 + cache 0.02 = 1.52
    assert cost == pytest.approx(1.52, abs=1e-6)


def test_estimate_cost_exact_match():
    cost = estimate_cost("gpt-4o", TokenUsage(1_000_000, 0, 0))
    assert cost == pytest.approx(2.5, abs=1e-6)


def test_estimate_cost_unknown_model_returns_zero():
    assert estimate_cost("totally-unknown-model", TokenUsage(100, 100, 100)) == 0.0


def test_estimate_cost_none_usage_returns_zero():
    assert estimate_cost("gpt-4o", None) == 0.0


def test_longest_prefix_wins():
    """deepseek-reasoner 应命中 deepseek-reasoner 全等键（而非更短的 deepseek 前缀假键）。"""
    cost = estimate_cost("deepseek-reasoner", TokenUsage(1_000_000, 0, 0))
    # deepseek-reasoner input 0.55
    assert cost == pytest.approx(0.55, abs=1e-6)


# ---------------------------------------------------------------------------
# TokenUsage.add
# ---------------------------------------------------------------------------

def test_token_usage_add():
    assert TokenUsage(1, 2, 3).add(TokenUsage(4, 5, 6)) == TokenUsage(5, 7, 9)
    # add None 等价不变（返回等值新实例）
    assert TokenUsage(1, 2, 3).add(None) == TokenUsage(1, 2, 3)


# ---------------------------------------------------------------------------
# _one_round 端到端：usage 捕获 + stream_options
# ---------------------------------------------------------------------------

class _FakeStream:
    """按给定 chunk 列表顺序 yield 的最小 async 迭代器。"""

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
        choices=[SimpleNamespace(delta=SimpleNamespace(
            content=text, tool_calls=None, reasoning_content=None,
        ))],
    )


def _usage_chunk(prompt, completion, cached=0):
    return SimpleNamespace(choices=[], usage=_openai_usage_obj(prompt, completion, cached))


@pytest.mark.asyncio
async def test_one_round_captures_usage_and_sends_stream_options(monkeypatch):
    """原 bug：usage 末块 choices=[] 被 `if not choices: continue` 跳掉。
    改后应在 continue 之前取出 usage，并要求 kwargs 带 stream_options。"""
    captured: dict = {}

    async def fake_create(client, kwargs, binding, model, circuit_breaker=None):
        captured.update(kwargs)
        return _FakeStream([_content_chunk("hello"), _usage_chunk(120, 80, cached=40)])

    monkeypatch.setattr("core.agentic.loop._create_with_image_fallback", fake_create)

    result = await _one_round([], None, "deepseek-v4-pro")

    # 1) kwargs 确实请求了 include_usage
    assert captured.get("stream_options") == {"include_usage": True}
    # 2) usage 从末块正确捕获（未被空-choices 跳过）
    assert result.usage == TokenUsage(input_tokens=120, output_tokens=80, cache_read_tokens=40)
    # 3) 正文不受影响
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_one_round_usage_none_when_provider_omits_it(monkeypatch):
    """provider 不支持 usage（或降级剥了 stream_options）时 result.usage 为 None，不崩。"""

    async def fake_create(client, kwargs, binding, model, circuit_breaker=None):
        return _FakeStream([_content_chunk("only content")])

    monkeypatch.setattr("core.agentic.loop._create_with_image_fallback", fake_create)
    result = await _one_round([], None, "deepseek-v4-pro")
    assert result.usage is None
    assert result.content == "only content"


# ---------------------------------------------------------------------------
# Anthropic _run 端到端：message_start/message_delta 攒 usage → message_stop 合成块
# ---------------------------------------------------------------------------

class _FakeAnthropicMessages:
    """模拟 anthropic 的 messages.stream(**kwargs) async 上下文管理器。"""

    def __init__(self, events):
        self._events = list(events)

    def stream(self, **kwargs):
        outer = self

        class _CM:
            async def __aenter__(self_inner):
                return outer

            async def __aexit__(self_inner, *exc):
                return False

            def __aiter__(self_inner):
                return self

        return _CM()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._events:
            return self._events.pop(0)
        raise StopAsyncIteration


class _FakeAnthropicClient:
    def __init__(self, events):
        self.messages = _FakeAnthropicMessages(events)


@pytest.mark.asyncio
async def test_anthropic_stream_emits_usage_chunk_from_events():
    events = [
        SimpleNamespace(  # message_start：input + cache_read
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(
                input_tokens=100, cache_read_input_tokens=30, output_tokens=1,
            )),
        ),
        SimpleNamespace(  # 文本 delta
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="hi"),
        ),
        SimpleNamespace(  # message_delta：output
            type="message_delta",
            usage=SimpleNamespace(output_tokens=50),
        ),
        SimpleNamespace(  # message_stop
            type="message_stop",
            message=SimpleNamespace(stop_reason="end_turn"),
        ),
    ]
    stream = _AnthropicStream(client=_FakeAnthropicClient(events), create_kwargs={})
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    # 末块必须是合成的 usage 块，且能被 normalizer 读回正确值
    assert usage_from_response_chunk(chunks[-1]) == TokenUsage(
        input_tokens=100, output_tokens=50, cache_read_tokens=30,
    )
    # 文本块仍在（验证 usage 合成没破坏正文流）
    assert any(
        getattr(getattr(getattr(c, "choices", [None])[0] or SimpleNamespace(), "delta", SimpleNamespace()), "content", None) == "hi"
        for c in chunks
    )
