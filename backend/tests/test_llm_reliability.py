"""core/llm/llm.py 可靠性下沉测试。

验证 retry+熔断下沉到 _create_with_image_fallback 后的关键特性：
  1. image fallback 零污染：图片不支持 → 闭包内剥图重试成功 → 熔断器 failure_count 不增
     （B-final++ 核心设计：image fallback 放进 with_retry_and_circuit 的 _call 内部，
      整体成功算 1 次 success；防回归——若有人误把 image fallback 移到外层、首次失败
      先计 failure 再剥图，本测试会失败）。
  2. 主路径瞬时错误（429）触发 retry：下沉后主路径裸调失败能自动重试。
  3. 自配 client 不熔断：circuit_breaker=None 时失败不计入任何熔断器——自配供应商是
     用户私有资源，平台不替它兜底，让真实错误冒给用户；也从根本上避免自配失败把全局
     default 熔断器打 OPEN 误伤他人（含同 binding 不同 key 的自配用户）。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.llm.llm import _create_with_image_fallback, _llm_circuit_breaker
from core.llm.reliability import LLMRetryError, RetryConfig


@pytest.mark.asyncio
async def test_image_fallback_does_not_pollute_circuit_breaker():
    """图片不支持 → 闭包内剥图重试成功 → 全局熔断器 failure_count 保持 0（零污染）。"""
    # 带图 messages（OpenAI content-parts 格式）；deepseek-chat 不在 vision 白名单
    messages: list[dict] = [{"role": "user", "content": [
        {"type": "text", "text": "图里是什么？"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]
    create_kwargs = {"model": "deepseek-chat", "messages": messages, "stream": False}

    call_count = 0

    async def _fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("model does not support image input")
        return MagicMock(choices=[MagicMock(message=MagicMock(content="已转为文字作答"))])

    mock_client = MagicMock()
    mock_client.chat.completions.create = _fake_create

    # 不传 circuit_breaker → 走默认全局熔断器（"default"）。校验它零污染。
    _llm_circuit_breaker.reset()
    try:
        result = await _create_with_image_fallback(
            mock_client, create_kwargs, "deepseek", "deepseek-chat"
        )
        # 第一次拒图 → 闭包内剥图 → 第二次成功
        assert call_count == 2
        # 零污染：剥图成功记 success，failure_count 不增
        assert _llm_circuit_breaker.failure_count == 0
        assert _llm_circuit_breaker.get_state().value == "closed"
        assert result is not None
    finally:
        _llm_circuit_breaker.reset()


@pytest.mark.asyncio
async def test_retry_kicks_in_on_transient_error():
    """主路径瞬时错误（429）触发 retry——下沉后主路径裸调失败能自动重试。"""
    create_kwargs = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    call_count = 0

    async def _fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("429 Too Many Requests")
        return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

    mock_client = MagicMock()
    mock_client.chat.completions.create = _fake_create

    # 小 base_delay 加速，避免测试等待
    fast_retry = RetryConfig(max_retries=3, base_delay=0.01, max_delay=0.05, exponential_base=2.0)
    _llm_circuit_breaker.reset()
    try:
        with patch("core.llm.llm._retry_config", fast_retry):
            result = await _create_with_image_fallback(
                mock_client, create_kwargs, "deepseek", "deepseek-chat"
            )
        assert call_count == 3  # 2 次 429 重试 + 第 3 次成功
        assert result is not None
    finally:
        _llm_circuit_breaker.reset()


@pytest.mark.asyncio
async def test_user_supplied_client_not_circuit_broken():
    """自配 client（circuit_breaker=None）失败不计入任何熔断器。

    自配供应商是用户私有资源，平台不替它兜底——失败仍会重试，但不熔断，让真实错误
    （这里是 401 key 失效）冒给用户；也保证自配失败不会把全局 default 熔断器打 OPEN
    误伤其他用户。
    """
    create_kwargs = {
        "model": "some-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }

    async def _always_fail(**kwargs):
        # 401 不可重试：模拟用户自配 key 失效，重试无意义，直接抛
        raise RuntimeError("401 unauthorized")

    mock_client = MagicMock()
    mock_client.chat.completions.create = _always_fail

    _llm_circuit_breaker.reset()
    try:
        with pytest.raises(LLMRetryError):
            await _create_with_image_fallback(
                mock_client, create_kwargs, "openai", "some-model",
                circuit_breaker=None,  # 自配：不熔断
            )
        # 核心保证：全局 default 熔断器零污染——自配失败不误伤他人
        assert _llm_circuit_breaker.failure_count == 0
        assert _llm_circuit_breaker.get_state().value == "closed"
    finally:
        _llm_circuit_breaker.reset()
