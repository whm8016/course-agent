"""loadtest_mock 单测：压测 mock 的「规格」——验证 chunk 结构对齐 loop.py 解析、
FORCE_FAIL 异常可被 reliability 识别。生产零影响（不设 LOAD_TEST_MOCK_LLM 时
llm.py 分流 if 永不进入）已由全量 pytest 494 绿覆盖，此处专测 mock 本身。"""
from __future__ import annotations

import pytest

from core.llm.loadtest_mock import _FakeAPIError, maybe_loadtest_mock_stream


@pytest.mark.asyncio
async def test_mock_stream_yields_content_chunks(monkeypatch):
    """mock 返回 async gen，迭代出的 chunk.choices[0].delta.content 是 str（对齐 loop.py:211）。"""
    monkeypatch.delenv("LOAD_TEST_FORCE_FAIL_RATIO", raising=False)
    gen = await maybe_loadtest_mock_stream(0.0, 20)  # ttft=0 加速
    contents: list[str] = []
    async for chunk in gen:
        choices = getattr(chunk, "choices", None) or []
        assert choices, "chunk 必须有 choices（loop.py:185）"
        delta = getattr(choices[0], "delta", None)
        assert delta is not None, "choices[0].delta 必须存在（loop.py:188）"
        content = getattr(delta, "content", None)
        if content:
            assert isinstance(content, str)
            contents.append(content)
    assert contents, "应至少产出一个 content chunk（否则 loop 测不到 TTFT）"


@pytest.mark.asyncio
async def test_mock_stream_final_chunk_has_stop(monkeypatch):
    """末尾 chunk 带 finish_reason=stop（贴近真实流式收尾语义）。"""
    monkeypatch.delenv("LOAD_TEST_FORCE_FAIL_RATIO", raising=False)
    gen = await maybe_loadtest_mock_stream(0.0, 5)
    last = None
    async for chunk in gen:
        last = chunk
    assert last is not None
    assert getattr(last.choices[0], "finish_reason", None) == "stop"


@pytest.mark.asyncio
async def test_force_fail_raises_retryable_503(monkeypatch):
    """FORCE_FAIL_RATIO=1.0 时抛 _FakeAPIError，status_code=503 命中 reliability.retryable_status_codes。"""
    monkeypatch.setenv("LOAD_TEST_FORCE_FAIL_RATIO", "1.0")
    with pytest.raises(_FakeAPIError) as ei:
        await maybe_loadtest_mock_stream(0.0, 10)
    # reliability.py:301-306 用 getattr(e, 'status_code', None) 提取，503 在 retryable_status_codes
    assert int(getattr(ei.value, "status_code", None)) == 503


@pytest.mark.asyncio
async def test_force_fail_zero_never_raises(monkeypatch):
    """FORCE_FAIL_RATIO=0（默认）时 mock 永不抛异常 → reliability 计 success，不触发熔断。"""
    monkeypatch.setenv("LOAD_TEST_FORCE_FAIL_RATIO", "0")
    gen = await maybe_loadtest_mock_stream(0.0, 10)
    async for _ in gen:  # 能正常迭代完即不抛
        pass
