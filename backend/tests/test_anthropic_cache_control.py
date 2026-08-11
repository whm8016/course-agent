"""anthropic_adapter cache_control 断点测试（免 anthropic SDK：直接测转换函数 + __new__ 绕 __init__）。

覆盖：
- _convert_tools：默认无断点；cache_control=True 时末工具加 ephemeral 断点
- _convert_messages：默认返回 str（逐字同旧）；cache_control=True 时返回 blocks 列表、首块（T1）放断点
- _create：cache_control_enabled=True 时 create_kwargs["system"] 为 blocks；False 时为 str（逐字同旧）
"""
from __future__ import annotations

import pytest

from core.llm.providers.anthropic_adapter import (
    AnthropicAdapter,
    _convert_messages,
    _convert_tools,
)


# ---------------------------------------------------------------------------
# _convert_tools
# ---------------------------------------------------------------------------

def test_convert_tools_no_cache_control_by_default():
    tools = [{"function": {"name": "f1", "description": "d", "parameters": {}}}]
    out = _convert_tools(tools)
    assert out and out[0]["name"] == "f1"
    assert "cache_control" not in out[-1]


def test_convert_tools_cache_control_on_last_tool():
    tools = [
        {"function": {"name": "f1", "description": "d", "parameters": {}}},
        {"function": {"name": "f2", "description": "d", "parameters": {}}},
    ]
    out = _convert_tools(tools, cache_control=True)
    assert "cache_control" not in out[0]               # 非末工具不加
    assert out[-1]["cache_control"] == {"type": "ephemeral"}  # 末工具加断点


def test_convert_tools_none_returns_none():
    assert _convert_tools(None, cache_control=True) is None


# ---------------------------------------------------------------------------
# _convert_messages
# ---------------------------------------------------------------------------

def test_convert_messages_off_returns_str():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    system, converted = _convert_messages(msgs)
    assert system == "sys"  # 字符串，逐字同旧行为
    assert converted[0]["role"] == "user"


def test_convert_messages_on_returns_blocks_with_breakpoint_on_first():
    # 两条 system 消息（T1 + T2，由 loop._build_messages 在 T1/T2 边界拆好）
    msgs = [
        {"role": "system", "content": "T1 稳定前缀"},
        {"role": "system", "content": "T2 易变后缀"},
        {"role": "user", "content": "hi"},
    ]
    system, converted = _convert_messages(msgs, cache_control=True)
    assert isinstance(system, list)
    assert len(system) == 2
    assert system[0]["cache_control"] == {"type": "ephemeral"}  # 首块（T1）放断点
    assert "cache_control" not in system[1]                      # T2 不缓存
    assert system[0]["text"] == "T1 稳定前缀"
    assert converted[0]["role"] == "user"


def test_convert_messages_on_single_system_block_still_caches():
    msgs = [{"role": "system", "content": "only"}, {"role": "user", "content": "hi"}]
    system, _ = _convert_messages(msgs, cache_control=True)
    assert isinstance(system, list) and len(system) == 1
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_convert_messages_on_no_system_returns_empty_str():
    msgs = [{"role": "user", "content": "hi"}]
    system, _ = _convert_messages(msgs, cache_control=True)
    assert system == ""  # 无 system 时不构 blocks


# ---------------------------------------------------------------------------
# _create（绕 __init__ 免 anthropic SDK；monkeypatch settings 开关）
# ---------------------------------------------------------------------------

class _FakeResp:
    content = []
    usage = None


def _make_adapter_with_fake_client(captured: dict) -> AnthropicAdapter:
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)  # 绕 __init__（免 anthropic SDK 依赖）

    class _FakeCreate:
        async def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return _FakeResp()

    class _FakeClient:
        def __init__(self) -> None:
            self.messages = _FakeCreate()

    adapter._client = _FakeClient()
    return adapter


@pytest.mark.asyncio
async def test_create_passes_system_blocks_when_enabled(monkeypatch):
    from settings import get_settings
    monkeypatch.setattr(get_settings().context_budget, "cache_control_enabled", True)

    captured: dict = {}
    adapter = _make_adapter_with_fake_client(captured)
    msgs = [
        {"role": "system", "content": "T1"},
        {"role": "system", "content": "T2"},
        {"role": "user", "content": "hi"},
    ]
    await adapter._create(model="claude-sonnet-4-5", messages=msgs, max_tokens=128)

    kw = captured["kwargs"]
    assert isinstance(kw["system"], list)                       # 结构化 blocks
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in kw["system"][1]


@pytest.mark.asyncio
async def test_create_passes_system_str_when_disabled(monkeypatch):
    from settings import get_settings
    monkeypatch.setattr(get_settings().context_budget, "cache_control_enabled", False)

    captured: dict = {}
    adapter = _make_adapter_with_fake_client(captured)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    await adapter._create(model="claude-sonnet-4-5", messages=msgs, max_tokens=128)
    assert captured["kwargs"]["system"] == "sys"  # 字符串，逐字同旧


@pytest.mark.asyncio
async def test_create_tools_get_breakpoint_when_enabled(monkeypatch):
    from settings import get_settings
    monkeypatch.setattr(get_settings().context_budget, "cache_control_enabled", True)

    captured: dict = {}
    adapter = _make_adapter_with_fake_client(captured)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    tools = [
        {"function": {"name": "rag", "description": "d", "parameters": {}}},
        {"function": {"name": "grades", "description": "d", "parameters": {}}},
    ]
    await adapter._create(model="claude-sonnet-4-5", messages=msgs, tools=tools, max_tokens=128)
    ant_tools = captured["kwargs"]["tools"]
    assert ant_tools[-1]["cache_control"] == {"type": "ephemeral"}  # 末工具断点
    assert "cache_control" not in ant_tools[0]
