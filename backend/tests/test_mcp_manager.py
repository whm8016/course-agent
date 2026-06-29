"""MCP manager + adapter + registry 路由单测。

不连真实 MCP server：直接构造 status=connected、session=mock 的连接，
测 call_tool 的瞬态重试/超时/未连接、adapter 命名与 schema、MCP 工具注册进
ToolRegistry、execute_tool("mcp_...") 经 registry.execute 调通。
"""
import asyncio

from core.agent.tool_registry import execute_tool
from core.mcp.adapter import MCPToolAdapter, wrapped_tool_name
from core.mcp.config import MCPServerConfig
from core.mcp.manager import MCPConnectionManager, _ServerConnection

import pytest


@pytest.fixture(autouse=True)
def _clean_mcp_registry():
    """_connected_manager 把 MCP 工具注册进全局 ToolRegistry 单例；每测后清残留。"""
    yield
    from core.agent.registry import get_tool_registry
    for name in [n for n in get_tool_registry().names() if n.startswith("mcp_")]:
        get_tool_registry().unregister(name)


# ── 纯函数 / 数据载体 ────────────────────────────────────────────────────

def test_wrapped_tool_name_sanitizes():
    assert wrapped_tool_name("math", "calc") == "mcp_math_calc"
    assert wrapped_tool_name("my.server", "calc") == "mcp_my_server_calc"  # '.' 替换为 '_'
    # '-' 是合法 tool name 字符，保留（OpenAI tool name 允许 [a-zA-Z0-9_-]）
    assert wrapped_tool_name("srv", "tool-1") == "mcp_srv_tool-1"


def test_adapter_naming_and_schema():
    a = MCPToolAdapter(
        server_name="math", original_name="calc", description="计算器",
        input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
        tool_timeout=20,
    )
    assert a.wrapped_name == "mcp_math_calc"
    assert a.deferred is True
    schema = a.to_openai_schema()
    assert schema["function"]["name"] == "mcp_math_calc"
    assert schema["function"]["description"] == "[math] 计算器"
    assert schema["function"]["parameters"]["properties"]["x"]["type"] == "number"


def test_adapter_schema_setdefaults_for_empty_input():
    a = MCPToolAdapter(server_name="s", original_name="t", description="d",
                       input_schema=None, tool_timeout=5)
    p = a.to_openai_schema()["function"]["parameters"]
    assert p["type"] == "object" and p["properties"] == {}


# ── mock session helpers ─────────────────────────────────────────────────

class _TextBlock:
    def __init__(self, text: str):
        self.text = text


class _CallResult:
    def __init__(self, texts: list[str]):
        self.content = [_TextBlock(t) for t in texts]


class _MockSession:
    """首次调用可选抛瞬态错误（测重试）；否则返回 result。"""

    def __init__(self, *, result: list[str] | None = None, transient_first: bool = False):
        self._result = result or ["ok"]
        self._transient_first = transient_first
        self.calls = 0

    async def call_tool(self, name, arguments=None):
        self.calls += 1
        if self._transient_first and self.calls == 1:
            raise BrokenPipeError()
        return _CallResult(self._result)


def _connected_manager(server, tool, *, session=None, result=None):
    mgr = MCPConnectionManager()
    conn = _ServerConnection(
        name=server, config=MCPServerConfig(url="https://h/mcp"), signature="x"
    )
    conn.status = "connected"
    conn.session = session or _MockSession(result=result)
    adapter = MCPToolAdapter(
        server_name=server, original_name=tool, description=tool,
        input_schema=None, tool_timeout=5,
    )
    conn.adapters = [adapter]
    mgr._connections[server] = conn
    mgr._register_adapters(conn)
    return mgr, adapter


# ── call_tool 行为 ───────────────────────────────────────────────────────

def test_call_tool_returns_text():
    async def run():
        mgr, _ = _connected_manager("math", "calc", result=["42"])
        assert await mgr.call_tool("math", "calc", {}, timeout=5) == "42"
    asyncio.run(run())


def test_call_tool_transient_retry():
    async def run():
        sess = _MockSession(result=["recovered"], transient_first=True)
        mgr, _ = _connected_manager("math", "calc", session=sess)
        assert await mgr.call_tool("math", "calc", {}, timeout=5) == "recovered"
        assert sess.calls == 2  # 重试一次
    asyncio.run(run())


def test_call_tool_not_connected():
    async def run():
        mgr = MCPConnectionManager()
        out = await mgr.call_tool("ghost", "x", {}, timeout=5)
        assert "not connected" in out
    asyncio.run(run())


def test_mcp_registered_into_registry():
    """MCP adapter 经 _register_adapters 注册进 ToolRegistry（替代原 resolve_wrapped 映射）。"""
    async def run():
        from core.agent.registry import get_tool_registry
        mgr, _ = _connected_manager("math", "calc", result=["ok"])
        registry = get_tool_registry()
        entry = registry.get("mcp_math_calc")
        assert entry is not None and entry.deferred is True
        assert registry.get("mcp_math_nope") is None
    asyncio.run(run())


# ── MCP 工具经 registry.execute 路由 ──────────────────────────────────────

def test_execute_tool_mcp_via_registry():
    async def run():
        # _connected_manager 已把 mcp_math_calc 注册进全局 ToolRegistry
        mgr, _ = _connected_manager("math", "calc", result=["42"])
        res = await execute_tool("mcp_math_calc", course_id="c1")
        assert res.content == "42"
        assert res.sources[0] == {"type": "mcp", "server": "math", "tool": "calc"}
        # 未注册的 mcp_ 工具 → registry 未命中 → success=False
        res2 = await execute_tool("mcp_math_ghost", course_id="c1")
        assert res2.success is False
    asyncio.run(run())
