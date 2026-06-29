"""DeferredToolLoader + render_deferred_tools_manifest + session_state 单测。

验证 load 三态（loaded/already/unknown）、bind_live_schemas 可变引用 append 可见、
initial_schemas stale 清理、session_state 持久化 round-trip。
"""
from core.mcp.adapter import MCPToolAdapter
from core.mcp.deferred_tools import DeferredToolLoader, render_deferred_tools_manifest
from core.mcp.session_state import load_loaded_tools, record_loaded_tools


def _adapter(server: str, tool: str, deferred: bool = True) -> MCPToolAdapter:
    return MCPToolAdapter(
        server_name=server, original_name=tool, description=f"{tool} 工具",
        input_schema=None, tool_timeout=5, deferred=deferred,
    )


def test_render_manifest_groups_by_server():
    tools = [_adapter("math", "calc"), _adapter("math", "plot"), _adapter("web", "fetch")]
    out = render_deferred_tools_manifest(tools)
    assert out.startswith("## 扩展工具")
    assert "### MCP 服务器：math" in out
    assert "### MCP 服务器：web" in out
    assert "**mcp_math_calc**" in out
    assert "**mcp_web_fetch**" in out


def test_render_manifest_empty():
    assert render_deferred_tools_manifest([]) == ""


def test_load_three_states():
    loader = DeferredToolLoader(pool=[_adapter("math", "calc")], session_id="s1")
    live: list = []
    loader.bind_live_schemas(live)
    r1 = loader.load(["mcp_math_calc"])
    assert r1 == {"loaded": ["mcp_math_calc"], "already_loaded": [], "unknown": []}
    assert len(live) == 1
    r2 = loader.load(["mcp_math_calc"])
    assert r2["already_loaded"] == ["mcp_math_calc"]
    assert r2["loaded"] == []
    assert len(live) == 1  # 不重复 append
    r3 = loader.load(["mcp_math_ghost"])
    assert r3["unknown"] == ["mcp_math_ghost"]


def test_initial_schemas_stale_cleanup():
    loader = DeferredToolLoader(
        pool=[_adapter("math", "calc")], session_id="s1",
        loaded={"mcp_math_calc", "mcp_math_gone"},
    )
    initial = loader.initial_schemas()
    names = {s["function"]["name"] for s in initial}
    assert names == {"mcp_math_calc"}  # gone 被清理
    assert "mcp_math_gone" not in loader.loaded_names


def test_bind_live_schemas_is_mutable_reference():
    loader = DeferredToolLoader(pool=[_adapter("math", "calc")], session_id="s1")
    live = [{"existing": True}]
    loader.bind_live_schemas(live)
    loader.load(["mcp_math_calc"])
    assert len(live) == 2  # 原地 append，同引用


def test_session_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("core.mcp.session_state._SESSIONS_BASE", tmp_path)
    record_loaded_tools("sess-1", {"mcp_math_calc", "mcp_web_fetch"})
    assert load_loaded_tools("sess-1") == {"mcp_math_calc", "mcp_web_fetch"}
    assert load_loaded_tools("") == set()


def test_load_persists(tmp_path, monkeypatch):
    monkeypatch.setattr("core.mcp.session_state._SESSIONS_BASE", tmp_path)
    loader = DeferredToolLoader(pool=[_adapter("math", "calc")], session_id="sess-2")
    loader.bind_live_schemas([])
    loader.load(["mcp_math_calc"])
    assert load_loaded_tools("sess-2") == {"mcp_math_calc"}
