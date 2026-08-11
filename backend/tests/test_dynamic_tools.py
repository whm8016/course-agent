"""DynamicToolResolver 单测：resolve 在 skill/deferred 有无组合下的 schema 集合、
loader 绑定的 list 可变引用（验证 loop 零改动可行性）、contextvar set/reset。
"""
import core.mcp.manager as mcp_manager_mod
from core.agentic.dynamic_tools import (
    LOAD_TOOLS_SCHEMA,
    current_deferred_loader,
    reset_deferred_loader,
    resolve,
)
from core.context import UnifiedContext
from core.mcp.adapter import MCPToolAdapter


def _adapter(server: str, tool: str) -> MCPToolAdapter:
    return MCPToolAdapter(
        server_name=server, original_name=tool, description=tool,
        input_schema=None, tool_timeout=5, deferred=True,
    )


class _FakeMgr:
    def __init__(self, adapters):
        self._adapters = adapters

    def tool_adapters(self):
        return self._adapters

    def tool_adapters_for_user(self, servers=None):
        # 测试场景：忽略 per-user 过滤，返回全部（对标 manager.tool_adapters_for_user）
        return self._adapters


def _inject_manager(adapters):
    mcp_manager_mod._manager = _FakeMgr(adapters)


def test_resolve_no_skill_no_mcp():
    ctx = UnifiedContext(enabled_tools=["rag"])
    schemas, token = resolve(ctx)
    names = {s["function"]["name"] for s in (schemas or [])}
    assert names == {"rag", "read_memory", "write_memory"}  # read_memory/write_memory 为 always_on
    assert token is None


def test_resolve_with_skill_appends_read_skill():
    ctx = UnifiedContext(enabled_tools=["rag"], skills_manifest="## Skills\n- **x** — y")
    schemas, token = resolve(ctx)
    names = {s["function"]["name"] for s in schemas}
    assert "read_skill" in names
    assert "rag" in names
    assert token is None  # skill 不经 deferred contextvar


def test_resolve_with_mcp_deferred(monkeypatch, tmp_path):
    monkeypatch.setattr("core.mcp.session_state._SESSIONS_BASE", tmp_path)
    _inject_manager([_adapter("math", "calc")])
    try:
        ctx = UnifiedContext(enabled_tools=["rag"], session_id="s1")
        schemas, token = resolve(ctx)
        names = {s["function"]["name"] for s in schemas}
        assert "rag" in names
        assert "load_tools" in names
        assert "mcp_math_calc" not in names  # deferred：未 load 不在初始 schema
        assert ctx.extended_tools_manifest.startswith("## 扩展工具")
        assert current_deferred_loader() is not None
        reset_deferred_loader(token)
        assert current_deferred_loader() is None
    finally:
        mcp_manager_mod._manager = None


def test_resolve_load_appends_to_bound_list(monkeypatch, tmp_path):
    """核心可行性验证：resolve 返回的 list 被 loader 绑定，load() 后同引用可见。"""
    monkeypatch.setattr("core.mcp.session_state._SESSIONS_BASE", tmp_path)
    _inject_manager([_adapter("math", "calc")])
    try:
        ctx = UnifiedContext(enabled_tools=[], session_id="s2")
        schemas, token = resolve(ctx)
        loader = current_deferred_loader()
        names = {s["function"]["name"] for s in schemas}
        assert names == {"load_tools", "read_memory", "write_memory"}  # always_on + load_tools
        # 模型调 load_tools → loader 把 calc schema append 到绑定的 schemas
        loader.load(["mcp_math_calc"])
        names_after = {s["function"]["name"] for s in schemas}
        assert "mcp_math_calc" in names_after  # 同引用，已可见
        reset_deferred_loader(token)
    finally:
        mcp_manager_mod._manager = None


def test_resolve_load_tools_schema_shape():
    assert LOAD_TOOLS_SCHEMA["function"]["name"] == "load_tools"
    params = LOAD_TOOLS_SCHEMA["function"]["parameters"]
    assert params["properties"]["names"]["type"] == "array"
    assert "names" in params["required"]


def test_read_skill_log_contextvar_lifecycle():
    """read_skill 去重集合的 set/current/reset 生命周期；默认 None（直调场景向后兼容）。"""
    from core.agentic.dynamic_tools import (
        current_read_skill_log,
        reset_read_skill_log,
        set_read_skill_log,
    )
    assert current_read_skill_log() is None                 # 默认 None
    token = set_read_skill_log()
    log = current_read_skill_log()
    assert isinstance(log, set) and log == set()            # 初始空 set
    log.add(("demo", "SKILL.md"))
    assert ("demo", "SKILL.md") in current_read_skill_log()  # 同一对象可见
    reset_read_skill_log(token)
    assert current_read_skill_log() is None                 # reset 回 None
