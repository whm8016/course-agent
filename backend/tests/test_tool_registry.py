"""ToolRegistry 中间层单测：register/get/unregister/幂等、schemas_for 过滤与去重、
execute 命中/未命中/异常兜底、deferred_entries。用独立 ToolRegistry 实例，
不污染全局单例（真实 executor 的路由由 test_solve_session / test_agent_loop 间接覆盖）。
"""
import asyncio

import pytest

from core.agent.registry import (
    ToolEntry,
    ToolRegistry,
    get_tool_registry,
    register_builtins,
)
from core.agent.tool_protocol import ToolResult


@pytest.fixture
def fresh_registry():
    """每测一个干净 registry（独立实例，不污染全局单例）。"""
    r = ToolRegistry()
    register_builtins(r)
    return r


def test_register_and_get(fresh_registry):
    assert fresh_registry.has("rag")
    e = fresh_registry.get("rag")
    assert e.name == "rag"
    assert e.deferred is False
    assert e.schema["function"]["name"] == "rag"
    assert fresh_registry.get("ghost") is None


def test_unregister_idempotent(fresh_registry):
    assert fresh_registry.has("rag")
    fresh_registry.unregister("rag")
    assert not fresh_registry.has("rag")
    fresh_registry.unregister("rag")  # 幂等不抛
    fresh_registry.unregister("ghost")  # 未注册也不抛


def test_register_builtins_idempotent(fresh_registry):
    """同名覆盖：register_builtins 二次调用不翻倍。"""
    n = len(fresh_registry.names())
    register_builtins(fresh_registry)
    assert len(fresh_registry.names()) == n


def test_builtins_registered(fresh_registry):
    names = set(fresh_registry.names())
    assert {
        "rag", "web_search", "ask_user", "read_skill", "load_tools",
        "solve_plan", "solve_finish_step", "solve_replan",
    } <= names


def test_schemas_for_filters(fresh_registry):
    out = fresh_registry.schemas_for(["rag", "web_search"])
    assert {s["function"]["name"] for s in out} == {"rag", "web_search"}
    assert fresh_registry.schemas_for(None) == []
    assert fresh_registry.schemas_for([]) == []
    # 未注册名被忽略
    out2 = fresh_registry.schemas_for(["rag", "ghost"])
    assert {s["function"]["name"] for s in out2} == {"rag"}


def test_schemas_for_dedup(fresh_registry):
    out = fresh_registry.schemas_for(["rag", "rag", "web_search"])
    assert [s["function"]["name"] for s in out] == ["rag", "web_search"]


def test_execute_hit(fresh_registry):
    async def fake_exec(*, course_id="", user_id="", **kw):
        return ToolResult(content=f"ok:{kw.get('q')}/{course_id}")

    fresh_registry.register(ToolEntry(name="probe", schema={}, executor=fake_exec))

    async def run():
        r = await fresh_registry.execute("probe", course_id="c", user_id="u", q="x")
        assert r.content == "ok:x/c"

    asyncio.run(run())


def test_execute_unknown(fresh_registry):
    async def run():
        r = await fresh_registry.execute("ghost", course_id="c")
        assert r.success is False
        assert "未知工具" in r.content

    asyncio.run(run())


def test_execute_exception_caught(fresh_registry):
    """executor 抛异常时 registry 兜底返回失败 ToolResult（不向上抛，保护 dispatch gather）。"""

    async def boom(*, course_id="", user_id="", **kw):
        raise RuntimeError("boom")

    fresh_registry.register(ToolEntry(name="bomb", schema={}, executor=boom))

    async def run():
        r = await fresh_registry.execute("bomb", course_id="c")
        assert r.success is False

    asyncio.run(run())


def test_deferred_entries(fresh_registry):
    # 内置全是 deferred=False
    assert fresh_registry.deferred_entries() == []
    fresh_registry.register(ToolEntry(name="mcp_x", schema={}, executor=boiler, deferred=True))
    assert {e.name for e in fresh_registry.deferred_entries()} == {"mcp_x"}


async def boiler(*, course_id="", user_id="", **kw):
    return ToolResult(content="")


def test_global_singleton_has_builtins():
    """全局单例首次访问时已注册内置工具。"""
    reg = get_tool_registry()
    assert reg.has("rag")
    assert reg.has("solve_plan")
