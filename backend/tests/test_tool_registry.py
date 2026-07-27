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


def test_execute_business_param_named_name_no_clash(fresh_registry):
    """回归：工具名形参不得与 read_skill 的 ``name`` 业务参数撞名。

    历史 bug：``execute_tool`` / ``ToolRegistry.execute`` 的工具名形参曾叫 ``name``，
    而 read_skill 的业务参数也叫 ``name``（技能名）。dispatch 实际调用
    ``execute("read_skill", name="aihot")`` 时，位置实参 "read_skill" 与关键字实参
    name="aihot" 都绑向同名形参 → ``got multiple values for argument 'name'``。
    工具名形参改名为 ``tool_name`` 后此处必须通过。
    """

    async def fake_read_skill(*, course_id="", user_id="", name="", **kw):
        return ToolResult(content=f"skill:{name}")

    fresh_registry.register(ToolEntry(name="read_skill", schema={}, executor=fake_read_skill))

    async def run():
        # 模拟 dispatch_tool_calls 真实调用形态：工具名走位置，业务 name 走关键字
        r = await fresh_registry.execute("read_skill", course_id="c", user_id="u", name="aihot")
        assert r.success is True
        assert r.content == "skill:aihot"

    asyncio.run(run())


def _stub_skill_service(tmp_path, monkeypatch):
    """建一个仅含 demo skill 的 SkillService，并 patch get_skill_service 返回它。"""
    import core.skills.skill_service as ss_mod
    from core.skills.skill_service import SkillService

    builtin = tmp_path / "builtin"
    (builtin / "demo").mkdir(parents=True)
    (builtin / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: d\n---\n这是完整正文", encoding="utf-8"
    )
    svc = SkillService(user_root=tmp_path / "user", builtin_root=builtin)
    monkeypatch.setattr(ss_mod, "get_skill_service", lambda *a, **k: svc)
    return svc


def test_read_skill_dedup_within_turn(tmp_path, monkeypatch):
    """同 turn 内重复 read_skill(demo)：首次返回全文，第二次返回「已加载过」不塞全文。"""
    from core.agent.tool_registry import _execute_read_skill
    from core.agentic.dynamic_tools import reset_read_skill_log, set_read_skill_log

    _stub_skill_service(tmp_path, monkeypatch)

    token = set_read_skill_log()
    try:
        async def run():
            r1 = await _execute_read_skill(course_id="c", name="demo")
            r2 = await _execute_read_skill(course_id="c", name="demo")
            return r1, r2
        r1, r2 = asyncio.run(run())
    finally:
        reset_read_skill_log(token)

    assert r1.success and "这是完整正文" in r1.content
    assert r2.success and "已读取过" in r2.content
    assert "这是完整正文" not in r2.content   # 第二次不重复塞全文


def test_read_skill_dedup_different_file_not_collapsed(tmp_path, monkeypatch):
    """同一 skill 不同文件 → key 不同，互不去重（读 SKILL.md 不影响读 references/x.md）。"""
    from core.agent.tool_registry import _execute_read_skill
    from core.agentic.dynamic_tools import reset_read_skill_log, set_read_skill_log

    _stub_skill_service(tmp_path, monkeypatch)
    # 参考文件写在 builtin 层（_stub 里 builtin_root=tmp_path/builtin）
    (tmp_path / "builtin" / "demo" / "references").mkdir()
    (tmp_path / "builtin" / "demo" / "references" / "extra.md").write_text(
        "参考细节", encoding="utf-8"
    )

    token = set_read_skill_log()
    try:
        async def run():
            r1 = await _execute_read_skill(course_id="c", name="demo")                    # SKILL.md
            r2 = await _execute_read_skill(course_id="c", name="demo", file="references/extra.md")
            return r1, r2
        r1, r2 = asyncio.run(run())
    finally:
        reset_read_skill_log(token)

    assert "这是完整正文" in r1.content
    assert "参考细节" in r2.content   # 不同文件，各自返回全文


def test_read_skill_no_dedup_without_log(tmp_path, monkeypatch):
    """未注入去重 log（直调/单测场景）：不去重，两次都返回全文（向后兼容）。"""
    from core.agent.tool_registry import _execute_read_skill

    _stub_skill_service(tmp_path, monkeypatch)

    async def run():
        r1 = await _execute_read_skill(course_id="c", name="demo")
        r2 = await _execute_read_skill(course_id="c", name="demo")
        return r1, r2
    r1, r2 = asyncio.run(run())
    assert "这是完整正文" in r1.content
    assert "这是完整正文" in r2.content   # 都返回全文，未去重
