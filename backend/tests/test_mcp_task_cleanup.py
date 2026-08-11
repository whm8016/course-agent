"""M-49：MCP 连接 task 退出后必须从 ToolRegistry 清除 stale adapter。

根因：``_run_server`` 是每个 server 的专属连接 task，在线注册 adapter 到
``ToolRegistry`` 后阻塞在 ``await conn.shutdown.wait()``。若 server 端断开 /
transport 异常 / task 被取消，task 自行退出但旧实现只清 ``conn.session``，adapter
仍留在 registry → agent 看到一个调用即失败的"幽灵工具"。

修复把清理逻辑抽成 ``_teardown_adapters_on_exit(conn, graceful=...)``，由
``_run_server`` 的 finally 统一调用。这里覆盖三种退出路径的状态机：

  - 主动断开（graceful=True，_disconnect 已反注册）→ 跳过，无副作用
  - transport 异常（graceful=False）→ 反注册 + 清 adapters + status→error
  - task 被取消（graceful=False）→ 同上

外加幂等性（多次调用不报错）和「连接失败时 registry 本就干净」的回归。
"""
import asyncio

import pytest

from core.agent.registry import get_tool_registry
from core.mcp.adapter import MCPToolAdapter, wrapped_tool_name
from core.mcp.config import MCPServerConfig
from core.mcp.manager import MCPConnectionManager, _ServerConnection


@pytest.fixture(autouse=True)
def _clean_registry():
    """每测后清掉全局 ToolRegistry 里残留的 mcp_* 工具。"""
    yield
    reg = get_tool_registry()
    for name in [n for n in reg.names() if n.startswith("mcp_")]:
        reg.unregister(name)


def _cfg() -> MCPServerConfig:
    return MCPServerConfig(url="https://example.invalid/mcp")


def _make_adapters(server: str, tools: list[str]) -> list[MCPToolAdapter]:
    return [
        MCPToolAdapter(
            server_name=server,
            original_name=t,
            description=t,
            input_schema=None,
            tool_timeout=5,
        )
        for t in tools
    ]


def _registered_conn(server: str, tools: list[str], status: str = "connected"):
    """构造一个已注册 adapter 的 manager+conn，模拟 _connect 成功后的状态。"""
    mgr = MCPConnectionManager()
    conn = _ServerConnection(name=server, config=_cfg(), signature="x")
    conn.status = status
    conn.adapters = _make_adapters(server, tools)
    mgr._connections[server] = conn
    mgr._register_adapters(conn)
    return mgr, conn


# ── 路径 1：主动断开（graceful=True）→ 跳过，由 _disconnect 自己清 ──────────

def test_disconnect_unregisters_adapters():
    """_disconnect 主动断开：反注册 + 清 adapters + status=disabled。"""
    async def run():
        mgr, conn = await asyncio.to_thread(_registered_conn, "srv", ["ping"])
        reg = get_tool_registry()
        assert reg.get(wrapped_tool_name("srv", "ping")) is not None

        await mgr._disconnect(conn)

        assert reg.get(wrapped_tool_name("srv", "ping")) is None
        assert conn.adapters == []
        assert conn.status == "disabled"
        assert conn.session is None

    asyncio.run(run())


def test_teardown_graceful_is_noop():
    """graceful=True：_teardown_adapters_on_exit 不动 adapter（_disconnect 已清）。

    模拟 _disconnect 已经反注册完后 task 才退出（shutdown 已 set），此时 finally
    看到 graceful=True 应完全跳过，避免重复操作/竞争。
    """
    mgr, conn = _registered_conn("srv", ["ping"])
    reg = get_tool_registry()
    assert reg.get(wrapped_tool_name("srv", "ping")) is not None

    # _disconnect 已把 adapter 反注册、清空，并 set 了 shutdown event
    conn.adapters = []
    for a in _make_adapters("srv", ["ping"]):  # registry 已清（模拟 _disconnect 跑过）
        reg.unregister(a.wrapped_name)
    conn.shutdown.set()

    mgr._teardown_adapters_on_exit(conn, graceful=True)

    # graceful 路径不触碰 status（_disconnect 已把 status 设为 disabled）
    assert reg.get(wrapped_tool_name("srv", "ping")) is None
    assert conn.adapters == []


# ── 路径 2：transport 异常 / server 断开（graceful=False）→ task 自行反注册 ──

def test_teardown_ungraceful_unregisters_and_marks_error():
    """graceful=False：task 因异常自行退出 → 反注册 + 清 adapters + status→error。"""
    mgr, conn = _registered_conn("srv", ["ping", "pong"], status="connected")
    reg = get_tool_registry()
    assert reg.get(wrapped_tool_name("srv", "ping")) is not None
    assert reg.get(wrapped_tool_name("srv", "pong")) is not None

    # shutdown 未被 set（task 是自己挂的，不是 _disconnect 主动断开）
    assert not conn.shutdown.is_set()

    mgr._teardown_adapters_on_exit(conn, graceful=False)

    assert reg.get(wrapped_tool_name("srv", "ping")) is None, "stale adapter 必须清除"
    assert reg.get(wrapped_tool_name("srv", "pong")) is None
    assert conn.adapters == []
    assert conn.status == "error", "已断开的 server 不应仍显示 connected"
    assert "exited unexpectedly" in conn.error


def test_teardown_ungraceful_preserves_pre_existing_error_status():
    """graceful=False 但 status 已是 error（except 块先设过）→ 不覆盖 error 消息。"""
    mgr, conn = _registered_conn("srv", ["ping"], status="error")
    conn.error = "ConnectionResetError: server gone"
    reg = get_tool_registry()

    mgr._teardown_adapters_on_exit(conn, graceful=False)

    assert reg.get(wrapped_tool_name("srv", "ping")) is None
    assert conn.adapters == []
    assert conn.status == "error"
    assert conn.error == "ConnectionResetError: server gone", "保留 except 块记的真实原因"


# ── 路径 3：task 被取消（CancelledError → finally，graceful=False）──────────

def test_teardown_after_cancel_unregisters():
    """task 被 cancel()：CancelledError 不被 except Exception 捕获 → 直达 finally，
    graceful=False（shutdown 未 set）→ 走反注册分支。这里直接验证 helper 语义。"""
    mgr, conn = _registered_conn("srv", ["a", "b", "c"], status="connected")
    reg = get_tool_registry()
    for t in ("a", "b", "c"):
        assert reg.get(wrapped_tool_name("srv", t)) is not None

    mgr._teardown_adapters_on_exit(conn, graceful=False)

    for t in ("a", "b", "c"):
        assert reg.get(wrapped_tool_name("srv", t)) is None
    assert conn.adapters == []
    assert conn.status == "error"


# ── 幂等性：重复清理不报错 ───────────────────────────────────────────────────

def test_teardown_is_idempotent():
    """unregister 幂等；连续两次 graceful=False 清理不应抛异常。"""
    mgr, conn = _registered_conn("srv", ["ping"])
    mgr._teardown_adapters_on_exit(conn, graceful=False)
    # 第二次：adapters 已空，unregister 遍历空 list，无副作用无异常
    mgr._teardown_adapters_on_exit(conn, graceful=False)
    reg = get_tool_registry()
    assert reg.get(wrapped_tool_name("srv", "ping")) is None
    assert conn.adapters == []


# ── 连接失败时 registry 本就干净（回归：不在 finally 制造噪音）──────────────

def test_teardown_on_never_registered_conn_is_safe():
    """连接失败（ready 抛异常）时 _connect 不会 _register_adapters，registry 干净。
    此时 task finally 仍会 graceful=False 调 _teardown_adapters_on_exit——遍历
    conn.adapters（task 在 stack 内已 set 但未注册）调 unregister，因 pop 幂等无异常。"""
    mgr = MCPConnectionManager()
    conn = _ServerConnection(name="srv", config=_cfg(), signature="x")
    conn.status = "connecting"
    # task 在 stack 内塞了 adapters 但 _connect 因 ready 异常没注册进 registry
    conn.adapters = _make_adapters("srv", ["ghost"])
    mgr._connections["srv"] = conn

    reg = get_tool_registry()
    assert reg.get(wrapped_tool_name("srv", "ghost")) is None, "预置：从未注册"

    mgr._teardown_adapters_on_exit(conn, graceful=False)

    assert reg.get(wrapped_tool_name("srv", "ghost")) is None
    assert conn.adapters == []


# ── 集成层：_run_server 的 finally 真的调用了清理（需 mcp 包）────────────────

def test_run_server_finally_cleans_on_cancel(monkeypatch):
    """端到端：_run_server 在 task 被 cancel 后，通过 finally → _teardown 触发清理。

    需要 mcp 包；venv 未装时跳过（单元层测试已覆盖清理语义）。
    """
    pytest.importorskip("mcp")
    from types import SimpleNamespace
    import mcp

    class _FakeSession:
        # ClientSession(read, write) 由 manager 传两个参数构造，_FakeSession 须接收。
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            ns = SimpleNamespace(name="ping", description="p", inputSchema=None)
            return SimpleNamespace(tools=[ns])

    async def fake_open_transport(stack, cfg):
        return object(), object()

    monkeypatch.setattr(mcp, "ClientSession", _FakeSession, raising=False)
    monkeypatch.setattr(
        MCPConnectionManager, "_open_transport", staticmethod(fake_open_transport)
    )

    async def run():
        mgr = MCPConnectionManager()
        conn = _ServerConnection(name="srv", config=_cfg(), signature="x")
        mgr._connections["srv"] = conn
        ready = asyncio.get_running_loop().create_future()
        conn.task = asyncio.create_task(mgr._run_server(conn, ready))

        await asyncio.wait_for(ready, timeout=5)
        conn.status = "connected"
        mgr._register_adapters(conn)

        reg = get_tool_registry()
        assert reg.get(wrapped_tool_name("srv", "ping")) is not None

        conn.task.cancel()
        try:
            await conn.task
        except (asyncio.CancelledError, Exception):
            pass

        assert reg.get(wrapped_tool_name("srv", "ping")) is None, \
            "_run_server 取消后 finally 必须清除 stale adapter"
        assert conn.adapters == []
        assert conn.status == "error"

    asyncio.run(run())
