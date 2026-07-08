"""MCP 连接管理器（进程级单例）。

``services/mcp/manager.py``：adapter 经 ``_register_adapters``
注册进 ``ToolRegistry``（``_make_mcp_entry`` 包成 ``ToolEntry``，executor 闭包调
``call_tool``），执行经 ``registry.execute`` 统一路由，无 ``mcp_*`` 前缀分支。

生命周期模型
------------
本项目的 chat 以 per-turn task 跑在单个 event loop 内，而 MCP session 必须在
**同一 task** 内开/关（SDK 的 anyio cancel scope 是 task-bound）。因此每个 server
拥有一个专属*连接 task*，端到端持有其 ``AsyncExitStack``::

    connect → 在 task 内 enter transport/session → 发布 adapters →
    等待 shutdown 事件 → 在同一 task 内退出 stack

``ensure_started()`` 懒连接（首 turn 付连接成本，单 server 超时上限 15s），
之后近乎零成本。``reload()`` 对比持久化配置与活动连接，仅重启配置确实变化的
server。
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

from core.mcp.adapter import MCPToolAdapter, wrapped_tool_name
from core.mcp.config import MCPConfig, MCPServerConfig, load_mcp_config
from core.observability import log_flow
from core.observability.metrics import observe_mcp_tool

if TYPE_CHECKING:
    # 仅类型注解用；运行时不 import（避免与 registry 的循环依赖）
    from core.agent.registry import ToolEntry

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_S = 15
# 值得恰好重试一次的瞬态传输错误（对标 nanobot）
_TRANSIENT_ERRORS = (BrokenPipeError, ConnectionResetError)


def _unwrap_exception_group(exc: BaseException) -> str:
    """把 anyio/mcp 抛出的 ExceptionGroup 递归拆开，拿到底层真实错误描述。

    MCP 连接走 anyio TaskGroup，失败会被包成 ``ExceptionGroup``，而
    ``str(ExceptionGroup)`` 只显示 "unhandled errors in a TaskGroup (N
    sub-exception)"，看不到根因（如 httpx 的 410 Gone / ConnectError）。这里
    递归下钻 ``.exceptions``，把叶子异常的类型+消息透出来，让设置页 Test 按钮
    和启动日志能直接看到"为什么连不上"，而不是无信息的 TaskGroup 概要。
    """
    seen: list[str] = []

    def _walk(e: BaseException) -> None:
        if isinstance(e, BaseExceptionGroup):
            for sub in e.exceptions:
                _walk(sub)
            return
        msg = str(e).strip()
        # httpx 的 HTTP 状态错误：追加响应体（如 410 的 "Url is expired"），
        # 比纯 status+url 更让用户直接看懂失败原因
        if isinstance(e, httpx.HTTPStatusError):
            try:
                body = (e.response.text or "").strip().replace("\n", " ")
            except Exception:
                body = ""
            if body:
                msg = f"{msg} | body: {body[:200]}"
        seen.append(f"{type(e).__name__}: {msg}" if msg else type(e).__name__)

    _walk(exc)
    if not seen:
        return f"{type(exc).__name__}: {exc}"
    # 去重保序：嵌套 TaskGroup 可能把同一根因包多层
    uniq: list[str] = []
    for s in seen:
        if s not in uniq:
            uniq.append(s)
    return "; ".join(uniq)


@dataclass
class _ServerConnection:
    """一个已配置 server 的活动状态。"""

    name: str
    config: MCPServerConfig
    signature: str
    status: str = "connecting"  # connecting | connected | error | disabled
    error: str = ""
    adapters: list[MCPToolAdapter] = field(default_factory=list)
    session: Any = None
    task: asyncio.Task | None = None
    shutdown: asyncio.Event = field(default_factory=asyncio.Event)


class MCPConnectionManager:
    """持有所有 MCP server 连接；每进程一个实例。"""

    def __init__(self) -> None:
        self._connections: dict[str, _ServerConnection] = {}
        self._lock = asyncio.Lock()
        self._started = False

    # ── 公共生命周期 ────────────────────────────────────────────────────

    async def ensure_started(self) -> None:
        """连接所有已启用且尚未活动的 server。懒连接：首 turn 付成本。"""
        if self._started:
            return
        async with self._lock:
            if self._started:
                return
            await self._sync_to_config(load_mcp_config())
            self._started = True

    async def reload(self) -> None:
        """重读持久化配置并对活动连接应用 diff。"""
        async with self._lock:
            await self._sync_to_config(load_mcp_config())
            self._started = True

    async def shutdown(self) -> None:
        async with self._lock:
            for conn in list(self._connections.values()):
                await self._disconnect(conn)
            self._connections.clear()
            self._started = False

    # ── 公共查询 ────────────────────────────────────────────────────────

    def status(self) -> list[dict[str, Any]]:
        """连接状态行，供设置 UI。"""
        rows: list[dict[str, Any]] = []
        for name, conn in sorted(self._connections.items()):
            rows.append(
                {
                    "name": name,
                    "transport": conn.config.resolved_type() or "",
                    "status": conn.status,
                    "error": conn.error,
                    "tools": [
                        {"name": a.wrapped_name, "description": a.description}
                        for a in conn.adapters
                    ],
                }
            )
        return rows

    def tool_adapters(self) -> list[MCPToolAdapter]:
        out: list[MCPToolAdapter] = []
        for conn in self._connections.values():
            out.extend(conn.adapters)
        return out

    def tool_adapters_for_user(self, enabled_servers: set[str] | None) -> list[MCPToolAdapter]:
        """按用户启用的 server 集合过滤工具。

        - None → 用户未配置，返回全部（向后兼容，保持原有行为）
        - 含 "*" 的集合 → 视为全部
        - 其他集合 → 仅返回命名在内的 server 的工具
        """
        if enabled_servers is None or "*" in enabled_servers:
            return self.tool_adapters()
        out: list[MCPToolAdapter] = []
        for conn in self._connections.values():
            if conn.name in enabled_servers:
                out.extend(conn.adapters)
        return out

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: int,
    ) -> str:
        """调用已连接 server 上的工具；瞬态错误重试一次。"""
        conn = self._connections.get(server_name)
        if conn is None or conn.session is None or conn.status != "connected":
            return f"(MCP server {server_name!r} is not connected)"
        _t0 = time.perf_counter()
        try:
            result = await self._call_once(conn, tool_name, arguments, timeout)
            _e = int((time.perf_counter() - _t0) * 1000)
            log_flow("mcp.tool_invoke", server=server_name, tool=tool_name,
                     status="ok", elapsed_ms=_e)
            observe_mcp_tool(server=server_name, status="ok", elapsed_ms=_e)
            return result
        except _TRANSIENT_ERRORS:
            logger.warning(
                "MCP tool %s/%s hit a transient transport error; retrying once",
                server_name,
                tool_name,
            )
            try:
                result = await self._call_once(conn, tool_name, arguments, timeout)
                log_flow("mcp.tool_invoke", server=server_name, tool=tool_name,
                         status="ok_retry", elapsed_ms=int((time.perf_counter() - _t0) * 1000))
                return result
            except Exception as exc:
                log_flow("mcp.tool_invoke", level=logging.WARNING,
                         server=server_name, tool=tool_name, status="error_retry",
                         elapsed_ms=int((time.perf_counter() - _t0) * 1000), error=str(exc))
                return f"(MCP tool call failed after retry: {type(exc).__name__})"
        except asyncio.TimeoutError:
            log_flow("mcp.tool_invoke", level=logging.WARNING,
                     server=server_name, tool=tool_name, status="timeout",
                     elapsed_ms=int((time.perf_counter() - _t0) * 1000))
            return f"(MCP tool call timed out after {timeout}s)"
        except asyncio.CancelledError:
            # MCP SDK 的 anyio scope 在内部失败时可能泄漏 CancelledError；
            # 仅当自身 task 被取消时才重新抛出。
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise
            return "(MCP tool call was cancelled)"
        except Exception as exc:
            logger.exception("MCP tool %s/%s failed", server_name, tool_name)
            log_flow("mcp.tool_invoke", level=logging.ERROR,
                     server=server_name, tool=tool_name, status="error",
                     elapsed_ms=int((time.perf_counter() - _t0) * 1000), error=str(exc))
            return f"(MCP tool call failed: {type(exc).__name__}: {exc})"

    @staticmethod
    async def _call_once(
        conn: _ServerConnection,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: int,
    ) -> str:
        result = await asyncio.wait_for(
            conn.session.call_tool(tool_name, arguments=arguments),
            timeout=timeout,
        )
        # duck-typing 提取文本：真实 MCP block 是 types.TextContent（带 .text），
        # 用 getattr 避免对 mcp 包的硬 import，且兼容不同 content block 类型。
        parts: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else str(block))
        return "\n".join(parts) or "(no output)"

    # ── 连接内部 ────────────────────────────────────────────────────────

    async def _sync_to_config(self, config: MCPConfig) -> None:
        """diff 活动连接与 config；调用方持锁。"""
        desired = {name: cfg for name, cfg in config.servers.items() if cfg.enabled}
        # 移除被删/禁用/变更的 server
        for name in list(self._connections):
            cfg = desired.get(name)
            if cfg is None or cfg.connection_signature() != self._connections[name].signature:
                await self._disconnect(self._connections.pop(name))
        # 并发连接新增/变更的 server
        pending = [
            self._connect(name, cfg)
            for name, cfg in desired.items()
            if name not in self._connections
        ]
        if pending:
            await asyncio.gather(*pending)

    async def _connect(self, name: str, cfg: MCPServerConfig) -> None:
        conn = _ServerConnection(
            name=name,
            config=cfg,
            signature=cfg.connection_signature(),
        )
        self._connections[name] = conn
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        conn.task = asyncio.create_task(self._run_server(conn, ready), name=f"mcp-server-{name}")
        try:
            await asyncio.wait_for(ready, timeout=_CONNECT_TIMEOUT_S)
            conn.status = "connected"
            conn.error = ""
            self._register_adapters(conn)
            logger.info("MCP server %r connected (%d tools)", name, len(conn.adapters))
            log_flow("mcp.connect", server=name, tools=len(conn.adapters))
        except asyncio.TimeoutError:
            conn.status = "error"
            conn.error = f"connect timed out after {_CONNECT_TIMEOUT_S}s"
            conn.shutdown.set()
            logger.error("MCP server %r: %s", name, conn.error)
        except Exception as exc:
            conn.status = "error"
            conn.error = _unwrap_exception_group(exc)
            conn.shutdown.set()
            logger.error("MCP server %r failed to connect: %s", name, conn.error)

    async def _run_server(self, conn: _ServerConnection, ready: asyncio.Future) -> None:
        """连接 task：端到端持有一个 server 的 AsyncExitStack。"""
        from mcp import ClientSession

        try:
            async with AsyncExitStack() as stack:
                read, write = await self._open_transport(stack, conn.config)
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listing = await session.list_tools()
                adapters = [
                    MCPToolAdapter(
                        server_name=conn.name,
                        original_name=tool_def.name,
                        description=tool_def.description or "",
                        input_schema=tool_def.inputSchema,
                        tool_timeout=conn.config.tool_timeout,
                    )
                    for tool_def in listing.tools
                    if conn.config.tool_allowed(
                        tool_def.name, wrapped_tool_name(conn.name, tool_def.name)
                    )
                ]
                conn.session = session
                conn.adapters = adapters
                if not ready.done():
                    ready.set_result(None)
                await conn.shutdown.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning("MCP server %r connection task ended: %s", conn.name, exc)
                conn.status = "error"
                conn.error = f"{type(exc).__name__}: {exc}"
        finally:
            # M-49：连接 task 退出时必须把它的 adapter 从 ToolRegistry 摘掉，否则
            # transport 异常断开 / task 被取消后，stale adapter 仍注册在 registry，
            # agent 会看到一个调用即失败的"幽灵工具"。
            #
            # 三种退出路径在此汇合（详见 _teardown_adapters_on_exit）：
            #  1) 主动断开（_disconnect 先 set shutdown event）→ graceful=True，跳过
            #     反注册（_disconnect 已做，幂等重复也无害，跳过避免与它竞争）。
            #  2) transport 异常（server 端断开/网络错）→ except 已记 status=error →
            #     graceful=False，由 task 自己反注册 + 清 adapters。
            #  3) task 被取消（CancelledError 不属 Exception）→ 直接到 finally →
            #     同 2 走 self 清理分支。unregister 幂等，重复调用安全。
            conn.session = None
            self._teardown_adapters_on_exit(conn, graceful=conn.shutdown.is_set())

    def _teardown_adapters_on_exit(
        self, conn: _ServerConnection, *, graceful: bool
    ) -> None:
        """连接 task 退出时清理 adapter 注册（M-49）。

        - ``graceful=True``：主动断开（``_disconnect`` 先 set 了 shutdown event 并
          已自行 ``_unregister_adapters`` + 清 adapters）→ 这里只清 session（已在
          调用方清掉），跳过反注册，避免与 ``_disconnect`` 竞争。
        - ``graceful=False``：transport 异常断开 / task 被取消 → ``_disconnect`` 没
          机会跑 → 由 task 自己反注册 adapter、清空 ``conn.adapters``，并把仍为
          ``connected`` 的 status 收敛到 ``error``（避免 UI 仍显示已断开的 server
          为 connected、agent 仍把它列进可用工具）。

        ``registry.unregister`` 幂等（``pop(name, None)``），故即便 ``_disconnect``
        与本方法先后都跑过也无残留、无异常。
        """
        if graceful:
            return
        self._unregister_adapters(conn)
        conn.adapters = []
        if conn.status == "connected":
            conn.status = "error"
            conn.error = "connection task exited unexpectedly"


    @staticmethod
    async def _open_transport(stack: AsyncExitStack, cfg: MCPServerConfig) -> tuple[Any, Any]:
        """在 stack 上 enter 配置的 transport；返回 (read, write)。"""
        from mcp import StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        transport = cfg.resolved_type()
        if transport == "stdio":
            params = StdioServerParameters(
                command=cfg.command,
                args=list(cfg.args),
                env=dict(cfg.env) or None,
                cwd=cfg.cwd or None,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            return read, write
        if transport == "sse":

            def httpx_client_factory(
                headers: dict[str, str] | None = None,
                timeout: httpx.Timeout | None = None,
                auth: httpx.Auth | None = None,
            ) -> httpx.AsyncClient:
                merged = {**(cfg.headers or {}), **(headers or {})}
                return httpx.AsyncClient(
                    headers=merged or None,
                    follow_redirects=True,
                    timeout=timeout,
                    auth=auth,
                )

            read, write = await stack.enter_async_context(
                sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
            )
            return read, write
        if transport == "streamableHttp":
            # 显式 client，避免 transport 继承 httpx 默认 5s 超时而抢先于 per-tool 超时
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(
                    headers=cfg.headers or None,
                    follow_redirects=True,
                    timeout=httpx.Timeout(60.0, connect=10.0),
                )
            )
            read, write, _ = await stack.enter_async_context(
                streamable_http_client(cfg.url, http_client=http_client)
            )
            return read, write
        raise ValueError(f"MCP server has no usable transport (type={cfg.type!r})")

    async def _disconnect(self, conn: _ServerConnection) -> None:
        self._unregister_adapters(conn)
        conn.shutdown.set()
        if conn.task is not None:
            try:
                await asyncio.wait_for(conn.task, timeout=10)
            except (asyncio.TimeoutError, Exception):
                conn.task.cancel()
        conn.status = "disabled"
        conn.adapters = []

    # ── MCP 工具同步进 ToolRegistry（连接时 register / 断开时 unregister）──

    def _register_adapters(self, conn: _ServerConnection) -> None:
        from core.agent.registry import get_tool_registry
        registry = get_tool_registry()
        for adapter in conn.adapters:
            registry.register(self._make_mcp_entry(adapter))

    def _unregister_adapters(self, conn: _ServerConnection) -> None:
        from core.agent.registry import get_tool_registry
        registry = get_tool_registry()
        for adapter in conn.adapters:
            registry.unregister(adapter.wrapped_name)

    def _make_mcp_entry(self, adapter: MCPToolAdapter) -> "ToolEntry":
        """构造 MCP 工具的 ToolEntry：executor 闭包捕获 self(manager) + adapter，
        经 registry.execute 统一路由到 call_tool。"""
        from core.agent.registry import ToolEntry
        from core.agent.tool_protocol import ToolResult

        manager = self
        server, original, timeout = adapter.server_name, adapter.original_name, adapter.tool_timeout

        async def _exec(*, course_id: str = "", user_id: str = "", **kwargs: Any) -> ToolResult:
            text = await manager.call_tool(server, original, kwargs, timeout=timeout)
            return ToolResult(
                content=text,
                sources=[{"type": "mcp", "server": server, "tool": original}],
            )

        return ToolEntry(
            name=adapter.wrapped_name,
            schema=adapter.to_openai_schema(),
            executor=_exec,
            deferred=True,
        )


async def _http_probe_detail(cfg: MCPServerConfig) -> str:
    """对 http/sse transport 补一次直连，拿 status+响应体。

    mcp 的 sse_client 用 streaming response，抛 ``HTTPStatusError`` 时流已被关闭，
    异常里的 ``response.text`` 读不出（``ResponseNotRead`` / ``StreamClosed``），
    真实失败原因（如 410 的 ``{"message":"Url is expired"}``）会丢。这里独立 GET
    一次把响应体补到错误信息里，让设置页 Test 按钮能直接告诉用户"为什么连不上"。
    只对 sse/streamableHttp 有效；stdio 无 url，跳过。
    """
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                cfg.url,
                headers={"Accept": "text/event-stream", **(cfg.headers or {})},
            )
        body = (r.text or "").strip().replace("\n", " ")
        detail = f" [直连探测: HTTP {r.status_code}"
        if body:
            detail += f" body={body[:200]}"
        return detail + "]"
    except Exception as he:
        return f" [直连探测失败: {type(he).__name__}]"


async def probe_server(
    cfg: MCPServerConfig, *, timeout: int = _CONNECT_TIMEOUT_S
) -> dict[str, Any]:
    """设置页 Test 按钮用：一次性 connect + list_tools，不触碰活动 manager。"""
    from mcp import ClientSession

    async def _probe() -> list[dict[str, str]]:
        async with AsyncExitStack() as stack:
            read, write = await MCPConnectionManager._open_transport(stack, cfg)
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listing = await session.list_tools()
            return [{"name": t.name, "description": t.description or ""} for t in listing.tools]

    try:
        tools = await asyncio.wait_for(_probe(), timeout=timeout)
        return {"ok": True, "tools": tools, "error": ""}
    except asyncio.TimeoutError:
        return {"ok": False, "tools": [], "error": f"connect timed out after {timeout}s"}
    except Exception as exc:
        err = _unwrap_exception_group(exc)
        # http/sse transport：mcp 抛异常时流式 response 已关、body 丢失，补一次直连
        if cfg.resolved_type() in ("sse", "streamableHttp") and cfg.url:
            err += await _http_probe_detail(cfg)
        return {"ok": False, "tools": [], "error": err}


_manager: MCPConnectionManager | None = None


def get_mcp_manager() -> MCPConnectionManager:
    global _manager
    if _manager is None:
        _manager = MCPConnectionManager()
    return _manager


__all__ = [
    "MCPConnectionManager",
    "MCPToolAdapter",
    "get_mcp_manager",
    "probe_server",
    "wrapped_tool_name",
]
