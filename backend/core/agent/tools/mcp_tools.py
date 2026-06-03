"""MCP (Model Context Protocol) 工具集成。

支持通过配置连接外部 MCP server（stdio / SSE / HTTP），
并将其工具注册为 LangChain tools 供 agent 使用。

配置（.env 或 config.py）：
  MCP_SERVERS='[{"name":"math","transport":"stdio","command":"npx","args":["@mcp/math-server"]}]'

参考 MathClaw mathclaw/agent/tools/mcp.py 的设计，简化适配 FastAPI。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

_MCP_SERVERS_RAW = os.getenv("MCP_SERVERS", "")


def _parse_mcp_config() -> list[dict]:
    if not _MCP_SERVERS_RAW.strip():
        return []
    try:
        servers = json.loads(_MCP_SERVERS_RAW)
        return servers if isinstance(servers, list) else []
    except json.JSONDecodeError:
        logger.warning("MCP_SERVERS env is not valid JSON")
        return []


class MCPClientWrapper:
    """轻量 MCP 客户端封装，支持 stdio 和 HTTP/SSE 两种 transport。"""

    def __init__(self, server_config: dict):
        self.name = server_config.get("name", "mcp")
        self.transport = server_config.get("transport", "stdio")
        self.command = server_config.get("command", "")
        self.args = server_config.get("args", [])
        self.url = server_config.get("url", "")
        self._client = None
        self._tools_cache: list[dict] | None = None

    async def connect(self) -> bool:
        """尝试连接 MCP server。"""
        try:
            if self.transport == "stdio":
                return await self._connect_stdio()
            elif self.transport in ("sse", "http"):
                return await self._connect_http()
            return False
        except Exception as e:
            logger.warning("MCP connect failed server=%s: %s", self.name, e)
            return False

    async def _connect_stdio(self) -> bool:
        """通过 stdio 连接 MCP server。"""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=self.command, args=self.args)
            self._stdio_ctx = stdio_client(params)
            streams = await self._stdio_ctx.__aenter__()
            self._client = ClientSession(*streams)
            await self._client.__aenter__()
            await self._client.initialize()
            return True
        except ImportError:
            logger.info("mcp package not installed, MCP tools disabled")
            return False
        except Exception as e:
            logger.warning("MCP stdio connect failed: %s", e)
            return False

    async def _connect_http(self) -> bool:
        """通过 HTTP/SSE 连接 MCP server。"""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            self._sse_ctx = sse_client(self.url)
            streams = await self._sse_ctx.__aenter__()
            self._client = ClientSession(*streams)
            await self._client.__aenter__()
            await self._client.initialize()
            return True
        except ImportError:
            logger.info("mcp package not installed, MCP tools disabled")
            return False
        except Exception as e:
            logger.warning("MCP HTTP/SSE connect failed: %s", e)
            return False

    async def list_tools(self) -> list[dict]:
        """列出 MCP server 提供的工具。"""
        if self._tools_cache is not None:
            return self._tools_cache
        if not self._client:
            return []
        try:
            result = await self._client.list_tools()
            self._tools_cache = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {},
                }
                for t in result.tools
            ]
            return self._tools_cache
        except Exception as e:
            logger.warning("MCP list_tools failed: %s", e)
            return []

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用 MCP 工具。"""
        if not self._client:
            return json.dumps({"error": "MCP not connected"})
        try:
            result = await self._client.call_tool(tool_name, arguments)
            if result.content:
                texts = [c.text for c in result.content if hasattr(c, "text")]
                return "\n".join(texts) if texts else str(result.content)
            return ""
        except Exception as e:
            return json.dumps({"error": f"MCP call failed: {e}"})

    async def close(self):
        """关闭连接。"""
        try:
            if self._client:
                await self._client.__aexit__(None, None, None)
            if hasattr(self, "_stdio_ctx"):
                await self._stdio_ctx.__aexit__(None, None, None)
            if hasattr(self, "_sse_ctx"):
                await self._sse_ctx.__aexit__(None, None, None)
        except Exception:
            pass


_mcp_clients: list[MCPClientWrapper] = []
_mcp_tools: list[StructuredTool] = []
_initialized = False


async def initialize_mcp_tools() -> list[StructuredTool]:
    """初始化所有配置的 MCP server 并收集工具。"""
    global _initialized, _mcp_tools

    if _initialized:
        return _mcp_tools

    configs = _parse_mcp_config()
    if not configs:
        _initialized = True
        return []

    for cfg in configs:
        client = MCPClientWrapper(cfg)
        connected = await client.connect()
        if not connected:
            continue
        _mcp_clients.append(client)

        tools = await client.list_tools()
        for tool_def in tools:
            mcp_tool = _create_langchain_tool(client, tool_def)
            _mcp_tools.append(mcp_tool)

    _initialized = True
    logger.info("MCP initialized: %d servers, %d tools", len(_mcp_clients), len(_mcp_tools))
    return _mcp_tools


def _create_langchain_tool(client: MCPClientWrapper, tool_def: dict) -> StructuredTool:
    """将 MCP 工具定义转换为 LangChain StructuredTool。"""
    name = f"mcp_{client.name}_{tool_def['name']}"
    description = tool_def.get("description") or f"MCP tool: {tool_def['name']}"

    async def _invoke(**kwargs) -> str:
        return await client.call_tool(tool_def["name"], kwargs)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=name,
        description=description,
    )


async def get_mcp_tools() -> list[StructuredTool]:
    """获取所有 MCP 工具（懒初始化）。"""
    if not _initialized:
        return await initialize_mcp_tools()
    return _mcp_tools


async def close_mcp_clients():
    """关闭所有 MCP 连接。"""
    for client in _mcp_clients:
        await client.close()
    _mcp_clients.clear()
    _mcp_tools.clear()
