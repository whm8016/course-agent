"""MCP (Model Context Protocol) 工具集成。

连接外部 MCP server（stdio / sse / streamableHttp），发现其工具并注册为
OpenAI ``tool_calls`` 工具供 agent loop 使用。配置部署级（``data/mcp.json``）。

取代早期基于 LangChain ``StructuredTool`` 的实现（那套与 agent loop 的 OpenAI
tool_calls 不兼容、全项目 0 引用，已删除），改为产出 OpenAI function schema +
``MCPConnectionManager`` 单例管理连接，adapter 经 ``_register_adapters`` 注册进
``ToolRegistry``，执行经 ``registry.execute`` 统一路由到 ``call_tool``。
"""
from core.mcp.adapter import MCPToolAdapter, wrapped_tool_name
from core.mcp.config import (
    MCPConfig,
    MCPServerConfig,
    load_mcp_config,
    save_mcp_config,
)
from core.mcp.manager import (
    MCPConnectionManager,
    get_mcp_manager,
    probe_server,
)

__all__ = [
    "MCPConfig",
    "MCPServerConfig",
    "MCPConnectionManager",
    "MCPToolAdapter",
    "get_mcp_manager",
    "probe_server",
    "wrapped_tool_name",
    "load_mcp_config",
    "save_mcp_config",
]
