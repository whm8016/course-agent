"""MCPToolAdapter — MCP 工具适配器（数据载体）。

承载 schema + server/original 元数据，**不继承 BaseTool、不实现 execute()**
（本项目用函数式 ToolRegistry：执行经 ``manager._make_mcp_entry`` 包成 ToolEntry、
``registry.execute`` 路由到 ``manager.call_tool``）。schema 的渐进式揭示（deferred）
由 ``DeferredToolLoader`` 负责，与执行路由分离。

命名 ``mcp_<server>_<tool>``（非法字符替换为 ``_``），因 server/tool 名都可能
含 ``_``，**不能靠名字反解**，registry 按 wrapped_name 查表。

``deferred=True``（对标 DeepTutor）：工具 schema 默认不进 LLM 初始列表，
经 ``load_tools`` 渐进式加载（阶段3 启用）。
"""
from __future__ import annotations

import re
from typing import Any

_NAME_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")


def wrapped_tool_name(server: str, tool: str) -> str:
    """``mcp_<server>_<tool>``，非标识符字符替换为 ``_``。"""
    return f"mcp_{_NAME_SANITIZE_RE.sub('_', server)}_{_NAME_SANITIZE_RE.sub('_', tool)}"


class MCPToolAdapter:
    """一个 MCP server 工具，本项目适配版（数据载体，deferred 默认 True）。"""

    def __init__(
        self,
        *,
        server_name: str,
        original_name: str,
        description: str,
        input_schema: dict[str, Any] | None,
        tool_timeout: int,
        deferred: bool = True,
    ) -> None:
        self.server_name = server_name
        self.original_name = original_name
        self.wrapped_name = wrapped_tool_name(server_name, original_name)
        self.description = description or original_name
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self.tool_timeout = tool_timeout
        self.deferred = deferred

    def to_openai_schema(self) -> dict[str, Any]:
        """对标 DeepTutor ``ToolDefinition.to_openai_schema``：
        inputSchema 透传 + setdefault type/object/properties，直接当 parameters。
        """
        schema = dict(self.input_schema)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return {
            "type": "function",
            "function": {
                "name": self.wrapped_name,
                "description": f"[{self.server_name}] {self.description}",
                "parameters": schema,
            },
        }


__all__ = ["MCPToolAdapter", "wrapped_tool_name"]
