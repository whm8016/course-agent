"""MCP 配置（部署级）。

Pydantic 模型 + 持久化。配置全局共享（``data/mcp.json``）：所有课程/用户
连接同一组 server。MCP server 是基础设施（数学计算/查词典），非课程内容，
故部署级而非 per-course。
路径从 admin path service 改为项目 data 目录。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from settings import get_settings
MCP_CONFIG_PATH = get_settings().paths.mcp_config_path

_SERVER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

MCP_CONFIG_PATH = Path(MCP_CONFIG_PATH)


class MCPServerConfig(BaseModel):
    """单个 MCP server 配置。

    ``type`` 省略时自动检测：有 ``command`` → stdio；``url`` 以 ``/sse`` 结尾 → sse；
    其他 ``url`` → streamableHttp。
    """

    type: Literal["stdio", "sse", "streamableHttp"] | None = None
    # stdio transport
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str = ""
    # http transports
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    # behaviour
    tool_timeout: int = Field(default=30, ge=1, le=600)
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])
    enabled: bool = True

    @field_validator("command", "url", "cwd", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def resolved_type(self) -> str | None:
        if self.type:
            return self.type
        if self.command:
            return "stdio"
        if self.url:
            return "sse" if self.url.rstrip("/").endswith("/sse") else "streamableHttp"
        return None

    def connection_signature(self) -> str:
        """reload 用来检测配置变更的稳定指纹。"""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)

    def tool_allowed(self, raw_name: str, wrapped_name: str) -> bool:
        allowed = set(self.enabled_tools or ["*"])
        return "*" in allowed or raw_name in allowed or wrapped_name in allowed


class MCPConfig(BaseModel):
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    @field_validator("servers")
    @classmethod
    def _validate_names(cls, value: dict[str, MCPServerConfig]) -> dict[str, MCPServerConfig]:
        for name in value:
            if not _SERVER_NAME_RE.match(name):
                raise ValueError(
                    f"Invalid MCP server name {name!r}: must match "
                    "^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"
                )
        return value


def load_mcp_config() -> MCPConfig:
    if not MCP_CONFIG_PATH.exists():
        return MCPConfig()
    try:
        data = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return MCPConfig()
    try:
        return MCPConfig.model_validate(data)
    except Exception:
        return MCPConfig()


def save_mcp_config(config: MCPConfig) -> None:
    MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MCP_CONFIG_PATH.write_text(
        json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "MCP_CONFIG_PATH",
    "MCPConfig",
    "MCPServerConfig",
    "load_mcp_config",
    "save_mcp_config",
]
