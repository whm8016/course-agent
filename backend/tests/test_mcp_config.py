"""MCP 配置单测：resolved_type 自动检测 / connection_signature 稳定性 /
tool_allowed 白名单 / load-save round-trip / 非法 server 名拒绝。
"""
import json

import pytest

from core.mcp.config import (
    MCPConfig,
    MCPServerConfig,
    load_mcp_config,
    save_mcp_config,
)


def test_resolved_type_auto_detect():
    assert MCPServerConfig(command="npx", args=["x"]).resolved_type() == "stdio"
    assert MCPServerConfig(url="https://h/sse").resolved_type() == "sse"
    assert MCPServerConfig(url="https://h/mcp").resolved_type() == "streamableHttp"
    assert MCPServerConfig(url="https://h/sse/").resolved_type() == "sse"  # 去尾斜杠
    assert MCPServerConfig().resolved_type() is None
    assert MCPServerConfig(type="sse", command="x").resolved_type() == "sse"  # 显式优先


def test_connection_signature_stable():
    cfg = MCPServerConfig(command="npx", args=["a", "b"])
    assert cfg.connection_signature() == cfg.connection_signature()
    assert cfg.connection_signature() == json.dumps(
        cfg.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
    )


def test_tool_allowed_whitelist():
    assert MCPServerConfig(enabled_tools=["*"]).tool_allowed("anything", "mcp_x_anything")
    cfg = MCPServerConfig(enabled_tools=["search", "mcp_x_calc"])
    assert cfg.tool_allowed("search", "mcp_x_search")     # 原名命中
    assert cfg.tool_allowed("calc", "mcp_x_calc")         # wrapped 名命中
    assert not cfg.tool_allowed("delete", "mcp_x_delete")


def test_load_save_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("core.mcp.config.MCP_CONFIG_PATH", tmp_path / "mcp.json")
    cfg = MCPConfig(servers={"math": MCPServerConfig(command="npx", args=["math-server"])})
    save_mcp_config(cfg)
    loaded = load_mcp_config()
    assert "math" in loaded.servers
    assert loaded.servers["math"].command == "npx"


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("core.mcp.config.MCP_CONFIG_PATH", tmp_path / "nope.json")
    assert load_mcp_config().servers == {}


def test_invalid_server_name_rejected():
    with pytest.raises(Exception):
        MCPConfig(servers={"bad name!": MCPServerConfig(command="x")})
