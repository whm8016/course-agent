"""MCP server 配置管理 REST API（admin only，部署级）。

复用 core/mcp 内核（config load/save + manager reload/status + probe_server），
路由层只做权限校验与配置聚合。

- GET    /api/mcp/servers              -> 列出 server 状态（连接/工具）+ 当前配置
- POST   /api/mcp/servers/{name}       -> 新增/更新 server（save + reload）
- DELETE /api/mcp/servers/{name}       -> 删除 server（save + reload）
- POST   /api/mcp/servers/{name}/test  -> 测试连接（probe_server，不持久化）
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user, require_admin
from core.db.database import get_db, UserMCPEnrollment
from core.mcp.config import MCPServerConfig, load_mcp_config, save_mcp_config
from core.mcp.manager import get_mcp_manager, probe_server

router = APIRouter(prefix="/mcp")


class ServerConfigRequest(BaseModel):
    type: str | None = None
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str = ""
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    tool_timeout: int = 30
    connect_timeout: int = 15
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])
    enabled: bool = True


async def _upsert(name: str, payload: ServerConfigRequest) -> dict:
    cfg = load_mcp_config()
    cfg.servers[name] = MCPServerConfig(**payload.model_dump())
    save_mcp_config(cfg)
    await get_mcp_manager().reload()
    return {"name": name, "saved": True}


@router.get("/servers")
async def list_servers(_: dict = Depends(require_admin)):
    cfg = load_mcp_config()
    # 管理页每次拉列表都 reload 当前 worker，避免 save 落在别的 worker 后 UI 仍显示旧 error
    await get_mcp_manager().reload()
    status_map = {row["name"]: row for row in get_mcp_manager().status()}
    servers: list[dict] = []
    for name, server_cfg in cfg.servers.items():
        row = status_map.get(name)
        if row is not None:
            servers.append(row)
        elif not server_cfg.enabled:
            servers.append(
                {
                    "name": name,
                    "transport": server_cfg.resolved_type() or "",
                    "status": "disabled",
                    "error": "",
                    "tools": [],
                }
            )
        else:
            servers.append(
                {
                    "name": name,
                    "transport": server_cfg.resolved_type() or "",
                    "status": "connecting",
                    "error": "",
                    "tools": [],
                }
            )
    return {
        "servers": servers,
        # 前端期望 config 为 { serverName: MCPServerConfig }，不是 MCPConfig 整包
        "config": {
            name: server_cfg.model_dump(mode="json")
            for name, server_cfg in cfg.servers.items()
        },
    }


@router.post("/servers/{name}")
async def upsert_server(
    name: str, payload: ServerConfigRequest, _: dict = Depends(require_admin)
):
    return await _upsert(name, payload)


@router.delete("/servers/{name}")
async def delete_server(name: str, _: dict = Depends(require_admin)):
    cfg = load_mcp_config()
    if name not in cfg.servers:
        raise HTTPException(status_code=404, detail="server not found")
    del cfg.servers[name]
    save_mcp_config(cfg)
    await get_mcp_manager().reload()
    return {"deleted": True}


@router.post("/servers/{name}/test")
async def test_server(name: str, _: dict = Depends(require_admin)):
    cfg = load_mcp_config()
    if name not in cfg.servers:
        raise HTTPException(status_code=404, detail="server not found")
    return await probe_server(cfg.servers[name])


@router.post("/probe")
async def probe(payload: ServerConfigRequest, _: dict = Depends(require_admin)):
    """直接测试一份配置（不持久化），供设置页 Test 按钮在保存前验证。"""
    return await probe_server(MCPServerConfig(**payload.model_dump()))


# --- 个人启用开关（所有登录用户；server 进程系统级共享，仅控可见性）---

class UserMcpEnabledRequest(BaseModel):
    enabled_servers: list[str] = Field(default_factory=list)


@router.get("/servers/catalog")
async def list_servers_catalog(user: dict = Depends(get_current_user)):
    """只读列出系统 MCP server 目录 + 连接状态（所有用户，供勾选启用）。"""
    cfg = load_mcp_config()
    status = {row["name"]: row for row in get_mcp_manager().status()}
    return {
        "servers": [
            {
                "name": name,
                "transport": server_cfg.resolved_type() or "",
                "enabled_globally": server_cfg.enabled,
                "connected": status.get(name, {}).get("status") == "connected",
                "tools": status.get(name, {}).get("tools", []),
            }
            for name, server_cfg in cfg.servers.items()
        ]
    }


@router.get("/me/enabled")
async def get_my_enabled(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """当前用户启用的 MCP server 列表。"""
    uid = str(user.get("id") or "")
    result = await db.execute(
        select(UserMCPEnrollment.server_name)
        .where(UserMCPEnrollment.user_id == uid)
        .where(UserMCPEnrollment.enabled.is_(True))
    )
    return {"enabled_servers": [r[0] for r in result.all()]}


@router.put("/me/enabled")
async def set_my_enabled(
    payload: UserMcpEnabledRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """全量替换当前用户启用的 MCP server 列表（仅限系统已配置的 server）。"""
    uid = str(user.get("id") or "")
    valid = set(load_mcp_config().servers.keys())
    wanted = [s for s in payload.enabled_servers if s in valid]
    await db.execute(delete(UserMCPEnrollment).where(UserMCPEnrollment.user_id == uid))
    for name in wanted:
        db.add(UserMCPEnrollment(user_id=uid, server_name=name, enabled=True))
    await db.flush()
    return {"enabled_servers": wanted}
