"""联网搜索配置 REST API：admin 默认 + 用户覆盖。

- GET    /api/search_config/providers  -> 引擎列表（全员，下拉用）
- GET    /api/search_config/admin      -> admin 默认配置（admin only，含 key 回填）
- PUT    /api/search_config/admin      -> 保存 admin 默认（admin only）
- GET    /api/search_config/me         -> 当前用户 override（全员，含 key 回填）
- PUT    /api/search_config/me         -> 保存用户 override（全员）
- DELETE /api/search_config/me         -> 清除用户 override（回退 admin 默认）
- POST   /api/search_config/probe      -> 测试连通（给定配置，admin/user 都可）
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user, require_admin
from core.db.database import get_db, UserSearchConfig
from services.search import get_providers_info, probe_search
from services.search.config import load_admin_default, save_admin_default

router = APIRouter(prefix="/search_config")


class SearchConfigPayload(BaseModel):
    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    max_results: int = 0
    proxy: str = ""


class ProbeRequest(BaseModel):
    provider: str = ""
    api_key: str = ""
    base_url: str = ""


def _empty_cfg() -> dict:
    return {"provider": "", "api_key": "", "base_url": "", "max_results": 0, "proxy": ""}


def _override_to_dict(row: UserSearchConfig) -> dict:
    return {
        "provider": row.provider or "",
        "api_key": row.api_key or "",
        "base_url": row.base_url or "",
        "max_results": row.max_results or 0,
        "proxy": row.proxy or "",
    }


@router.get("/providers")
async def list_providers(user: dict = Depends(get_current_user)):
    """列出支持的搜索引擎（全员，下拉用）。"""
    return {"providers": get_providers_info()}


@router.get("/admin")
async def get_admin_config(_: dict = Depends(require_admin)):
    """admin 默认配置（admin only，含 key 回填）。"""
    return load_admin_default() or _empty_cfg()


@router.put("/admin")
async def put_admin_config(payload: SearchConfigPayload, _: dict = Depends(require_admin)):
    """保存 admin 默认到 data/search_config.json（admin only）。"""
    saved = save_admin_default(payload.model_dump())
    return {"saved": True, "config": saved}


@router.get("/me")
async def get_my_config(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """当前用户的搜索 override（全员；无记录返回空配置）。"""
    uid = str(user.get("id") or "")
    row = (
        await db.execute(
            select(UserSearchConfig).where(UserSearchConfig.user_id == uid)
        )
    ).scalar_one_or_none()
    if row is None:
        data = _empty_cfg()
        data["has_override"] = False
        return data
    data = _override_to_dict(row)
    data["has_override"] = True
    return data


@router.put("/me")
async def put_my_config(
    payload: SearchConfigPayload,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """保存当前用户的搜索 override（全员，upsert）。"""
    uid = str(user.get("id") or "")
    row = (
        await db.execute(
            select(UserSearchConfig).where(UserSearchConfig.user_id == uid)
        )
    ).scalar_one_or_none()
    if row is None:
        row = UserSearchConfig(
            user_id=uid,
            provider=payload.provider,
            api_key=payload.api_key,
            base_url=payload.base_url,
            max_results=payload.max_results,
            proxy=payload.proxy,
        )
        db.add(row)
    else:
        row.provider = payload.provider
        row.api_key = payload.api_key
        row.base_url = payload.base_url
        row.max_results = payload.max_results
        row.proxy = payload.proxy
    await db.flush()
    return {"saved": True, "config": _override_to_dict(row)}


@router.delete("/me")
async def delete_my_config(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """清除当前用户的搜索 override（回退 admin 默认）。"""
    uid = str(user.get("id") or "")
    await db.execute(
        delete(UserSearchConfig).where(UserSearchConfig.user_id == uid)
    )
    await db.flush()
    return {"deleted": True}


@router.post("/probe")
async def probe(payload: ProbeRequest, user: dict = Depends(get_current_user)):
    """测试连通（给定 provider+key，不持久化；admin/user 都可）。"""
    return probe_search(payload.provider, payload.api_key, payload.base_url)
