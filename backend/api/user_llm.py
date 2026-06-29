"""用户级 LLM provider REST API。

路由前缀 /llm/me —— 当前用户的 provider 配置（多租户隔离）。
所有端点需认证（get_current_user）。api_key 在 GET 时脱敏（仅返回 is_set bool）。

端点：
- GET    /llm/me         → 当前用户 provider（key 脱敏）
- PUT    /llm/me         → upsert（覆盖或新增）
- DELETE /llm/me         → 删除（回退平台默认）
- POST   /llm/me/test    → 测试连通性（用刚填配置发极简 completion）
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import get_current_user
from core.db.user_llm_provider import (
    UserProviderPayload,
    delete_provider,
    get_provider_admin_view,
    upsert_provider,
)
from core.llm.provider_factory import clear_llm_client_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm/me", tags=["user-llm-provider"])


class UserProviderResponse(BaseModel):
    """用户端响应：api_key 脱敏为 is_set。"""

    binding: str = ""
    api_key_set: bool = False  # True=已设置，不回传明文
    base_url: str = ""
    api_version: str = ""
    text_model: str = ""
    fast_model: str = ""
    vision_model: str = ""


@router.get("", response_model=UserProviderResponse)
async def get_my_provider(user: dict = Depends(get_current_user)) -> UserProviderResponse:
    """获取当前用户的 LLM provider 配置（api_key 脱敏）。"""
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="未认证")
    view = await get_provider_admin_view(user_id)
    if not view:
        return UserProviderResponse()
    return UserProviderResponse(
        binding=view.get("binding", ""),
        api_key_set=bool(view.get("api_key")),
        base_url=view.get("base_url", ""),
        api_version=view.get("api_version", ""),
        text_model=view.get("text_model", ""),
        fast_model=view.get("fast_model", ""),
        vision_model=view.get("vision_model", ""),
    )


@router.put("")
async def upsert_my_provider(
    payload: UserProviderPayload, user: dict = Depends(get_current_user)
) -> dict:
    """新增或更新当前用户的 LLM provider 配置。"""
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="未认证")
    saved = await upsert_provider(user_id, payload)
    clear_llm_client_cache()  # key 变更后强制重建 client 缓存
    logger.info("user %s upsert provider binding=%s", user_id, payload.binding)
    return {"saved": True, "provider": saved}


@router.delete("")
async def delete_my_provider(user: dict = Depends(get_current_user)) -> dict:
    """删除当前用户的 LLM provider 配置（回退平台默认）。"""
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="未认证")
    deleted = await delete_provider(user_id)
    clear_llm_client_cache()
    logger.info("user %s delete provider deleted=%s", user_id, deleted)
    return {"deleted": deleted}


@router.post("/test")
async def test_my_provider(
    payload: UserProviderPayload, user: dict = Depends(get_current_user)
) -> dict:
    """测试用户提交的 provider 配置连通性（不持久化）。"""
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="未认证")

    # 复用 api/llm.py 的 _probe_profile 逻辑
    binding = (payload.binding or "").strip()
    api_key = (payload.api_key or "").strip()
    base_url = (payload.base_url or "").strip() or None
    api_version = (payload.api_version or "").strip() or None
    model = payload.text_model or ""
    if not model:
        return {"ok": False, "error": "未配置 text_model"}

    try:
        from core.llm.provider_factory import get_llm_client

        client = get_llm_client(
            binding=binding or None,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            model=model,
            timeout=20,
        )
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            stream=False,
        )
        return {"ok": True, "binding": binding or None, "model": model}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}
