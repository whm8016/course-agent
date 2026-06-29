"""用户级 LLM provider 服务层。

职责：
- 查询用户配置 → 解密 api_key → 组装成与 get_llm_client_for_profile 兼容的 profile dict
- upsert/delete（写时加密）
- 管理 admin 视图（含 key 回填）

返回的 profile dict 结构与 model_catalog.json 的 profile 一致：
    {
        "binding": str,
        "api_key": str,   # 解密后的明文，供 provider_factory 使用
        "base_url": str,
        "api_version": str,
        "models": {"text": {"model": str}, "fast": {"model": str}, "vision": {"model": str}},
    }
"""
from __future__ import annotations

import logging
import time

from pydantic import BaseModel
from sqlalchemy import delete, select

from core.db.database import AsyncSessionLocal, UserLLMProvider
from utils.crypto import decrypt_secret, encrypt_secret

logger = logging.getLogger(__name__)


class UserProviderPayload(BaseModel):
    """PUT /llm/me 请求体（用户自配 LLM provider）。"""

    binding: str = ""
    api_key: str = ""
    base_url: str = ""
    api_version: str = ""
    text_model: str = ""
    fast_model: str = ""
    vision_model: str = ""


async def get_active_provider_view(user_id: str) -> dict | None:
    """查询用户活跃 provider，解密 key，返回与 catalog profile 兼容的 dict。

    Args:
        user_id: 用户 ID（空 → None）

    Returns:
        profile dict（含 models 子结构）或 None（无记录）

    Note:
        返回的 dict 直接喂给 provider_factory.get_llm_client_for_profile，
        该函数会做指纹缓存 + 回退 .env 兜底。
    """
    if not user_id:
        return None
    try:
        async with AsyncSessionLocal() as db:
            row = await db.execute(
                select(UserLLMProvider).where(UserLLMProvider.user_id == user_id)
            )
            prov = row.scalar_one_or_none()
            if not prov:
                return None

            api_key_plain = decrypt_secret(prov.api_key_encrypted)
            return {
                "binding": prov.binding or "",
                "api_key": api_key_plain,
                "base_url": prov.base_url or "",
                "api_version": prov.api_version or "",
                "models": {
                    "text": {"model": prov.text_model or ""},
                    "fast": {"model": prov.fast_model or ""},
                    "vision": {"model": prov.vision_model or ""},
                },
            }
    except Exception:
        logger.exception("get_active_provider_view failed user=%s", user_id)
        return None


async def get_provider_admin_view(user_id: str) -> dict | None:
    """管理员视图（含 api_key 回填用于编辑页）。用户端视图用 /llm/me（key 脱敏）。"""
    if not user_id:
        return None
    try:
        async with AsyncSessionLocal() as db:
            row = await db.execute(
                select(UserLLMProvider).where(UserLLMProvider.user_id == user_id)
            )
            prov = row.scalar_one_or_none()
            if not prov:
                return None
            return {
                "user_id": prov.user_id,
                "binding": prov.binding,
                "api_key": decrypt_secret(prov.api_key_encrypted),
                "base_url": prov.base_url,
                "api_version": prov.api_version,
                "text_model": prov.text_model,
                "fast_model": prov.fast_model,
                "vision_model": prov.vision_model,
                "updated_at": prov.updated_at,
                "created_at": prov.created_at,
            }
    except Exception:
        logger.exception("get_provider_admin_view failed user=%s", user_id)
        return None


async def upsert_provider(user_id: str, payload: UserProviderPayload) -> dict:
    """新增或更新用户 provider（写时加密 api_key）。"""
    async with AsyncSessionLocal() as db:
        row = await db.execute(
            select(UserLLMProvider).where(UserLLMProvider.user_id == user_id)
        )
        prov = row.scalar_one_or_none()
        now = time.time()
        encrypted_key = encrypt_secret(payload.api_key)
        if prov:
            prov.binding = payload.binding
            prov.api_key_encrypted = encrypted_key
            prov.base_url = payload.base_url
            prov.api_version = payload.api_version
            prov.text_model = payload.text_model
            prov.fast_model = payload.fast_model
            prov.vision_model = payload.vision_model
            prov.updated_at = now
        else:
            prov = UserLLMProvider(
                user_id=user_id,
                binding=payload.binding,
                api_key_encrypted=encrypted_key,
                base_url=payload.base_url,
                api_version=payload.api_version,
                text_model=payload.text_model,
                fast_model=payload.fast_model,
                vision_model=payload.vision_model,
                updated_at=now,
                created_at=now,
            )
            db.add(prov)
        await db.commit()
    return await get_provider_admin_view(user_id) or {}


async def delete_provider(user_id: str) -> bool:
    """删除用户 provider（回退平台默认）。"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            delete(UserLLMProvider).where(UserLLMProvider.user_id == user_id)
        )
        await db.commit()
        return result.rowcount > 0


__all__ = [
    "UserProviderPayload",
    "get_active_provider_view",
    "get_provider_admin_view",
    "upsert_provider",
    "delete_provider",
]