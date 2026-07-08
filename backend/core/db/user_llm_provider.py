"""用户级 LLM provider 服务层。

职责：
- 查询用户配置 → 解密 api_key → 组装成与 get_llm_client_for_profile 兼容的 profile dict
- upsert/delete（写时加密）
- 管理 admin 视图（含 key 回填）

对话模型与视觉模型可走**不同供应商**（对话 deepseek，视觉 dashscope/qwen-vl），故拆成
两组独立字段。embedding 平台统一（per-course 共享库要求一致），不在此配置。

返回的 profile dict：
    {
        # 对话供应商（catalog 兼容，供 get_llm_client_for_profile）
        "binding": str, "api_key": str, "base_url": str, "api_version": str,
        "models": {"text": {"model": str}},
        # 视觉独立供应商
        "vision_binding": str, "vision_api_key": str, "vision_base_url": str, "vision_model": str,
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
    """PUT /llm/me 请求体（用户自配 LLM provider）。

    对话与视觉可走不同供应商。两把 key 各自留空时保留原值（仅改模型不重输 key）。
    """

    # 对话模型供应商
    binding: str = ""
    api_key: str = ""
    base_url: str = ""
    api_version: str = ""
    text_model: str = ""
    # 视觉模型供应商（独立，可异于对话供应商）
    vision_binding: str = ""
    vision_api_key: str = ""
    vision_base_url: str = ""
    vision_model: str = ""


async def get_active_provider_view(user_id: str) -> dict | None:
    """查询用户活跃 provider，解密 key，返回对话 + 视觉两组供应商信息的 dict。

    Args:
        user_id: 用户 ID（空 → None）

    Returns:
        profile dict 或 None（无记录）。对话组顶层字段 catalog 兼容（直接喂
        get_llm_client_for_profile）；视觉组用 vision_* 前缀字段，由调用方独立构造 client。
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
            vision_key_plain = (
                decrypt_secret(prov.vision_api_key_encrypted)
                if prov.vision_api_key_encrypted
                else ""
            )
            return {
                # 对话供应商
                "binding": prov.binding or "",
                "api_key": api_key_plain,
                "base_url": prov.base_url or "",
                "api_version": prov.api_version or "",
                "models": {"text": {"model": prov.text_model or ""}},
                # 视觉独立供应商
                "vision_binding": prov.vision_binding or "",
                "vision_api_key": vision_key_plain,
                "vision_base_url": prov.vision_base_url or "",
                "vision_model": prov.vision_model or "",
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
                "vision_binding": prov.vision_binding,
                "vision_api_key": decrypt_secret(prov.vision_api_key_encrypted),
                "vision_base_url": prov.vision_base_url,
                "vision_model": prov.vision_model,
                "updated_at": prov.updated_at,
                "created_at": prov.created_at,
            }
    except Exception:
        logger.exception("get_provider_admin_view failed user=%s", user_id)
        return None


def _resolve_key(raw: str, existing_encrypted: str) -> str:
    """key 留空保留原值，否则加密。供对话/视觉两把 key 共用。

    M-45：encrypt_secret 在异常（prod 无主密钥/Fernet 故障）下返回空串。旧实现直接落
    库空串 → 用户以为存了 key，运行时 LLM 调用 401，且静默无提示。现对「非空 raw 却
    加密出空串」显式报错，让 upsert_provider 把它变成 4xx，拒绝落库无效配置。
    """
    raw = (raw or "").strip()
    if existing_encrypted and not raw:
        return existing_encrypted  # 留空保留原 key
    encrypted = encrypt_secret(raw)
    if raw and not encrypted:
        # 加密失败：raw 非空却得到空密文 → 不允许静默存空 key
        raise ValueError("API key 加密失败：检查 PROVIDER_ENCRYPTION_KEY 是否配置")
    return encrypted


async def upsert_provider(user_id: str, payload: UserProviderPayload) -> dict:
    """新增或更新用户 provider（写时加密两把 key，各自留空保留原值）。"""
    async with AsyncSessionLocal() as db:
        row = await db.execute(
            select(UserLLMProvider).where(UserLLMProvider.user_id == user_id)
        )
        prov = row.scalar_one_or_none()
        now = time.time()
        encrypted_key = _resolve_key(
            payload.api_key, prov.api_key_encrypted if prov else ""
        )
        encrypted_vision_key = _resolve_key(
            payload.vision_api_key, prov.vision_api_key_encrypted if prov else ""
        )
        if prov:
            prov.binding = payload.binding
            prov.api_key_encrypted = encrypted_key
            prov.base_url = payload.base_url
            prov.api_version = payload.api_version
            prov.text_model = payload.text_model
            prov.vision_binding = payload.vision_binding
            prov.vision_api_key_encrypted = encrypted_vision_key
            prov.vision_base_url = payload.vision_base_url
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
                vision_binding=payload.vision_binding,
                vision_api_key_encrypted=encrypted_vision_key,
                vision_base_url=payload.vision_base_url,
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
