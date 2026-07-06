"""对称加密工具 —— 用于用户级 LLM API key 安全存储。

密钥来源：
- prod: 必须显式设置 PROVIDER_ENCRYPTION_KEY（Fernet key，32字节 base64）
- dev:  未设置时从 JWT_SECRET 派生并告警（确定性，重启仍可解密）

API：
    encrypt_secret(plain: str) -> str   # 返回 Fernet token（或空串）
    decrypt_secret(token: str) -> str   # 返回原文（或空串，失败不抛）

空串不加密，解密失败返回空串（用户级配置错误不阻断服务）。
"""
from __future__ import annotations

import base64
import hashlib
import logging
import warnings

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


def _derive_fernet_key(secret: str) -> bytes:
    """从任意长 secret 派生 32 字节 Fernet key（SHA256 → base64）。"""
    # Fernet 要求 32 字节 URL-safe base64 key
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _get_master_key() -> bytes:
    """获取主密钥：prod 必填，dev 未设置时从 JWT_SECRET 派生并告警。"""
    from settings import get_settings

    s = get_settings()
    enc_key = s.security.provider_encryption_key.get_secret_value()

    if enc_key:
        # 用户已提供 Fernet key（直接用）
        try:
            # Fernet 会校验格式；若非有效 Fernet key 则视为普通 secret 派生
            return base64.urlsafe_b64decode(enc_key.encode()) if len(enc_key) == 44 else _derive_fernet_key(enc_key)
        except Exception:
            return _derive_fernet_key(enc_key)

    # 未设置：dev 从 JWT_SECRET 派生，prod 抛错（但校验已在 Settings._check_prod 完成，此处为兜底）
    if s.is_production:
        raise RuntimeError(
            "PROVIDER_ENCRYPTION_KEY not set in production. "
            "Generate one: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )

    jwt_secret = s.security.jwt_secret.get_secret_value()
    warnings.warn(
        "PROVIDER_ENCRYPTION_KEY not set — deriving from JWT_SECRET (dev only). "
        "Set a dedicated Fernet key for production to avoid key reuse.",
        stacklevel=3,
    )
    return _derive_fernet_key(jwt_secret)


_FERNET: Fernet | None = None


def _fernet() -> Fernet:
    """懒加载 Fernet 单例（首次调用时初始化主密钥）。"""
    global _FERNET
    if _FERNET is None:
        _FERNET = Fernet(_get_master_key())
    return _FERNET


def encrypt_secret(plain: str) -> str:
    """加密敏感字符串。

    Args:
        plain: 明文（空串不加密）

    Returns:
        Fernet token 字串（空串→空串）

    Raises:
        RuntimeError: prod 未设置主密钥
    """
    if not plain:
        return ""
    try:
        return _fernet().encrypt(plain.encode("utf-8")).decode("utf-8")
    except Exception:
        logger.exception("encrypt_secret failed")
        return ""


def decrypt_secret(token: str) -> str:
    """解密 Fernet token。

    Args:
        token: Fernet token（空串→空串）

    Returns:
        明文（解密失败返回空串，不抛）
    """
    if not token:
        return ""
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.warning("decrypt_secret: InvalidToken (可能密钥变更)")
        return ""
    except Exception:
        logger.exception("decrypt_secret failed")
        return ""


__all__ = ["encrypt_secret", "decrypt_secret"]