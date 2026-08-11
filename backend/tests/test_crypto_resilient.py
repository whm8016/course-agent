"""1.5 part2 回归：decrypt_secret_or_plain 的 legacy 明文兼容（无数据迁移平滑升级）。

UserSearchConfig.api_key 从明文落库迁到加密落库：旧明文行 decrypt 失败返回空，
本函数回退原值（明文）；新加密行正常解密。用户下次 PUT 即加密落库。
"""
from utils.crypto import decrypt_secret, decrypt_secret_or_plain, encrypt_secret


def test_roundtrip_returns_plaintext():
    plain = "sk-search-key-12345"
    token = encrypt_secret(plain)
    assert token and token != plain  # 确实加密了
    assert decrypt_secret_or_plain(token) == plain


def test_legacy_plaintext_passthrough():
    """存量明文 api_key（非 Fernet token）原样返回，不丢。"""
    plain = "legacy-plain-key"
    assert decrypt_secret_or_plain(plain) == plain


def test_empty_returns_empty():
    assert decrypt_secret_or_plain("") == ""
    assert decrypt_secret_or_plain(None) == ""  # type: ignore[arg-type]


def test_decrypt_secret_returns_empty_on_plaintext():
    """底层 decrypt_secret 对明文返回空（解密失败）--这是 resilient 版存在的理由。"""
    assert decrypt_secret("not-a-fernet-token") == ""
