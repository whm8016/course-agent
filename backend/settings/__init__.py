"""集中式配置入口。

用法：
    from settings import get_settings
    s = get_settings()
    s.database_url.get_secret_value()   # secrets 用 SecretStr
    s.text_model
    s.chunk_size

config.py 已降级为读取本模块的零破坏 shim（保留全部历史导出名），
存量 `from config import X` 无需改动。
"""
from settings.base import BASE_DIR, Settings, get_settings

__all__ = ["Settings", "get_settings", "BASE_DIR"]
