"""集中式配置入口（pydantic-settings 组合式嵌套，单一事实源）。

用法：
    from settings import get_settings
    s = get_settings()
    s.db.url.get_secret_value()        # SecretStr → .get_secret_value()
    s.llm.text_model
    s.chunking.size
    s.lightrag.safe_top_k_value()      # 计算方法内聚在子组

env 用 <组>__<字段> 分隔式注入：LLM__API_KEY → s.llm.api_key、
DB__URL → s.db.url、SECURITY__JWT_SECRET → s.security.jwt_secret。
"""
from settings.base import BASE_DIR, Settings, get_settings

__all__ = ["Settings", "get_settings", "BASE_DIR"]
