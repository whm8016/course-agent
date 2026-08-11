"""上传文件命名单一事实源（``{user_id}_{uuid}.ext``）。

``api.upload``（写入侧）与 ``core.storage.gc``（孤儿判定侧）共享本解析器，避免命名格式
漂移时 GC 静默漏判孤儿（api 层另保留 ``_safe_basename`` 做路径遍历防护，那是请求校验
职责，不在此）。
"""
from __future__ import annotations

from pathlib import Path


def parse_upload_owner_id(name: str) -> str | None:
    """``{user_id}_{uuid}.ext`` → user_id 段；无下划线或无法判定 → None。"""
    base = Path(name).name
    if "_" not in base:
        return None
    return base.split("_", 1)[0]


__all__ = ["parse_upload_owner_id"]
