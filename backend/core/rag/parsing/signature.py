"""ParserSignature：解析配置的稳定指纹（缓存键的第二维度）。

只有影响输出字节的字段进 hash（engine / version / 输出旋钮），换 token / api_key
不该让缓存失效（同文件 + 同引擎配置 → 命中）。借鉴 DeepTutor ``signature.py``。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ParserSignature:
    """``(engine, engine_version, output-affecting params)`` 身份。

    ``params`` 是排序后的 (key, value) 元组，保证 hash 与顺序无关。各引擎决定哪些
    旋钮影响输出字节并折叠进来（MinerU 含 model/language/enable_formula/enable_table，
    但永不含 api_token）。
    """

    engine: str
    engine_version: str
    params: tuple[tuple[str, str], ...]

    def hash(self) -> str:
        """短 hex 摘要，用作缓存 signature 目录名。"""
        payload = {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "params": [list(item) for item in self.params],
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def build(
        cls, engine: str, engine_version: str, params: Mapping[str, object]
    ) -> "ParserSignature":
        items = tuple(sorted((str(k), str(v)) for k, v in params.items()))
        return cls(engine=engine, engine_version=engine_version or "", params=items)


__all__ = ["ParserSignature"]
