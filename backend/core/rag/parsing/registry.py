"""解析引擎注册表（dict 惰性工厂，同构 chunking/registry 与 rag/registry）。

core 不在此 import 重依赖；引擎模块 lazy 加载（首次 get_engine 才 import），
缺依赖仅 warning 不阻断默认链路。默认 mineru_api。借鉴 DeepTutor ``factory.py``。
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# name → 零参 loader（内部 lazy import，注册表本身导入永远不失败）
_ENGINE_LOADERS: dict[str, Callable[[], Any]] = {
    "mineru_api": lambda: _load(
        "core.rag.parsing.engines.mineru_api", "MinerUApiEngine"
    ),
    "docling": lambda: _load(
        "core.rag.parsing.engines.docling", "DoclingEngine"
    ),
}

DEFAULT_ENGINE = "mineru_api"


def _load(module: str, name: str) -> Any:
    try:
        mod = importlib.import_module(module)
        return getattr(mod, name)()
    except ImportError as exc:
        logger.warning("解析引擎 %s 加载失败（依赖缺失？）: %s", module, exc)
        raise


def _normalize(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_").replace(" ", "_")


def get_engine(name: str | None = None) -> Any:
    """取引擎实例。None/未知名 → 默认 mineru_api（未知名带 warning）。"""
    n = _normalize(name or DEFAULT_ENGINE)
    loader = _ENGINE_LOADERS.get(n)
    if loader is None:
        if n and n != DEFAULT_ENGINE:
            logger.warning("解析引擎 '%s' 未注册，回退默认 '%s'", name, DEFAULT_ENGINE)
        n = DEFAULT_ENGINE
        loader = _ENGINE_LOADERS[n]
    return loader()


def is_engine_available(name: str) -> bool:
    """引擎是否可导入（不验证 api_key，那是 is_ready 的事）。"""
    try:
        eng = get_engine(name)
        return hasattr(eng, "is_available") and eng.is_available()
    except Exception:
        return False


__all__ = ["get_engine", "is_engine_available", "DEFAULT_ENGINE"]
