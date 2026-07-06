"""Prompt YAML 加载器。

把各 capability 的提示词从代码外部化到 prompts/{language}/{name}.yaml，
便于对齐 / 迭代提示词，且不重新部署代码。

加载整个 YAML 为 dict，调用方按结构取具体字段（如 data["loop"]["system"]）。
与 core/question/agent_base.py 的 get_prompt 思路一致，但返回完整 dict 以支持
多段提示词（system / user / notices ...），供后续 solve / question / research 复用。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_prompt_dict(yaml_path: str | Path) -> dict[str, Any]:
    """加载 YAML prompt 文件为 dict；文件缺失或格式异常时返回空 dict 并告警。

    不做内存缓存——prompt 文件小，每次直读便于开发期改动即时生效；
    生产若需要可在外层加缓存。
    """
    path = Path(yaml_path)
    if not path.exists():
        logger.warning("prompt file not found: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        logger.exception("failed to load prompt file: %s", path)
        return {}
    if not isinstance(data, dict):
        logger.warning("prompt file is not a mapping: %s", path)
        return {}
    return data
