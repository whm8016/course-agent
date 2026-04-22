"""
LLM 返回文本 → dict（替代 DeepTutor parse_json_response）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def parse_json_response(raw: str, logger_instance: logging.Logger | None = None) -> dict[str, Any]:
    """
    从模型输出中解析 JSON 对象；支持 ```json 代码块、首尾杂质。
    """
    log = logger_instance or logger
    if not raw or not str(raw).strip():
        return {}

    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", str(raw).strip())
    block = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if block:
        cleaned = block.group(1).strip()

    try:
        out = json.loads(cleaned)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            out = json.loads(m.group(0))
            return out if isinstance(out, dict) else {}
        except json.JSONDecodeError:
            log.debug("parse_json_response: fallback brace extract failed")
    return {}