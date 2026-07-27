"""LLM 输出 JSON 提取 —— 容忍 markdown fence / 前后噪声的公共解析入口。

question / research 两条 pipeline 原各持一份逐字相同的 _extract_json（research 的注释
甚至写着「参照 question」），统一到此单一真相源；后续需解析 LLM JSON 输出的 pipeline
直接 import extract_json_from_llm 即可。
"""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json_from_llm(text: str) -> dict[str, Any]:
    """从 LLM 输出提取 JSON 对象，容忍 markdown fence / 前后噪声。"""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start == -1:
        return {}
    try:
        data, _end = json.JSONDecoder().raw_decode(cleaned, start)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
