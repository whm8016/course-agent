#!/usr/bin/env python
"""JSON extraction from LLM output (triple-quoted string fix from DeepTutor)."""

from __future__ import annotations

import json
import re
from typing import Any


def _escape_triple_quoted_strings(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        content = match.group(1)
        return json.dumps(content)

    pattern = re.compile(r'"""([\s\S]*?)"""')
    return pattern.sub(replacer, text)


def extract_json_from_text(text: str) -> dict[str, Any] | list[Any] | None:
    if not text:
        return None

    text = _escape_triple_quoted_strings(text)

    code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
    match = code_block_pattern.search(text)

    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_obj_pattern = re.compile(r"\{[\s\S]*\}")
    match_obj = json_obj_pattern.search(text)
    if match_obj:
        try:
            return json.loads(match_obj.group(0))
        except json.JSONDecodeError:
            pass

    json_arr_pattern = re.compile(r"\[[\s\S]*\]")
    match_arr = json_arr_pattern.search(text)
    if match_arr:
        try:
            return json.loads(match_arr.group(0))
        except json.JSONDecodeError:
            pass

    return None


def clean_json_string(json_str: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", json_str)


__all__ = ["extract_json_from_text", "clean_json_string"]
