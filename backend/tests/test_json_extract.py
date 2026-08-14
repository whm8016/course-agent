"""extract_json_from_llm 公共解析入口单测（fence/噪声/尾逗号容错）。"""
from core.llm.json_extract import extract_json_from_llm


def test_plain_json():
    assert extract_json_from_llm('{"a": 1}') == {"a": 1}


def test_markdown_fence_stripped():
    assert extract_json_from_llm('```json\n{"a": 1}\n```') == {"a": 1}


def test_trailing_comma_tolerated():
    """LLM JSON 最常见语法错误：对象/数组尾逗号（2026-08 判定器实测踩中）。"""
    assert extract_json_from_llm('{"a": 1,}') == {"a": 1}
    assert extract_json_from_llm('{"judgements": [{"a": 1},]}') == {
        "judgements": [{"a": 1}]
    }


def test_noise_before_json():
    assert extract_json_from_llm('好的，结果是 {"a": 1}') == {"a": 1}


def test_garbage_returns_empty():
    assert extract_json_from_llm("not json {{{") == {}
    assert extract_json_from_llm("") == {}
    assert extract_json_from_llm('[1, 2]') == {}  # 非对象也返回 {}
