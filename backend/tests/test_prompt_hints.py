"""prompt hint 加载/渲染/注入测试（prompting 层）。

渲染层是纯函数、不依赖 chat_pipeline，可独立测试核心逻辑：
加载（字段/语言归一/缺失回退）、渲染（基本格式/空 names/跳过缺失/去重保序/
动态工具条件渲染/跳过通配符）。
"""
from core.agent.prompting import (
    ToolPromptHints,
    build_tool_hint_text,
    load_prompt_hints,
)


def test_load_prompt_hints_rag():
    h = load_prompt_hints("rag", "zh")
    assert h.short_description
    assert "知识库" in h.short_description
    assert h.when_to_use
    assert h.input_format
    assert h.guideline
    assert h.phase == "exploration"


def test_load_prompt_hints_language_normalize():
    # zh-CN / ZH / 默认 都归一到 zh
    assert load_prompt_hints("rag", "zh-CN").short_description
    assert load_prompt_hints("rag", "ZH").short_description
    assert load_prompt_hints("rag").short_description


def test_load_prompt_hints_missing_returns_empty():
    h = load_prompt_hints("nonexistent_tool_xyz", "zh")
    assert h == ToolPromptHints()


def test_build_hint_text_basic():
    text = build_tool_hint_text(["rag", "web_search"], "zh")
    assert "## 可用工具" in text
    assert "`rag`" in text
    assert "`web_search`" in text
    assert "适用场景" in text
    assert "参数格式" in text


def test_build_hint_text_empty_names():
    # 空 names 仍渲染 always-on 工具（记忆读写），不再返回空串：always_on 工具无视 names 追加
    text = build_tool_hint_text([], "zh")
    assert "`read_memory`" in text
    assert "`write_memory`" in text
    text_none = build_tool_hint_text(None, "zh")
    assert "`read_memory`" in text_none


def test_build_hint_text_skips_missing_hint_files():
    text = build_tool_hint_text(["rag", "ghost_tool"], "zh")
    assert "`rag`" in text
    assert "ghost_tool" not in text


def test_build_hint_text_dedup_preserve_order():
    text = build_tool_hint_text(["web_search", "rag", "web_search"], "zh")
    assert text.index("`web_search`") < text.index("`rag`")
    assert text.count("`web_search`") == 1


def test_build_hint_text_dynamic_read_skill_conditional():
    with_manifest = build_tool_hint_text([], "zh", skills_manifest="## Skills\n- x")
    assert "`read_skill`" in with_manifest
    assert "`read_skill`" not in build_tool_hint_text([], "zh")


def test_build_hint_text_dynamic_load_tools_conditional():
    with_manifest = build_tool_hint_text([], "zh", extended_tools_manifest="## 扩展工具\n- y")
    assert "`load_tools`" in with_manifest
    assert "`load_tools`" not in build_tool_hint_text([], "zh")


def test_build_hint_text_skips_star():
    text = build_tool_hint_text(["*", "rag"], "zh")
    assert "`rag`" in text
    assert "`*`" not in text


def test_build_hint_text_static_plus_dynamic_order():
    # 静态工具在前，动态 read_skill / load_tools 在后
    text = build_tool_hint_text(
        ["rag"],
        "zh",
        skills_manifest="## Skills\n- x",
        extended_tools_manifest="## 扩展工具\n- y",
    )
    assert text.index("`rag`") < text.index("`read_skill`")
    assert text.index("`read_skill`") < text.index("`load_tools`")
