"""Prompt hint 加载与渲染（对标 DeepTutor ``tools/prompting/__init__.py``）。

把每个工具的**使用提示**从 YAML 加载，渲染成文本注入 system prompt，与工具的
**功能 schema**（``TOOLS_OPENAI_SCHEMA``）分离——两套独立数据源，工具说明单一真相源。

加载器对标 DeepTutor ``load_prompt_hints``：从 ``hints/{lang}/{tool_name}.yaml`` 读，
zh 回退 en（当前只提供 zh，回退分支保留以便后续加 en，零代码改动）。
渲染对标 DeepTutor ``ToolPromptComposer.format_list_with_usage``（带 适用场景 + 参数格式），
让 LLM 有足够线索判断*是否*调用一个工具，而不仅知其名。

本模块为纯叠加层：**不改** ``TOOLS_OPENAI_SCHEMA`` / ``execute_tool`` /
``DynamicToolResolver.resolve`` 中的任何一字，只新增一处「使用提示」渲染入口。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_HINTS_DIR = Path(__file__).parent / "hints"


@dataclass
class ToolPromptHints:
    """单个工具的使用提示（对标 DeepTutor ``core/tool_protocol.ToolPromptHints``）。

    ``short_description`` 决定该工具是否进入渲染的 hint 块（为空则跳过）。
    ``phase`` 预留给后续分阶段渲染（``format_phased``），当前 list 渲染不分组。
    """

    short_description: str = ""
    when_to_use: str = ""
    input_format: str = ""
    guideline: str = ""
    note: str = ""
    phase: str = ""


def _normalize_language(language: str) -> str:
    normalized = (language or "").lower()
    if normalized.startswith("zh"):
        return "zh"
    if normalized.startswith("en"):
        return "en"
    return normalized or "zh"


def load_prompt_hints(tool_name: str, language: str = "zh") -> ToolPromptHints:
    """从 ``hints/{lang}/{tool_name}.yaml`` 加载提示，zh 回退 en；缺失/异常返回空 hints。

    不缓存（文件小，开发期改动即时生效；与 ``prompt_loader.load_prompt_dict`` 同策略）。
    对标 DeepTutor ``tools/prompting/__init__.py:load_prompt_hints``。
    """
    lang = _normalize_language(language)
    candidates = [_HINTS_DIR / lang / f"{tool_name}.yaml"]
    if lang != "en":
        candidates.append(_HINTS_DIR / "en" / f"{tool_name}.yaml")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return ToolPromptHints()
        if not isinstance(data, dict):
            return ToolPromptHints()
        return ToolPromptHints(
            short_description=str(data.get("short_description", "")).strip(),
            when_to_use=str(data.get("when_to_use", "")).strip(),
            input_format=str(data.get("input_format", "")).strip(),
            guideline=str(data.get("guideline", "")).strip(),
            note=str(data.get("note", "")).strip(),
            phase=str(data.get("phase", "")).strip(),
        )
    return ToolPromptHints()


def _render_one(name: str, hint: ToolPromptHints) -> str | None:
    """``format_list_with_usage`` 的单条渲染；无 ``short_description`` 返回 None（跳过）。"""
    if not hint.short_description:
        return None
    entry = [f"- `{name}` — {hint.short_description}"]
    if hint.when_to_use:
        entry.append(f"    适用场景: {hint.when_to_use}")
    if hint.input_format:
        entry.append(f"    参数格式: {hint.input_format}")
    return "\n".join(entry)


def build_tool_hint_text(
    names: list[str] | None,
    language: str = "zh",
    *,
    skills_manifest: str = "",
    extended_tools_manifest: str = "",
) -> str:
    """渲染工具使用提示文本块，注入 chat system prompt。

    对标 DeepTutor ``ToolRegistry.build_prompt_text``：按启用的工具名渲染提示。

    过滤逻辑（与 ``_get_tool_schemas`` / ``DynamicToolResolver.resolve`` 同语义）：
    - ``names`` 来自 ``context.enabled_tools``；空/None → 返回 ``""``（本轮无工具，
      不注入；与 schema 过滤一致）。
    - 按 ``names`` 顺序去重保留；缺 hint 文件（``short_description`` 空）的工具静默跳过。
    - 跳过 ``"*"``（``context.enabled_tools`` 不应含通配符；``*`` 仅 MCP server 内部
      白名单用，见 ``core/mcp/config.py``）。

    动态工具（与 ``resolve`` 挂载时机 1:1 对齐，避免「提示说有、schema 没挂」错位）：
    - ``skills_manifest`` 非空 → 追加 ``read_skill`` 提示（``resolve`` 同条件下挂
      ``READ_SKILL_SCHEMA``）。
    - ``extended_tools_manifest`` 非空 → 追加 ``load_tools`` 提示（``resolve`` 同条件下
      挂 ``LOAD_TOOLS_SCHEMA``；用 manifest 非空作 deferred pool 非空的代理信号，
      避免渲染层依赖 MCP manager）。
    """
    blocks: list[str] = []
    seen: set[str] = set()
    for nm in names or []:
        nm = str(nm).strip()
        if not nm or nm in seen or nm == "*":
            continue
        seen.add(nm)
        rendered = _render_one(nm, load_prompt_hints(nm, language))
        if rendered:
            blocks.append(rendered)

    if skills_manifest:
        rendered = _render_one("read_skill", load_prompt_hints("read_skill", language))
        if rendered:
            blocks.append(rendered)
    if extended_tools_manifest:
        rendered = _render_one("load_tools", load_prompt_hints("load_tools", language))
        if rendered:
            blocks.append(rendered)

    if not blocks:
        return ""
    return "## 可用工具\n\n" + "\n".join(blocks)


__all__ = ["ToolPromptHints", "load_prompt_hints", "build_tool_hint_text"]
