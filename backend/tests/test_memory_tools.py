"""read_memory / write_memory 工具回归测试（plan §6）。

核心断言：
- write_memory 调 add 时 infer=False（逐字存原文，不让 LLM 抽取改写字面值）、
  metadata.course_id 正确、原文未被截断改写。
- write_memory 无 user_id / 空 content -> success=False；超长 content 截到上限。
- read_memory 的 filters 同时带 user_id 与 course_id；零命中 -> 友好文案 + success=False。
- 两个 schema 的 parameters.properties 均不含 user_id（IDOR 结构性守卫，同 academic 工具）。
- registry 装配成功：get_tool_schemas(enabled_tools=[...]) 能取到 schema。
- hint YAML 必须存在且能渲染，否则工具在 system prompt 里静默隐身（plan 标注的最易漏点）。

身份注入契约（fail-closed）：schema 不暴露 user_id，模型幻觉出该参数会与注入值撞 TypeError
-> registry 兜底「工具执行失败」。本测试固化该契约。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ── 身份注入契约（IDOR 结构性守卫）──────────────────────────────────────────

def test_memory_schemas_have_no_identity_params():
    """schema 绝不暴露 user_id 等身份参数（身份只走 registry 注入）。"""
    from core.memory.tools import READ_MEMORY_SCHEMA, WRITE_MEMORY_SCHEMA

    forbidden = {"user_id", "student_id", "uid", "owner_id"}
    for schema in (WRITE_MEMORY_SCHEMA, READ_MEMORY_SCHEMA):
        props = set(schema["function"]["parameters"].get("properties", {}).keys())
        leaked = props & forbidden
        assert not leaked, f"{schema['function']['name']} schema 泄漏身份参数: {leaked}"


def test_memory_schemas_names():
    from core.memory.tools import READ_MEMORY_SCHEMA, WRITE_MEMORY_SCHEMA
    assert WRITE_MEMORY_SCHEMA["function"]["name"] == "write_memory"
    assert READ_MEMORY_SCHEMA["function"]["name"] == "read_memory"


# ── registry 装配 ────────────────────────────────────────────────────────────

def test_memory_tools_registered():
    """两个工具已注册进 ToolRegistry（名字 + schema 名一致）。"""
    from core.agent.registry import get_tool_registry

    reg = get_tool_registry()
    for name in ("read_memory", "write_memory"):
        assert reg.has(name), f"工具 {name} 未注册"
        assert reg.get(name).schema["function"]["name"] == name


def test_get_tool_schemas_returns_memory_schema():
    """enabled_tools 含记忆工具时 get_tool_schemas 能取到其 schema（装配链路通）。"""
    from core.agent.registry import get_tool_schemas
    from core.context import UnifiedContext

    schemas = get_tool_schemas(
        UnifiedContext(enabled_tools=["read_memory", "write_memory"])
    )
    assert schemas is not None
    names = {s["function"]["name"] for s in schemas}
    assert "read_memory" in names
    assert "write_memory" in names


def test_memory_hints_render_not_invisible():
    """hint YAML 必须存在且有 short_description，否则 build_tool_hint_text 静默跳过使工具隐身。"""
    from core.agent.prompting import build_tool_hint_text

    text = build_tool_hint_text(["write_memory", "read_memory"])
    assert text, "记忆工具 hint 未渲染（YAML 缺失或 short_description 为空 -> 工具隐身）"
    assert "write_memory" in text
    assert "read_memory" in text


# ── always_on 挂载（SSE/WS/bot 三入口共用 resolve，不必各自合并）──────────────

def test_memory_tools_marked_always_on():
    """read_memory/write_memory 注册时标 always_on=True（resolve/build_tool_hint_text 据此追加）。"""
    from core.agent.registry import get_tool_registry

    reg = get_tool_registry()
    for name in ("read_memory", "write_memory"):
        entry = reg.get(name)
        assert entry is not None
        assert entry.always_on is True, f"{name} 应标记 always_on"
    assert {e.name for e in reg.always_on_entries()} == {"read_memory", "write_memory"}


def test_resolve_appends_always_on_even_when_not_enabled():
    """enabled_tools 不含记忆工具时 resolve 仍追加其 schema（修 WS/bot 入口遗漏 gap）。"""
    from core.agentic.dynamic_tools import resolve
    from core.context import UnifiedContext

    schemas, _ = resolve(UnifiedContext(enabled_tools=["rag"]))
    names = {s["function"]["name"] for s in (schemas or [])}
    assert "rag" in names
    assert "read_memory" in names  # always_on 追加，无需 enabled_tools 显式包含
    assert "write_memory" in names


def test_resolve_appends_always_on_for_empty_tools():
    """enabled_tools 为空时 resolve 仍返回 always_on schema（学生无工具也能记/读记忆）。"""
    from core.agentic.dynamic_tools import resolve
    from core.context import UnifiedContext

    schemas, _ = resolve(UnifiedContext(enabled_tools=[]))
    names = {s["function"]["name"] for s in (schemas or [])}
    assert names == {"read_memory", "write_memory"}


def test_build_hint_text_includes_always_on_for_empty_names():
    """空 names 时 build_tool_hint_text 仍渲染 always_on 工具（schema 与 prompt 不脱节）。"""
    from core.agent.prompting import build_tool_hint_text

    text = build_tool_hint_text([], "zh")
    assert "`read_memory`" in text
    assert "`write_memory`" in text


# ── write_memory ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_memory_uses_infer_false_and_stores_verbatim():
    """核心断言：infer=False + 原文逐字入库 + metadata.course_id 正确。"""
    from core.memory.tools import execute_write_memory

    mem = AsyncMock()
    mem.add.return_value = {"results": [{"id": "m1", "memory": "x", "event": "ADD"}]}
    with patch("core.memory.mem0_client.get_memory", return_value=mem):
        res = await execute_write_memory(
            course_id="c1", user_id="u1", content="我的会员卡号是 6225880137001234"
        )

    assert res.success is True
    assert mem.add.await_count == 1
    args, kwargs = mem.add.await_args
    # infer=False 关键：逐字存原文，不让 mem0 抽取改写字面值
    assert args[0] == "我的会员卡号是 6225880137001234"
    assert kwargs["infer"] is False
    assert kwargs["user_id"] == "u1"
    assert kwargs["metadata"]["course_id"] == "c1"
    assert kwargs["metadata"]["source"] == "explicit_tool"
    # 回执复述原文供学生确认
    assert "6225880137001234" in res.content


@pytest.mark.asyncio
async def test_write_memory_rejects_empty_user_id_and_content():
    from core.memory.tools import execute_write_memory

    # 空 user_id 直接拒（身份只走注入）
    res = await execute_write_memory(user_id="", course_id="c1", content="x")
    assert res.success is False

    # 空 content 直接拒
    res = await execute_write_memory(user_id="u1", course_id="c1", content="   ")
    assert res.success is False


@pytest.mark.asyncio
async def test_write_memory_truncates_overlong_content():
    from core.memory.tools import _MAX_WRITE_CHARS, execute_write_memory

    long = "记" * (_MAX_WRITE_CHARS + 50)
    mem = AsyncMock()
    mem.add.return_value = {"results": []}
    with patch("core.memory.mem0_client.get_memory", return_value=mem):
        res = await execute_write_memory(course_id="c1", user_id="u1", content=long)

    assert res.success is True
    args, _ = mem.add.await_args
    assert len(args[0]) == _MAX_WRITE_CHARS  # 截到上限
    assert "截断" in res.content  # 提示已截断


@pytest.mark.asyncio
async def test_write_memory_empty_course_id_still_works():
    """无课程上下文（自由问答 general）时 course_id 留空，不阻塞写入。"""
    from core.memory.tools import execute_write_memory

    mem = AsyncMock()
    mem.add.return_value = {"results": []}
    with patch("core.memory.mem0_client.get_memory", return_value=mem):
        res = await execute_write_memory(course_id="", user_id="u1", content="目标分数 90")

    assert res.success is True
    _, kwargs = mem.add.await_args
    assert kwargs["metadata"]["course_id"] == ""


# ── read_memory ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_memory_filters_isolate_user_and_course():
    """filters 必须同时带 user_id 与 course_id（多租户 + 课程隔离）。"""
    from core.memory.tools import execute_read_memory

    mem = AsyncMock()
    mem.search.return_value = {"results": [{"memory": "目标分数 90"}]}
    with patch("core.memory.mem0_client.get_memory", return_value=mem):
        res = await execute_read_memory(course_id="c1", user_id="u1", query="目标")

    assert res.success is True
    assert "目标分数 90" in res.content
    _, kwargs = mem.search.await_args
    assert kwargs["filters"] == {"user_id": "u1", "course_id": "c1"}
    assert kwargs["top_k"] == 5  # 默认 limit


@pytest.mark.asyncio
async def test_read_memory_empty_query_falls_back_to_get_all():
    """query 为空退化 get_all 列最近若干条（不调 search）。"""
    from core.memory.tools import execute_read_memory

    mem = AsyncMock()
    mem.get_all.return_value = {"results": [{"memory": "偏好图解"}, {"memory": "教材第3版"}]}
    with patch("core.memory.mem0_client.get_memory", return_value=mem):
        res = await execute_read_memory(course_id="c1", user_id="u1")

    assert res.success is True
    assert mem.search.await_count == 0
    assert mem.get_all.await_count == 1
    _, kwargs = mem.get_all.await_args
    assert kwargs["filters"] == {"user_id": "u1", "course_id": "c1"}


@pytest.mark.asyncio
async def test_read_memory_zero_hits_returns_failure():
    """零命中 -> 友好文案 + success=False（不抛异常）。"""
    from core.memory.tools import execute_read_memory

    mem = AsyncMock()
    mem.search.return_value = {"results": []}
    with patch("core.memory.mem0_client.get_memory", return_value=mem):
        res = await execute_read_memory(course_id="c1", user_id="u1", query="不存在的东西")

    assert res.success is False
    assert "没有找到" in res.content


@pytest.mark.asyncio
async def test_read_memory_rejects_empty_user_id():
    from core.memory.tools import execute_read_memory

    res = await execute_read_memory(user_id="", course_id="c1", query="x")
    assert res.success is False


@pytest.mark.asyncio
async def test_read_memory_handles_mem0_exception():
    """mem0 抛异常时降级为友好文案（不把异常抛给 agent loop）。"""
    from core.memory.tools import execute_read_memory

    mem = AsyncMock()
    mem.search.side_effect = RuntimeError("pg down")
    with patch("core.memory.mem0_client.get_memory", return_value=mem):
        res = await execute_read_memory(course_id="c1", user_id="u1", query="x")

    assert res.success is False
    assert "读取记忆失败" in res.content
