"""pipeline_common 公共 helper 单测。

验证：
- assemble_common_context 段顺序 / 空段过滤 / 全空
- ProfileRuntime / CommonContextLayers 默认值与 frozen 不可变
- describe_images 把 rt.text_model/binding 透传给 describe_images_into
- build_common_context_layers 组装（mock get_course_prompt；include_skills 开关）
- resolve_profile_runtime 回退路径（无 provider / 无 profile → 全 None）

asyncio_mode=auto（pyproject.toml），async def test 自动按 asyncio 跑。
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, patch

import pytest

from core.context import UnifiedContext
from core.pipeline_common import (
    CommonContextLayers,
    ProfileRuntime,
    assemble_common_context,
    build_common_context_layers,
    describe_images,
    resolve_profile_runtime,
)


# ── assemble_common_context（纯函数）─────────────────────────────────────────


def test_assemble_common_context_orders_and_filters():
    layers = CommonContextLayers(
        course_prompt="[course]",
        bot_persona="",  # 空段应被过滤
        always_skills="[always]",
        memory_context="[mem]",
        session_summary="",  # 空段过滤
        now_text="[now]",
    )
    out = assemble_common_context(layers)
    # 按稳定性递减顺序拼接，空段剔除
    assert out == "[course]\n\n[always]\n\n[mem]\n\n[now]"


def test_assemble_common_context_all_empty():
    assert assemble_common_context(CommonContextLayers()) == ""


def test_assemble_common_context_single():
    assert assemble_common_context(CommonContextLayers(now_text="[now]")) == "[now]"


def test_assemble_common_context_course_before_memory():
    """课程级（稳定）必须排在用户级记忆（易变）之前——通用层拼装顺序契约。"""
    layers = CommonContextLayers(course_prompt="C", memory_context="M", now_text="N")
    out = assemble_common_context(layers)
    assert out.index("C") < out.index("M") < out.index("N")


# ── dataclass 默认值 + frozen ────────────────────────────────────────────────


def test_profile_runtime_defaults():
    rt = ProfileRuntime()
    assert rt.client is None
    assert rt.text_model is None
    assert rt.binding is None


def test_profile_runtime_frozen():
    rt = ProfileRuntime(text_model="gpt-4o", binding="openai")
    with pytest.raises(FrozenInstanceError):
        rt.text_model = "x"  # type: ignore[misc]


def test_common_context_layers_defaults():
    layers = CommonContextLayers()
    assert layers.course_prompt == ""
    assert layers.now_text == ""


# ── describe_images 透传 rt ───────────────────────────────────────────────────


async def test_describe_images_passes_runtime():
    ctx = UnifiedContext(user_id="u1")
    rt = ProfileRuntime(text_model="gpt-4o", binding="openai")
    with patch(
        "core.llm.vision_describe.describe_images_into",
        new=AsyncMock(return_value="described"),
    ) as m:
        result = await describe_images(ctx, "base", rt)
    assert result == "described"
    _args, kwargs = m.call_args
    assert kwargs["text_model"] == "gpt-4o"
    assert kwargs["binding"] == "openai"
    assert kwargs["user_id"] == "u1"


async def test_describe_images_none_runtime_passes_none():
    """rt 全 None（回退）时透传 None，describe_images_into 内部回退全局默认模型。"""
    ctx = UnifiedContext(user_id="u2")
    rt = ProfileRuntime()
    with patch(
        "core.llm.vision_describe.describe_images_into",
        new=AsyncMock(return_value="base"),
    ) as m:
        await describe_images(ctx, "base", rt)
    _args, kwargs = m.call_args
    assert kwargs["text_model"] is None
    assert kwargs["binding"] is None


# ── build_common_context_layers ──────────────────────────────────────────────


async def test_build_common_context_layers_basic():
    ctx = UnifiedContext(
        course_id="math101",
        memory_context="[mem snapshot]",
        session_summary="早期摘要",
        metadata={"bot_persona": "你是助教"},
    )
    with patch(
        "core.llm.prompts.get_course_prompt", new=AsyncMock(return_value="[course prompt]")
    ):
        layers = await build_common_context_layers(ctx)
    assert layers.course_prompt == "[course prompt]"
    assert layers.bot_persona == "你是助教"
    assert layers.memory_context == "[mem snapshot]"
    assert "早期摘要" in layers.session_summary
    assert layers.now_text.startswith("【当前时间】")
    # include_skills=False 默认 → always_skills 不查
    assert layers.always_skills == ""


async def test_build_common_context_layers_include_skills():
    ctx = UnifiedContext(course_id="c1", user_id="u1")
    fake_svc = type(
        "Svc", (), {"load_always_for_context": lambda self: "[always skill]"}
    )()
    with patch("core.llm.prompts.get_course_prompt", new=AsyncMock(return_value="")), \
         patch("core.skills.skill_service.get_skill_service", return_value=fake_svc):
        layers = await build_common_context_layers(ctx, include_skills=True)
    assert layers.always_skills == "[always skill]"


async def test_build_common_context_layers_empty_summary():
    """session_summary 为空时该层为 ""（被 assemble 过滤）。"""
    ctx = UnifiedContext()
    with patch("core.llm.prompts.get_course_prompt", new=AsyncMock(return_value="")):
        layers = await build_common_context_layers(ctx)
    assert layers.session_summary == ""


# ── resolve_profile_runtime 回退路径 ─────────────────────────────────────────


async def test_resolve_profile_runtime_fallback(monkeypatch):
    """无 user_id、catalog 无 profile → 全 None 回退（loop 用全局默认）。"""
    import core.llm.catalog as catalog

    # resolve_profile_runtime 走 cached 异步读路径，故 monkeypatch cached 版本
    monkeypatch.setattr(catalog, "active_profile_id_cached", AsyncMock(return_value=""))
    monkeypatch.setattr(catalog, "get_profile_cached", AsyncMock(return_value=None))
    rt = await resolve_profile_runtime(profile_id="", user_id="")
    assert isinstance(rt, ProfileRuntime)
    assert rt.client is None
    assert rt.text_model is None
    assert rt.binding is None


async def test_resolve_profile_runtime_platform_profile(monkeypatch):
    """有平台 profile → 返回其 text_model/binding（client 经 provider_factory 构造）。"""
    import core.llm.catalog as catalog

    fake_prof = {"binding": "openai", "text": {"model": "gpt-4o"}}
    monkeypatch.setattr(catalog, "active_profile_id_cached", AsyncMock(return_value="p1"))
    monkeypatch.setattr(catalog, "get_profile_cached", AsyncMock(return_value=fake_prof))
    monkeypatch.setattr(catalog, "profile_text_model", lambda p: "gpt-4o")
    with patch("core.llm.provider_factory.get_llm_client_for_profile", return_value="CLIENT"):
        rt = await resolve_profile_runtime(profile_id="p1", user_id="")
    assert rt.text_model == "gpt-4o"
    assert rt.binding == "openai"
    assert rt.client == "CLIENT"
