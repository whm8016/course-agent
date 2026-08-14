"""Phase 4b: mastery 回流进 prompt——assemble 段顺序（memory 之后）+ build 从 DB 构建。

验证：mastery_context 放在 memory 之后（不破 prefix cache 前缀），空则过滤，
build_common_context_layers 在有 user+course 时从 DB 构建、无则跳过、ctx 预置则用之。
"""
from unittest.mock import AsyncMock, patch

from core.capabilities.chat_pipeline import assemble_system_prompt
from core.context import UnifiedContext
from core.pipeline_common import (
    CommonContextLayers,
    assemble_common_context,
    build_common_context_layers,
)


def test_assemble_common_context_mastery_after_memory():
    """通用层：mastery 紧跟 memory（同用户级易变，不破更稳定的前缀）。"""
    layers = CommonContextLayers(
        memory_context="[mem]", mastery_context="[mastery]", now_text="[now]"
    )
    out = assemble_common_context(layers)
    assert out.index("[mem]") < out.index("[mastery]") < out.index("[now]")


def test_assemble_system_prompt_mastery_after_memory():
    """chat system prompt：mastery 在 memory 之后（L2 用户级）。"""
    out = assemble_system_prompt(
        loop_system="L", course_prompt="C", memory_context="M", mastery_context="MAS", now_text="N"
    )
    assert out.index("M") < out.index("MAS") < out.index("N")


def test_mastery_empty_is_filtered():
    assert assemble_common_context(CommonContextLayers(memory_context="[mem]")) == "[mem]"


async def test_build_layers_builds_mastery_from_db():
    """有 user+course → 经学情拼接门控构建 mastery（mock stitch_for_turn 返回应拼简报）。

    门控（proactive.stitch_for_turn）替换了旧的无条件 get_mastery_context：build 调门控，
    应拼则 mastery_context=brief.text，不拼则空。此处 mock 门控返回"应拼"验证回流。
    """
    from core.memory.proactive import StitchBrief

    ctx = UnifiedContext(user_id="u1", course_id="c1", user_message="怎么求戴维南等效")
    brief = StitchBrief(True, "kcl", "prereq_gap", 0.8, 0.29, text="## 学情提示\n该生KCL偏薄弱")
    with patch("core.llm.prompts.get_course_prompt", new=AsyncMock(return_value="")), \
         patch("core.memory.proactive.stitch_for_turn", new=AsyncMock(return_value=brief)) as sf:
        layers = await build_common_context_layers(ctx)
    assert layers.mastery_context == "## 学情提示\n该生KCL偏薄弱"
    sf.assert_awaited_once()


async def test_build_layers_skips_mastery_without_user_or_course():
    """无 user_id → 不查 mastery（gm 不被调）。"""
    ctx = UnifiedContext(course_id="c1")  # 无 user_id
    with patch("core.memory.mastery.get_mastery_context", new=AsyncMock()) as gm, \
         patch("core.llm.prompts.get_course_prompt", new=AsyncMock(return_value="")):
        layers = await build_common_context_layers(ctx)
    assert layers.mastery_context == ""
    gm.assert_not_called()


async def test_build_layers_uses_preset_ctx_mastery():
    """ctx.mastery_context 预置非空 → 用之，不查 DB。"""
    ctx = UnifiedContext(user_id="u1", course_id="c1", mastery_context="[preset mastery]")
    with patch("core.memory.mastery.get_mastery_context", new=AsyncMock()) as gm, \
         patch("core.llm.prompts.get_course_prompt", new=AsyncMock(return_value="")):
        layers = await build_common_context_layers(ctx)
    assert layers.mastery_context == "[preset mastery]"
    gm.assert_not_called()
