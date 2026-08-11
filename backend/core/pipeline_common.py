"""Pipeline 公共 helper —— 四条 pipeline（chat / solve / research / quiz）共享的运行时解析、
通用上下文层组装与图片两阶段描述包装。

设计选择：**组合优于继承**。本模块提供「无状态函数 + 不可变 dataclass」，四条 pipeline
各自 import 调用，不引入 BasePipeline 抽象基类。理由：research 是 4 阶段多 loop、quiz 是
3 阶段多 loop，套不进「基类 run() 调一次 loop」的模板方法，强套产生坏继承。真正的共性是
「loop 外的三个准备步骤」（解析 profile / 组装通用上下文层 / 两阶段图片描述）——把它们
函数化下沉，让每条 pipeline 在自己的 execute 里按需组合，是更优的解耦。

对外接口：
  resolve_profile_runtime(profile_id, user_id) → ProfileRuntime
      解析**对话**供应商 (client, text_model, binding)。用户自配 > 平台 profile > 全 None。
  build_common_context_layers(ctx, *, include_skills=False) → CommonContextLayers
      组装通用上下文层（course_prompt / bot_persona / memory / session_summary / now）。
      include_skills=True 时额外急切注入 always-on skill 全文（仅 chat 挂 read_skill 工具；
      solve/research/quiz 不挂，传 False 避免提示有 skill 却无工具可读）。
  assemble_common_context(layers) → str
      纯函数：按稳定性递减拼接通用层，过滤空段。
  describe_images(ctx, base_text, rt) → str
      describe_images_into 的薄包装，统一把 rt.text_model/binding 透传（修复 solve/research/quiz
      此前只传 user_id、误判主模型不支持 vision 而白走两阶段的问题）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from settings import get_settings
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.context import UnifiedContext

logger = logging.getLogger(__name__)

# T2 动态层（memory/mastery/summary）集体预算 = soft_trigger // 此除数（模块常量，非配置项）。
# 旧实现用 mecw×target_pct×tier2_pct 百分比链（已随上下文管理重构移除）；新双阈值无百分比，
# 改用 compute_budgets 的 soft_trigger 派生。10 ≈ 旧 qwen-plus 量级（soft_trigger 64k/10=6.4k vs 旧 5.9k）。
_T2_BUDGET_DIVISOR = 10


@dataclass(frozen=True)
class ProfileRuntime:
    """一次 turn 的对话供应商运行时（run_agent_loop override 用）。

    frozen=True → 不可变，research 多阶段并行块只读共享安全。
    全 None 时 loop 回退全局 _default_client + TEXT_MODEL（向后兼容）。
    """

    client: Any | None = None
    text_model: str | None = None
    binding: str | None = None


@dataclass(frozen=True)
class CommonContextLayers:
    """通用上下文层（chat 9 层里去掉 loop_system/skills_manifest/tool_hint/extended 的那几层）。

    frozen=True → 不可变，多阶段 pipeline 复用同一份安全。assemble_common_context 自动过滤空段。
    """

    course_prompt: str = ""
    bot_persona: str = ""
    always_skills: str = ""
    memory_context: str = ""
    mastery_context: str = ""
    session_summary: str = ""
    now_text: str = ""


async def resolve_profile_runtime(
    profile_id: str, user_id: str = ""
) -> ProfileRuntime:
    """解析**对话**供应商 → ProfileRuntime(client, text_model, binding)。

    视觉供应商由 _resolve_vision_runtime 独立解析（可走不同 binding/key/url）。

    优先级：
    1. 用户自配 provider（DB，user_id 非空时查）
    2. 用户下拉选的平台 profile_id → 平台 active profile（catalog）
    3. 回退 ProfileRuntime()（全 None），loop 用全局 _default_client + TEXT_MODEL

    client 经 provider_factory.get_llm_client_for_profile 按 profile 指纹缓存，空 key 回退 .env。
    """
    from core.llm.catalog import (
        active_profile_id_cached,
        get_profile_cached,
        profile_text_model,
    )
    from core.llm.provider_factory import get_llm_client_for_profile

    try:
        # 1. 用户自配 provider（优先）
        if user_id:
            from core.db.user_llm_provider import get_active_provider_view

            user_prof = await get_active_provider_view(user_id)
            if user_prof:
                text_m = (
                    (user_prof.get("models", {}) or {}).get("text", {}) or {}
                ).get("model") or None
                binding = (user_prof.get("binding") or "").strip() or None
                return ProfileRuntime(
                    client=get_llm_client_for_profile(user_prof),
                    text_model=text_m,
                    binding=binding,
                )

        # 2/3. 平台 profile（profile_id → active，走 Redis 缓存避免同步文件 I/O 阻塞）
        pid = (profile_id or "").strip() or await active_profile_id_cached()
        prof = await get_profile_cached(pid)
        if prof:
            return ProfileRuntime(
                client=get_llm_client_for_profile(prof),
                text_model=profile_text_model(prof) or None,
                binding=(prof.get("binding") or "").strip() or None,
            )

        # 4. 回退默认
        return ProfileRuntime()
    except Exception:
        logger.exception(
            "resolve profile runtime failed profile_id=%r user_id=%r", profile_id, user_id
        )
        return ProfileRuntime()


async def build_common_context_layers(
    ctx: "UnifiedContext", *, include_skills: bool = False
) -> CommonContextLayers:
    """组装通用上下文层。各 pipeline 在 run() 开头调一次，多阶段复用同一份（避免每阶段重查 DB）。

    include_skills=True 时急切注入 always-on skill 全文（仅 chat——它挂 read_skill 工具）。
    solve/research/quiz 传 False：它们不挂 read_skill，注入全文却无工具会让模型误以为有 skill 可读。
    course_prompt 经 get_course_prompt 走 Redis 缓存。
    """
    from core.llm.prompts import get_course_prompt

    course_prompt = await get_course_prompt(ctx.course_id)

    always_skills = ""
    if include_skills:
        from core.skills.skill_service import get_skill_service

        svc = get_skill_service(ctx.course_id, ctx.user_id)
        always_skills = svc.load_always_for_context()

    session_summary = (
        f"## 本次对话前情摘要（早期对话的压缩，非完整原文）\n{ctx.session_summary}"
        if ctx.session_summary
        else ""
    )
    now_text = f"【当前时间】{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %A')}"

    # L3 掌握度（薄弱点）：所有 pipeline 共享。ctx.mastery_context 预置则用之，否则从 DB 构建。
    # 顺带修「只有 chat 注入记忆」的缺口——solve/research/quiz 也拿到掌握度，按需诊断反复性错误。
    mastery_context = ctx.mastery_context or ""
    if not mastery_context and ctx.user_id and ctx.course_id:
        try:
            from core.db.database import AsyncSessionLocal
            from core.memory.mastery import get_mastery_context

            async with AsyncSessionLocal() as db:
                mastery_context = await get_mastery_context(db, ctx.user_id, ctx.course_id)
        except Exception:
            logger.exception(
                "build mastery context failed user=%s course=%s", ctx.user_id, ctx.course_id
            )
            mastery_context = ""

    metadata = ctx.metadata or {}
    memory_context = ctx.memory_context or ""
    # Phase 2：coordinator_enabled 时对 T2 动态层（memory/mastery/summary）集体预算裁剪，低优先级
    # 先裁（memory->mastery->summary，summary 不可重建最后裁）。默认关=各层走原 per-slice cap。
    if get_settings().context_budget.coordinator_enabled:
        from core.agentic.context_budget import cap_dynamic_slices
        from core.agentic.context_window import compute_budgets
        # T2 动态层集体裁剪：超预算时按优先级（低->高）整段丢弃最低优先级切片，直到回到预算内。
        # 预算受同一 soft_trigger 驱动（与 enforce 级联同阈值），非旧百分比链。
        _soft_trigger, _ = compute_budgets(get_settings().llm.text_model)
        _t2_budget = max(1, _soft_trigger // _T2_BUDGET_DIVISOR)
        _capped = cap_dynamic_slices(
            {
                "memory_context": memory_context,
                "mastery_context": mastery_context,
                "session_summary": session_summary,
            },
            _t2_budget,
            frozenset(),
            ["memory_context", "mastery_context", "session_summary"],
        )
        memory_context = _capped["memory_context"]
        mastery_context = _capped["mastery_context"]
        session_summary = _capped["session_summary"]
    return CommonContextLayers(
        course_prompt=course_prompt,
        bot_persona=(metadata.get("bot_persona") or "").strip(),
        always_skills=always_skills,
        memory_context=memory_context,
        mastery_context=mastery_context,
        session_summary=session_summary,
        now_text=now_text,
    )


def assemble_common_context(layers: CommonContextLayers) -> str:
    """纯函数：按稳定性递减拼接通用层（course→persona→always→memory→summary→now），过滤空段。

    与 chat 的 assemble_system_prompt 同语义（稳定性递减 + 空段过滤 + 空行连接），但不含
    chat 专属的 loop_system/skills_manifest/tool_hint/extended 段。各 pipeline 在 task_system
    之后叠加本输出：``task_system + "\\n\\n" + assemble_common_context(layers)``。
    """
    return "\n\n".join(
        p
        for p in (
            layers.course_prompt,
            layers.bot_persona,
            layers.always_skills,
            layers.memory_context,
            layers.mastery_context,
            layers.session_summary,
            layers.now_text,
        )
        if p
    )


def with_common_prompt(task_system: str, layers: CommonContextLayers) -> str:
    """任务提示词 + 通用上下文层拼接；通用层为空时只保留任务提示词。

    solve/research/quiz 的通用入口：先 assemble_common_context 拼通用层，非空则与
    task_system 用空行连接，否则原样返回 task_system（避免多余空行）。与 chat 的
    assemble_system_prompt 同语义的薄包装，统一三处原先各自复制的 _with_common。
    """
    common = assemble_common_context(layers)
    return f"{task_system}\n\n{common}" if common else task_system


async def describe_images(
    ctx: "UnifiedContext", base_text: str, rt: ProfileRuntime
) -> str:
    """两阶段图片描述的统一包装：把 rt.text_model/binding 透传给 describe_images_into。

    修复 solve/research/quiz 此前只传 user_id、用全局默认模型判断 → 即使主模型支持 vision
    也误走两阶段（白调一次 vision 模型把图描述成文字）的问题。
    """
    from core.llm.vision_describe import describe_images_into

    return await describe_images_into(
        ctx,
        base_text,
        user_id=ctx.user_id,
        text_model=rt.text_model,
        binding=rt.binding,
    )


__all__ = [
    "ProfileRuntime",
    "CommonContextLayers",
    "resolve_profile_runtime",
    "build_common_context_layers",
    "assemble_common_context",
    "with_common_prompt",
    "describe_images",
]
