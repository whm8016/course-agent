"""ChatPipeline — chat capability 的 agent loop 执行层。

路由完全由上层 Orchestrator + CapabilityRegistry 按 context.mode 完成：
  chat          → ChatCapability         → ChatPipeline (本文件) → run_agent_loop
  deep_solve    → DeepSolveCapability    → DeepSolvePipeline（单 agent loop + solve 工具状态机）
  deep_research → DeepResearchCapability → ResearchPipeline（四阶段：rephrase → decompose → research → reporting）
  quiz          → QuizCapability

本文件只负责 chat 模式：组装 system prompt → 驱动 agent loop。

chat 的 agent loop 行为规范外部化到 prompts/zh/chat.yaml（agentic_chat.yaml，去除 label 协议外壳，适配 tool_calls loop）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from core.agentic.loop import run_agent_loop
from core.context import UnifiedContext
from core.observability import log_flow
from core.pipeline_common import (
    build_common_context_layers,
    describe_images,
    resolve_profile_runtime,
)
from core.stream_bus import StreamBus
from settings import get_settings

logger = logging.getLogger(__name__)

# chat agent loop 行为规范（外部化 YAML，agentic_chat.yaml）
_CHAT_PROMPT_PATH = Path(__file__).parent / "prompts" / "zh" / "chat.yaml"

# system prompt 总预算护栏阈值（字符）。超阈值打 WARNING + 各段字符分解，定位是哪一层膨胀。
# 只告警不截断——system prompt 静默截断会破坏 prefix cache 且难排查。
_SYSTEM_PROMPT_WARN_CHARS = 8_000

# chat 探索轮温度（对标 DeepTutor chat 0.2）。0.7 对工具决策偏高，是「同一问题拆成多次检索」
# 的推手；降到 0.3 收敛工具选择，配合 KB Seed 让材料够时第 1 轮直接作答。
_CHAT_TEMPERATURE = 0.3


async def retrieve_kb_seed(context: UnifiedContext, stream: StreamBus, header: str) -> str:
    """进 loop 前用原问题预检索一次课程知识库，命中则作为 [知识库预检索] 块注入本轮消息。

    复用 ``execute_tool("rag", strategy="fact")``：自动继承就绪检查（``_get_ready_backends``）、
    mode/strategy 路由、无命中哨兵与字符上限（见 ``tool_registry._execute_rag``）。固定 fact
    策略，避免误走每次内部跑 2 次 LightRAG 查询的 ``graph_augmented_retrieve``。``asyncio.wait_for``
    超时/失败一律降级为空串——绝不让预检索反而拖慢主链路。

    消融开关：``context.metadata["kb_seed"]`` 覆盖 settings（对齐 research_observer / solve_force_replan
    范式），评测 harness 可 per-task 强制 on/off。返回拼接好的 ``header + 证据`` 文本，空串=不注入。
    """
    cfg = get_settings().kb_seed
    # 消融开关：metadata 覆盖优先（None=回落 settings.kb_seed.enabled）
    _override = context.metadata.get("kb_seed")
    enabled = cfg.enabled if _override is None else bool(_override)
    query = (context.user_message or "").strip()
    if not enabled or "rag" not in (context.enabled_tools or []) or not query:
        return ""

    from core.agent.tool_registry import execute_tool
    args = {"query": query, "mode": context.rag_mode or "auto", "strategy": "fact"}
    _t0 = time.perf_counter()
    await stream.emit({"type": "tool_call", "tool": "rag", "input": args})
    result = None
    try:
        result = await asyncio.wait_for(
            execute_tool("rag", course_id=context.course_id, user_id=context.user_id, **args),
            timeout=cfg.timeout_s,
        )
    except Exception:
        logger.exception("kb_seed 预检索失败 course=%s", context.course_id)
    text = str(result.content).strip() if (result and result.success) else ""
    log_flow("chat.kb_seed", hit=bool(text), chars=len(text),
             elapsed_ms=int((time.perf_counter() - _t0) * 1000))
    if not text:
        return ""
    if len(text) > cfg.max_chars:
        text = text[: cfg.max_chars].rstrip() + "\n...[已截断]"
    await stream.emit({"type": "tool_result", "tool": "rag", "content": text[:2000]})
    return f"{header}\n\n{text}" if header else text


def assemble_system_prompt(
    *,
    loop_system: str,
    course_prompt: str,
    bot_persona: str = "",
    always_skills: str = "",
    skills_manifest: str = "",
    tool_hint_text: str = "",
    extended_tools_manifest: str = "",
    memory_context: str = "",
    mastery_context: str = "",
    session_summary: str = "",
    now_text: str = "",
) -> str:
    """按「稳定性递减」拼装 chat system prompt，最大化 prefix cache 命中。

    deepseek/qwen 的 prefix cache 要求多次请求的 system prompt 前缀逐字一致才命中
    （langsmith 里看 cached_tokens）。越稳定的段越靠前，同课程/同用户的多轮请求就能
    共享越长的前缀。各段稳定性依据（见 ChatPipeline.run 取值 + build_tool_hint_text）：

      L0 全局稳定 : loop_system      — chat.yaml，所有请求逐字一致（最稳，放最前）
      L1 课程级   : course_prompt    — 教师为课程配的 system_prompt，同课程一致
      L1 课程级   : bot_persona      — web 恒空(一致)；仅 IM bot 注入非空人设
      L1 课程级   : always_skills    — always-on skill 全文，课程配置为主
      L1/用户     : skills_manifest  — skill 清单，课程级为主；建了 personal skill 的用户分叉
      L2 用户级   : tool_hint_text   — 依赖 enabled_tools + skills/extended 的「有无」
                                       (build_tool_hint_text 仅按有无追加 read_skill/load_tools
                                       提示)，故放在 skills 之后、extended 之前——extended
                                       内容变化不波及更靠前 tool_hint 的 cache 命中
      L2 用户级   : extended_tools_manifest — MCP 个人启用集合，内容随用户变
      L2 用户级   : memory_context   — 每用户每轮可能不同
      L2 用户级   : mastery_context  — 掌握度薄弱点，紧邻 memory（同为用户级易变，不破前缀）
      L2 用户级   : session_summary  — 每隔几轮才更新
      L3 每请求   : now_text         — 当前时间，永远最后

    纯函数：只做「定序 + 过滤空段 + 以空行连接」，不读 DB/时间/配置/格式化，便于
    行为测试（tests/test_prompt_assembly.py）。
    """
    parts = [
        loop_system,
        course_prompt,
        bot_persona,
        always_skills,
        skills_manifest,
        tool_hint_text,
        extended_tools_manifest,
        memory_context,
        mastery_context,
        session_summary,
        now_text,
    ]
    return "\n\n".join(p for p in parts if p)


class ChatPipeline:
    """Run the chat agent loop."""

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.prompt_loader import load_prompt_dict
        _t0 = time.perf_counter()

        # ── 解析对话供应商 runtime（client / text_model / binding）─────────────
        # 四条 pipeline 共享步骤（pipeline_common）：用户自配 > 平台 profile > 全 None
        # （loop 回退全局 _default_client + TEXT_MODEL）。binding 决定图片直注格式。
        rt = await resolve_profile_runtime(context.llm_profile_id, context.user_id)
        log_flow("chat.profile_resolved",
                 profile_id=context.llm_profile_id or "default",
                 user_provider=bool(context.user_id),
                 model_override=str(rt.text_model) if rt.text_model else "none")

        # ── 成本配额（第四批）：超 daily_budget → 降级到便宜档 fast_model（只降级不拒绝）──
        # 软限流：硬拒绝会造成「上一轮刚好超限→下一轮被锁死」；降级 cheaper model 控本又保可用。
        # 仅平台 profile 有 fast_model 时实际切换；用户自配 provider 或无便宜档→仅记录不降级。
        _cq = get_settings().cost_quota
        if _cq.enabled and _cq.degrade_model and context.user_id:
            from core.quota.cost_quota import check_quota
            over, used, budget = await check_quota(context.user_id, context.course_id)
            if over:
                from dataclasses import replace
                from core.llm.catalog import (
                    active_profile_id_cached,
                    get_profile_cached,
                    profile_fast_model,
                )
                _pid = (context.llm_profile_id or "").strip() or await active_profile_id_cached()
                _prof = await get_profile_cached(_pid)
                _fast = profile_fast_model(_prof) if _prof else None
                if _fast and _fast != rt.text_model:
                    rt = replace(rt, text_model=_fast)
                    log_flow("chat.cost_quota_degraded",
                             user_id=context.user_id, course_id=context.course_id,
                             used_usd=used, budget_usd=budget,
                             from_model=context.llm_profile_id or "default",
                             to_model=_fast)
                else:
                    log_flow("chat.cost_quota_over_no_fast_model",
                             user_id=context.user_id, course_id=context.course_id,
                             used_usd=used, budget_usd=budget)

        # ── 通用上下文层（course_prompt / bot_persona / always_skills / memory / session / now）──
        # include_skills=True：chat 挂 read_skill 工具，急切注入 always-on skill 全文。
        layers = await build_common_context_layers(context, include_skills=True)

        # ── chat agent loop 行为规范（YAML 外部化，agentic_chat.yaml）──
        chat_cfg = load_prompt_dict(_CHAT_PROMPT_PATH)
        loop_system = (chat_cfg.get("loop") or {}).get("system") or ""

        # ── Skill 渐进式揭示：渲染 manifest 清单（always 全文已在 layers.always_skills）──
        from core.skills.skill_service import get_skill_service, render_skills_manifest
        svc = get_skill_service(context.course_id, context.user_id)
        context.skills_manifest = render_skills_manifest(svc.summary_entries())

        # ── MCP 个人启用开关：查出 user 启用的 server 集合，供 dynamic_tools 过滤 ──
        # 无记录 → None（默认全部，向后兼容）；有记录 → 仅启用集合内 server 的工具可见
        if context.user_id:
            try:
                from core.db.database import AsyncSessionLocal, UserMCPEnrollment
                from sqlalchemy import select
                async with AsyncSessionLocal() as db:
                    rows = (await db.execute(
                        select(UserMCPEnrollment.server_name, UserMCPEnrollment.enabled)
                        .where(UserMCPEnrollment.user_id == context.user_id)
                    )).all()
                context.metadata["mcp_enabled_servers"] = (
                    None if not rows else {r[0] for r in rows if r[1]}
                )
            except Exception:
                logger.exception("load user mcp enrollment failed user=%s", context.user_id)

        # ── 组装 tool_schemas + deferred loader（resolve 内设置 extended_tools_manifest）──
        from core.agentic.dynamic_tools import resolve as resolve_tool_schemas
        tool_schemas, loader_token = resolve_tool_schemas(context)

        # ── 工具使用提示 hint（单一数据源，取代手写在 chat.yaml 的工具说明）──
        from core.agent.prompting import build_tool_hint_text
        tool_hint_text = build_tool_hint_text(
            context.enabled_tools,
            context.language,
            skills_manifest=context.skills_manifest,
            extended_tools_manifest=context.extended_tools_manifest,
        )

        # ── 按「稳定性递减」拼装 system prompt（见 assemble_system_prompt docstring）──
        # 通用 6 层取自 layers，chat 专属 4 层（loop_system / skills_manifest / tool_hint /
        # extended）本地组装。assemble_system_prompt 是 prefix-cache 资产，段顺序不动。
        prompt_kwargs = dict(
            loop_system=loop_system,
            course_prompt=layers.course_prompt,
            bot_persona=layers.bot_persona,
            always_skills=layers.always_skills,
            skills_manifest=context.skills_manifest,
            tool_hint_text=tool_hint_text,
            extended_tools_manifest=context.extended_tools_manifest,
            memory_context=layers.memory_context,
            mastery_context=layers.mastery_context,
            session_summary=layers.session_summary,
            now_text=layers.now_text,
        )
        system_prompt = assemble_system_prompt(**prompt_kwargs)

        # cache_control：算 T1 稳定前缀字符偏移存入 metadata，loop._build_messages 据此在 T1/T2
        # 边界把 system 拆成两条消息、适配器给 T1 放 ephemeral 断点命中≈0.1x 成本。取 prompt_kwargs
        # 前 7 段（与 assemble_system_prompt 的 T1 段同序同空段过滤），用字符偏移精确切齐拼装字符串。
        from core.agentic.context_budget import system_t1_chars
        context.metadata["_system_t1_chars"] = system_t1_chars(
            list(prompt_kwargs.values())[:7]
        )

        log_flow("chat.prompt_assembled",
                 parts=sum(1 for v in prompt_kwargs.values() if v),
                 system_prompt_chars=len(system_prompt),
                 tools_count=len(tool_schemas) if tool_schemas else 0,
                 skills_manifest=bool(context.skills_manifest),
                 elapsed_ms=int((time.perf_counter() - _t0) * 1000))
        # system prompt 预算护栏：token_accounting_enabled 时按 token 口径告警 + 逐切片分解
        # （比字符口径准，定位膨胀层）；否则保留旧字符口径。只告警不截断--system prompt 静默
        # 截断会破坏 prefix cache 且难排查。Phase 2 协调器接管后此处改走 T2 集体裁剪。
        _cb = get_settings().context_budget
        if _cb.token_accounting_enabled:
            from core.agentic.context_budget import token_count_slices
            _slice_tokens = token_count_slices(prompt_kwargs)
            _total_t = sum(_slice_tokens.values())
            if _total_t > _cb.system_prompt_warn_tokens:
                _seg = sorted(_slice_tokens.items(), key=lambda x: -x[1])
                logger.warning(
                    "system_prompt 超预算阈值 %d > %d tokens；各段 token(降序): %s",
                    _total_t, _cb.system_prompt_warn_tokens, _seg,
                )
        elif len(system_prompt) > _SYSTEM_PROMPT_WARN_CHARS:
            seg = sorted(
                ((k, len(v or "")) for k, v in prompt_kwargs.items() if v),
                key=lambda x: -x[1],
            )
            logger.warning(
                "system_prompt 超预算阈值 %d > %d chars；各段字符数(降序): %s",
                len(system_prompt), _SYSTEM_PROMPT_WARN_CHARS, seg,
            )

        # ── KB Seed 预检索：进 loop 前用原问题查一次课程知识库，命中则前置注入 ──
        # 必须在 describe_images 之前取原始 user_message 做 query（图描述会改写 user_message）。
        # 命中作为 [知识库预检索] 块经 extra_context 注入首轮消息，材料够时第 1 轮直接作答；
        # 未命中/超时/未挂 rag → 返回空串，loop 行为零变化。
        kb_seed = await retrieve_kb_seed(
            context, stream, header=(chat_cfg.get("kb_seed") or {}).get("header") or ""
        )

        # ── 两阶段图片处理（rt.text_model/binding 判断主模型能否看图）──────────
        # 主模型不支持 vision → 视觉模型描述成文字；支持 → 跳过，走 loop 内直注。
        context.user_message = await describe_images(context, context.user_message, rt)

        # cron owner 注入（bot 对话：loop._run_turn 写入 metadata；web 无则不 set，cron 工具不挂载）
        from core.bot.cron_tool import reset_cron_owner, set_cron_owner
        # read_skill 同轮去重集合（turn 级：每 turn set 空 / finally reset）
        from core.agentic.dynamic_tools import (
            reset_deferred_loader,
            reset_read_skill_log,
            set_read_skill_log,
        )
        cron_token = set_cron_owner(context.metadata.get("cron_owner"))
        rs_token = set_read_skill_log()
        try:
            await run_agent_loop(
                context=context,
                stream=stream,
                system_prompt=system_prompt,
                tool_schemas=tool_schemas,
                client=rt.client,
                model=rt.text_model,
                binding=rt.binding,
                extra_context=kb_seed,
                temperature=_CHAT_TEMPERATURE,
            )
        finally:
            reset_cron_owner(cron_token)
            reset_deferred_loader(loader_token)
            reset_read_skill_log(rs_token)


__all__ = ["ChatPipeline"]
