"""上下文预算策略消融 runner：遍历 CONTEXT_POLICY_CONFIGS，每臂跑 run_agent_loop。

对照 arXiv:2508.21433：量化各臂的 token 成本（input/output）、trajectory 轮数、
额外压缩 LLM 调用（summary/hybrid 臂的 _cp_extra_llm_calls）、答案长度。raw 是真基线
（成本爆炸）；masking 成本减半解题率持平；summary_only trajectory elongation；hybrid 最优。

复用 Batch 2 已建的 context_policy.set_arm/apply_arm（contextvar 覆盖层）：runner
``set_arm(arm)`` + 覆盖 ``settings.context_policy.keep_recent_turns``，``loop.py:466`` 自动按臂
应用，生产代码零侵入——arm 经 contextvar 注入，串行评测 set→跑→reset 不污染下一组合。

每条 case 的 metrics 取自 LoopOutcome（rounds/tools_used/final_text）+ loop 写入的
context.metadata（llm_usage 的 input/output tokens、_cp_extra_llm_calls 的压缩调用数）。
返回 ``{config_label: [per-item metrics]}``。
"""
from __future__ import annotations

import asyncio
import logging

from . import config

logger = logging.getLogger(__name__)


async def run_ablation(
    items: list[dict],
    configs: list[dict] | None = None,
    *,
    max_iterations: int | None = None,
    loop_fn=None,
) -> dict[str, list[dict]]:
    """对每个 context-policy 配置跑 agent loop，收集 trajectory/cost 指标。

    Args:
        items: 评测集，每项含 ``id``/``input``/``metadata{mode,course_id,rag_mode,enabled_tools?}``。
        configs: ``config.CONTEXT_POLICY_CONFIGS``（或子集）。
        max_iterations: loop 最大轮数；None → ``config.MAX_ITERATIONS``。
        loop_fn: 注入的 run_agent_loop（测试用，默认惰性 import 生产实现）。
    """
    from core.agentic.context_policy import reset_arm, set_arm
    from core.context import UnifiedContext
    from core.stream_bus import StreamBus
    from settings import get_settings

    configs = config.CONTEXT_POLICY_CONFIGS if configs is None else configs
    max_iter = config.MAX_ITERATIONS if max_iterations is None else max_iterations
    cp_cfg = get_settings().context_policy

    async def _default_loop(**kwargs):
        from core.agentic.loop import run_agent_loop
        return await run_agent_loop(**kwargs)

    _loop = loop_fn if loop_fn is not None else _default_loop

    all_results: dict[str, list[dict]] = {}
    for cfg in configs:
        label, arm, m = cfg["label"], cfg["arm"], cfg["keep_recent_turns"]
        logger.info("=== context 消融: %s (arm=%s M=%d) ===", label, arm, m)
        results: list[dict] = []
        # 覆盖 M（mask_old_observations / _summarize_old_turns 读 cp_cfg.keep_recent_turns）
        orig_m = cp_cfg.keep_recent_turns
        cp_cfg.keep_recent_turns = m
        token = set_arm(arm)
        try:
            for item in items:
                meta = item.get("metadata") or {}
                ctx = UnifiedContext(
                    course_id=meta.get("course_id", "general"),
                    user_id="eval",
                    user_message=item.get("input", ""),
                    mode=meta.get("mode", "chat"),
                    enabled_tools=meta.get("enabled_tools", ["rag"]),
                    rag_mode=meta.get("rag_mode", "naive"),
                )
                rec: dict = {"id": item.get("id"), "arm": arm, "label": label}
                try:
                    outcome = await _loop(
                        context=ctx, stream=StreamBus(),
                        system_prompt="", tool_schemas=None,
                        max_iterations=max_iter,
                    )
                    usage = ctx.metadata.get("llm_usage") or {}
                    rec.update({
                        "rounds": getattr(outcome, "rounds", 0),
                        "tools_used": len(getattr(outcome, "tools_used", []) or []),
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "extra_llm_calls": ctx.metadata.get("_cp_extra_llm_calls", 0),
                        "answer_chars": len(getattr(outcome, "final_text", "") or ""),
                        "error": None,
                    })
                except Exception as e:  # noqa: BLE001
                    logger.error("[%s] %s loop 失败: %s", label, item.get("id"), e)
                    rec.update({"error": repr(e)})
                results.append(rec)
                logger.info(
                    "[%s] %s rounds=%s in=%s out=%s extra=%s",
                    label, item.get("id"), rec.get("rounds"),
                    rec.get("input_tokens"), rec.get("output_tokens"),
                    rec.get("extra_llm_calls"),
                )
                if config.QUERY_DELAY:
                    await asyncio.sleep(config.QUERY_DELAY)
        finally:
            reset_arm(token)
            cp_cfg.keep_recent_turns = orig_m
        all_results[label] = results
        logger.info("配置 %s 完成，%d 条结果", label, len(results))
    return all_results


__all__ = ["run_ablation"]
