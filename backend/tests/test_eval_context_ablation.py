"""第四批-2：上下文预算策略消融框架回归测试。

覆盖（不依赖真实 LLM，loop_fn 注入 mock）：
- CONTEXT_POLICY_CONFIGS 契约：每项含 label/arm/keep_recent_turns，arm 在 ARMS 内
- run_ablation 形状：遍历每个配置、每条 case 收集 trajectory/cost metrics
- contextvar 注入+还原：set_arm/reset_arm 成对（跑完 current_arm 回到 None）
- keep_recent_turns 覆盖+还原（settings 单例不被污染）
- error 容错：单 case loop 抛错不中断整批，rec 带 error
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_configs_contract():
    """每项含 label/arm/keep_recent_turns，arm 在 context_policy.ARMS 内，label 唯一。"""
    from core.agentic.context_policy import ARMS
    from scripts.eval_context.config import CONTEXT_POLICY_CONFIGS

    labels = [c["label"] for c in CONTEXT_POLICY_CONFIGS]
    assert len(labels) == len(set(labels)), "label 必须唯一"
    assert len(CONTEXT_POLICY_CONFIGS) >= 4  # 至少覆盖论文四臂
    arms_present = {c["arm"] for c in CONTEXT_POLICY_CONFIGS}
    assert {"raw", "masking", "summary_only", "hybrid"} <= arms_present
    for c in CONTEXT_POLICY_CONFIGS:
        assert c["arm"] in ARMS, f"非法 arm: {c['arm']}"
        assert isinstance(c["keep_recent_turns"], int) and c["keep_recent_turns"] >= 0


@pytest.mark.asyncio
async def test_run_ablation_collects_metrics_and_restores_state(monkeypatch):
    """mock loop_fn → 验证 metrics 收集 + contextvar/settings 还原。"""
    from core.agentic.context_policy import current_arm
    from scripts.eval_context.ablation_runner import run_ablation
    from settings import get_settings

    cp_cfg = get_settings().context_policy
    orig_m = cp_cfg.keep_recent_turns

    call_log: list[str] = []

    async def fake_loop(**kwargs):
        ctx = kwargs["context"]
        arm_seen = current_arm()  # 跑 loop 时 contextvar 应已被 set_arm 设成当前臂
        call_log.append(f"{ctx.user_message[:8]}|arm={arm_seen}|M={cp_cfg.keep_recent_turns}")
        # 模拟 loop 写入 metadata（真实 loop 在 cost 聚合段写）
        ctx.metadata["llm_usage"] = {"input_tokens": 100, "output_tokens": 50}
        ctx.metadata["_cp_extra_llm_calls"] = 7 if arm_seen == "summary_only" else 0
        return SimpleNamespace(rounds=3, tools_used=["rag"], final_text="答案文本")

    items = [
        {"id": "i1", "input": "问题一内容", "metadata": {"course_id": "c", "mode": "chat"}},
        {"id": "i2", "input": "问题二内容", "metadata": {"course_id": "c", "mode": "chat"}},
    ]
    configs = [
        {"label": "raw", "arm": "raw", "keep_recent_turns": 3},
        {"label": "summary_only", "arm": "summary_only", "keep_recent_turns": 3},
    ]

    results = await run_ablation(items, configs, loop_fn=fake_loop)

    # 每配置 2 条结果
    assert set(results.keys()) == {"raw", "summary_only"}
    assert len(results["raw"]) == 2 and len(results["summary_only"]) == 2

    # metrics 正确采集
    raw0 = results["raw"][0]
    assert raw0["rounds"] == 3 and raw0["tools_used"] == 1
    assert raw0["input_tokens"] == 100 and raw0["output_tokens"] == 50
    assert raw0["answer_chars"] == len("答案文本")
    assert raw0["extra_llm_calls"] == 0  # raw 无压缩调用
    # summary_only 臂记录了 7 次额外压缩调用
    assert results["summary_only"][0]["extra_llm_calls"] == 7

    # contextvar 注入：fake_loop 跑时 current_arm 是当前臂（call_log 印证）
    assert any("arm=raw" in x for x in call_log)
    assert any("arm=summary_only" in x for x in call_log)

    # 还原：跑完后 contextvar 回 None、keep_recent_turns 复原
    assert current_arm() is None
    assert cp_cfg.keep_recent_turns == orig_m


@pytest.mark.asyncio
async def test_run_ablation_tolerates_loop_error():
    """单 case loop 抛错不中断整批，该 rec 带 error，其余正常。"""
    from scripts.eval_context.ablation_runner import run_ablation

    async def flaky_loop(**kwargs):
        ctx = kwargs["context"]
        if ctx.user_message.startswith("炸"):
            raise RuntimeError("boom")
        ctx.metadata["llm_usage"] = {}
        return SimpleNamespace(rounds=1, tools_used=[], final_text="ok")

    items = [
        {"id": "ok1", "input": "正常问题", "metadata": {}},
        {"id": "bad", "input": "炸了", "metadata": {}},
        {"id": "ok2", "input": "另一个正常", "metadata": {}},
    ]
    configs = [{"label": "raw", "arm": "raw", "keep_recent_turns": 3}]
    results = await run_ablation(items, configs, loop_fn=flaky_loop)

    recs = results["raw"]
    assert len(recs) == 3
    assert recs[0]["error"] is None and recs[0]["rounds"] == 1
    assert recs[1]["error"] is not None
    assert recs[2]["error"] is None
