"""第二批上下文预算治理回归测试。

覆盖：
- cap_tool_result：头尾保留/短串不动/<=0 不限
- mask_old_observations：8 轮 20k 字 → 最近 3 轮 tool 原文完整、更早掩码、总量落预算内
- mask 不动 assistant/system
- apply（settings 开启）：端到端掩码生效
- skill_service.load_for_context：5+1 个大 always skill → 裁到预算内、personal 优先保留
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.agentic.context_policy import (
    _MASK_MARKER,
    apply,
    apply_arm,
    cap_tool_result,
    current_arm,
    mask_old_observations,
    reset_arm,
    set_arm,
)
from core.skills.skill_service import SkillService, _ALWAYS_MAX_CHARS


# ---------------------------------------------------------------------------
# cap_tool_result
# ---------------------------------------------------------------------------

def test_cap_single_paragraph_falls_back_to_head_tail():
    """单段（无空行边界）退化走 _cap_head_tail：头尾各半 + 省略标注，行为等价旧实现。"""
    out = cap_tool_result("X" * 1000, max_chars=100)
    assert len(out) <= 100
    assert out.startswith("X")  # 头部保留
    assert out.endswith("X")    # 尾部保留（退化路径仍头尾各半）
    assert "省略" in out


def test_cap_short_unchanged():
    assert cap_tool_result("short", max_chars=100) == "short"
    assert cap_tool_result("正好", max_chars=2) == "正好"


def test_cap_zero_or_negative_means_unlimited():
    assert cap_tool_result("X" * 1000, max_chars=0) == "X" * 1000
    assert cap_tool_result("X" * 1000, max_chars=-1) == "X" * 1000


def test_cap_respects_paragraph_boundaries():
    """多段输入：头部完整段保留、尾部段被省略、总长落预算内（不从段中间砍）。"""
    paras = [f"第{i}段，这是一些完整的句子内容。" for i in range(6)]
    content = "\n\n".join(paras)
    out = cap_tool_result(content, max_chars=60)
    assert len(out) <= 60
    assert "省略" in out
    # 首段 + 第二段完整保留（未被从中间砍断）
    assert out.startswith("第0段，这是一些完整的句子内容。")
    assert "第1段，这是一些完整的句子内容。" in out
    # 尾部段被省略
    assert "第5段" not in out


def test_cap_keeps_source_suffix():
    """尾部「来源」段优先保留，即便 head 被大幅省略也不丢来源列表。"""
    body = "这是正文内容，足够长需要被截断。" * 5  # ~80 字
    source = "来源：高等数学教材第3章第5节"
    content = body + "\n\n" + source
    out = cap_tool_result(content, max_chars=50)
    assert len(out) <= 50
    assert "来源" in out  # 尾部来源段保留


def test_cap_truncates_last_paragraph_at_sentence():
    """最后一段放不下时在句号处截断取前缀（不在句子/词中间断开）。"""
    paras = [
        "短。",
        "这是一个很长的第二段包含好几个句子。第二句在这里继续展开。第三句还有更多内容。",
        "尾段也比较长，包含一些额外信息。",
    ]
    content = "\n\n".join(paras)
    out = cap_tool_result(content, max_chars=43)
    assert len(out) <= 43
    # 第二段前缀保留到第一个句号（截断点在句号边界，不在词中间）
    assert "这是一个很长的第二段包含好几个句子。" in out
    assert "第二句在这里继续展开" not in out  # 第一个句号之后被省略


# ---------------------------------------------------------------------------
# mask_old_observations
# ---------------------------------------------------------------------------

def _mk_rounds(n: int, content_len: int = 20000) -> list[dict]:
    """构造 n 个工具轮：每轮 assistant(tool_calls) + role=tool(大结果)，末尾加最终答案。"""
    msgs: list[dict] = []
    for i in range(n):
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "rag", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "name": "rag", "content": "X" * content_len})
    msgs.append({"role": "assistant", "content": "最终答案"})
    return msgs


def test_mask_keeps_recent_3_rounds_full():
    msgs = _mk_rounds(8)
    masked = mask_old_observations(msgs, keep_recent_turns=3, budget_chars=80_000)
    by_id = {m["tool_call_id"]: m for m in msgs if m["role"] == "tool"}
    # 最近 3 轮（c5/c6/c7）tool 原文完整
    for i in (5, 6, 7):
        assert by_id[f"c{i}"]["content"] == "X" * 20000
    # 最早轮被掩码
    assert by_id["c0"]["content"] == _MASK_MARKER
    # 总字符降到预算内
    assert sum(len(str(m.get("content", ""))) for m in msgs) <= 80_000
    assert masked >= 1


def test_mask_noop_under_budget():
    msgs = _mk_rounds(3, content_len=100)  # 总量远小于预算
    masked = mask_old_observations(msgs, keep_recent_turns=3, budget_chars=80_000)
    assert masked == 0
    # 没有任何 tool 被掩码
    assert all(m["content"] == "X" * 100 for m in msgs if m["role"] == "tool")


def test_mask_preserves_assistant_and_system():
    msgs = _mk_rounds(8)
    msgs.insert(0, {"role": "system", "content": "系统提示不应被掩码"})
    mask_old_observations(msgs, keep_recent_turns=3, budget_chars=80_000)
    # system 与最终答案 assistant 不受影响
    assert msgs[0]["content"] == "系统提示不应被掩码"
    assert msgs[-1]["content"] == "最终答案"
    # assistant(带 tool_calls) 的 content 不被改（掩码只动 role=tool）
    assts = [m for m in msgs if m["role"] == "assistant"]
    assert all(a["content"] == "" or a["content"] == "最终答案" for a in assts)


def test_mask_keep_zero_masks_all_tools():
    """M=0：所有工具轮掩码，但最终答案 assistant 不受影响（mask 只动 tool）。"""
    msgs = _mk_rounds(4, content_len=10000)
    masked = mask_old_observations(msgs, keep_recent_turns=0, budget_chars=1)
    tools = [m for m in msgs if m["role"] == "tool"]
    assert all(t["content"] == _MASK_MARKER for t in tools)
    assert msgs[-1]["content"] == "最终答案"
    assert masked >= 1


# ---------------------------------------------------------------------------
# apply（settings 开启时端到端）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_masks_when_enabled(monkeypatch):
    from settings import get_settings
    cfg = get_settings().context_policy
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "summary_enabled", False)
    monkeypatch.setattr(cfg, "budget_chars", 80_000)
    monkeypatch.setattr(cfg, "keep_recent_turns", 3)

    msgs = _mk_rounds(8)
    await apply(msgs, "deepseek-v4-pro")
    by_id = {m["tool_call_id"]: m for m in msgs if m["role"] == "tool"}
    assert by_id["c7"]["content"] == "X" * 20000  # 最近轮保真
    assert by_id["c0"]["content"] == _MASK_MARKER  # 最早轮掩码


@pytest.mark.asyncio
async def test_apply_noop_when_under_budget(monkeypatch):
    from settings import get_settings
    cfg = get_settings().context_policy
    monkeypatch.setattr(cfg, "enabled", True)
    monkeypatch.setattr(cfg, "summary_enabled", False)

    msgs = _mk_rounds(2, content_len=100)
    await apply(msgs, "deepseek-v4-pro")
    assert all(m["content"] == "X" * 100 for m in msgs if m["role"] == "tool")


# ---------------------------------------------------------------------------
# skill_service.load_for_context 预算
# ---------------------------------------------------------------------------

def _write_skill(root: Path, name: str, body: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")


def test_load_for_context_budget_truncation(tmp_path):
    """5 个 course + 1 个 personal，单个 ~1.8k 字、合计超预算 → 从尾部裁 course，personal 优先保留。

    单个 skill 必须 < _ALWAYS_MAX_CHARS，否则裁剪守卫 len(parts)>1 会停在只剩 1 个高优先级
    skill（仍超预算但无法再裁）——那是正确行为，但测不到「pop 多个低优先级」的路径。
    """
    body = "规则条目行\n" * 300  # ~1.8k 字（单个 < 预算 8000，合计超）
    # personal 高优先级
    _write_skill(tmp_path / "personal", "p-skill",
                 f"---\nname: p-skill\ndescription: p\nalways: true\n---\n{body}")
    # course 低优先级 ×5
    for i in range(5):
        _write_skill(tmp_path / "course", f"c-skill-{i}",
                     f"---\nname: c-skill-{i}\ndescription: c\nalways: true\n---\n{body}")
    svc = SkillService(
        user_root=tmp_path / "course",
        builtin_root=None,
        personal_root=tmp_path / "personal",
    )
    out = svc.load_for_context(
        ["p-skill", "c-skill-0", "c-skill-1", "c-skill-2", "c-skill-3", "c-skill-4"]
    )
    # 裁到预算内（省略行有少量余量）
    assert len(out) <= _ALWAYS_MAX_CHARS + 100
    # personal 高优先级保留
    assert "p-skill" in out
    # 低优先级被裁（至少裁掉若干 course skill）
    assert "已省略" in out


def test_load_for_context_no_budget_pressure(tmp_path):
    """小 skill 不触发裁剪，行为等价（无省略行）。"""
    _write_skill(tmp_path / "course", "small",
                 "---\nname: small\ndescription: s\nalways: true\n---\n短守则")
    svc = SkillService(user_root=tmp_path / "course", builtin_root=None)
    out = svc.load_for_context(["small"])
    assert "短守则" in out
    assert "已省略" not in out


# ---------------------------------------------------------------------------
# apply_arm 评测四臂（contextvar 覆盖层，对照 arXiv:2508.21433）+ contextvar 切换
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_apply_arm_raw_does_nothing():
    """raw：完全不裁（论文真基线）。"""
    msgs = _mk_rounds(5)
    before = [m["content"] for m in msgs if m["role"] == "tool"]
    extra = await apply_arm(msgs, "fake-model", "raw")
    after = [m["content"] for m in msgs if m["role"] == "tool"]
    assert extra == 0
    assert before == after


@pytest.mark.asyncio
async def test_apply_arm_masking_masks():
    msgs = _mk_rounds(5)
    extra = await apply_arm(msgs, "fake-model", "masking")
    assert extra == 0
    assert any(_MASK_MARKER in str(m["content"])
               for m in msgs if m["role"] == "tool")


@pytest.mark.asyncio
async def test_apply_arm_summary_only(monkeypatch):
    """summary_only：窗口外每一轮 tool 结果各摘一次（H2 关键臂）。mock 压缩 LLM。"""
    from core.agentic import context_policy as cp

    async def fake_summarize(text, model):
        return "摘要"
    monkeypatch.setattr(cp, "_summarize_masked_text", fake_summarize)

    msgs = _mk_rounds(5)
    extra = await apply_arm(msgs, "fake-model", "summary_only")
    # 5 轮 - keep_recent(3) = 窗口外 2 轮，每轮一次摘要
    assert extra == 2
    assert any("早期工具结果摘要" in str(m["content"])
               for m in msgs if m["role"] == "tool")


@pytest.mark.asyncio
async def test_apply_arm_hybrid_under_threshold_no_summary(monkeypatch):
    """被掩码轮数 < summary_threshold(默认 4) 时不触发摘要。"""
    from core.agentic import context_policy as cp

    called = []

    async def fake_summarize(text, model):
        called.append(text)
        return "摘要"
    monkeypatch.setattr(cp, "_summarize_masked_text", fake_summarize)

    extra = await apply_arm(_mk_rounds(5), "fake-model", "hybrid")  # 掩码 1 轮 < 4
    assert extra == 0
    assert called == []


@pytest.mark.asyncio
async def test_apply_arm_hybrid_over_threshold(monkeypatch):
    """被掩码轮数 >= 阈值时触发一次整体摘要。"""
    from core.agentic import context_policy as cp

    async def fake_summarize(text, model):
        return "摘要"
    monkeypatch.setattr(cp, "_summarize_masked_text", fake_summarize)

    msgs = _mk_rounds(8)  # 掩码 4 轮 >= 4
    extra = await apply_arm(msgs, "fake-model", "hybrid")
    assert extra == 1
    assert any("[早期工具结果摘要]" in str(m["content"])
               for m in msgs if m["role"] == "tool")


@pytest.mark.asyncio
async def test_apply_arm_unknown_arm_defaults_raw():
    extra = await apply_arm(_mk_rounds(3, content_len=10), "fake-model", "bogus")
    assert extra == 0


# ── contextvar 覆盖层（harness per-task 切臂用）──
def test_contextvar_set_current_reset():
    assert current_arm() is None          # 默认未设（生产回落 settings）
    token = set_arm("masking")
    assert current_arm() == "masking"
    reset_arm(token)
    assert current_arm() is None
