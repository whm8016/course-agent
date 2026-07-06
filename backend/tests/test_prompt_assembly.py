"""Prompt 拼装行为契约 —— 锁定 assemble_system_prompt 的不变量。

仿 test_settings.py 的契约测试风格：不启动 LLM/DB，直接调纯函数，断言段顺序、
空段过滤、连接符、以及 prefix-cache 共享前缀这些行为不变量。

为什么锁顺序：deepseek/qwen 的 prefix cache 要求多次请求的 system prompt 前缀逐字
一致才命中。assemble_system_prompt 按「稳定性递减」定序（loop_system/course_prompt
最稳 → … → now_text 每请求必变）。这里固化该顺序，防止未来误改把易变段提前、悄悄
打散缓存（langsmith 里表现为 cached_tokens 掉零）。
"""
from __future__ import annotations

from core.capabilities.chat_pipeline import assemble_system_prompt

# 段顺序约定标记（位置即稳定性层级，越靠前越稳）
_LS, _CP, _BP, _AS, _SM, _TH, _ET, _MC, _SS, _NT = (
    "[LS]", "[CP]", "[BP]", "[AS]", "[SM]", "[TH]", "[ET]", "[MC]", "[SS]", "[NT]"
)


# ── 段顺序契约（稳定性递减）──────────────────────────────────────────────

def test_full_order_is_stable_decreasing():
    """10 段全在时，顺序严格为 loop→course→persona→always→skills→hint→extended→memory→summary→now。"""
    p = assemble_system_prompt(
        loop_system=_LS, course_prompt=_CP, bot_persona=_BP,
        always_skills=_AS, skills_manifest=_SM, tool_hint_text=_TH,
        extended_tools_manifest=_ET, memory_context=_MC,
        session_summary=_SS, now_text=_NT,
    )
    markers = [_LS, _CP, _BP, _AS, _SM, _TH, _ET, _MC, _SS, _NT]
    positions = [p.index(m) for m in markers]
    assert positions == sorted(positions), f"段顺序错乱: {markers} → 位置 {positions}"


def test_loop_system_is_first_segment():
    p = assemble_system_prompt(loop_system=_LS, course_prompt=_CP,
                               memory_context=_MC, now_text=_NT)
    assert p.startswith(_LS)


def test_now_text_is_always_last():
    # 满段
    assert assemble_system_prompt(
        loop_system=_LS, course_prompt=_CP, now_text=_NT,
        memory_context=_MC, session_summary=_SS,
    ).endswith(_NT)
    # 仅 loop/course/now
    assert assemble_system_prompt(
        loop_system=_LS, course_prompt=_CP, now_text=_NT,
    ).endswith(_NT)


def test_course_prompt_precedes_memory():
    p = assemble_system_prompt(loop_system=_LS, course_prompt=_CP, memory_context=_MC)
    assert p.index(_CP) < p.index(_MC)


def test_skills_precede_tool_hint_and_extended():
    """skills(课程级) → tool_hint(依赖 skills/extended「有无」) → extended(用户级)。

    tool_hint 放 skills 之后、extended 之前：extended 内容变化不波及更靠前 tool_hint
    的 cache 命中（build_tool_hint_text 仅按 extended 有无追加 load_tools 提示）。
    """
    p = assemble_system_prompt(loop_system=_LS, course_prompt=_CP,
                               skills_manifest=_SM, tool_hint_text=_TH,
                               extended_tools_manifest=_ET)
    assert p.index(_SM) < p.index(_TH) < p.index(_ET)


def test_memory_precedes_summary():
    p = assemble_system_prompt(loop_system=_LS, course_prompt=_CP,
                               memory_context=_MC, session_summary=_SS)
    assert p.index(_MC) < p.index(_SS)


# ── 空段过滤 + 连接符 ────────────────────────────────────────────────────

def test_segments_joined_by_blank_line():
    p = assemble_system_prompt(loop_system="A", course_prompt="B", now_text="C")
    assert p == "A\n\nB\n\nC"


def test_optional_empty_dropped_leaves_loop_and_course():
    p = assemble_system_prompt(loop_system=_LS, course_prompt=_CP)
    assert p == f"{_LS}\n\n{_CP}"


def test_everything_empty_returns_empty_string():
    assert assemble_system_prompt(loop_system="", course_prompt="") == ""


def test_default_bot_persona_omitted_for_web():
    """web 用户 bot_persona 恒空（默认 ""）→ 不出现人设段。"""
    p = assemble_system_prompt(loop_system=_LS, course_prompt=_CP)  # bot_persona 默认 ""
    assert _BP not in p


def test_no_content_mutation_pure_function():
    """纯函数不 strip / 不改内容；bot_persona 的 strip 责任在调用处，不在纯函数。"""
    p = assemble_system_prompt(loop_system="  [LS]  ", course_prompt="\n[CP]\n")
    assert "  [LS]  " in p
    assert "\n[CP]\n" in p


# ── prefix cache 共享前缀（重排的核心收益）────────────────────────────────

def test_same_course_shares_prefix_through_extended():
    """同课程两请求（不同 memory/now）：loop→…→extended 段逐字一致，memory 才分叉。

    这段共享前缀（常含长 course_prompt）正是 prefix cache 命中的部分，每请求省下
    其重复计算。
    """
    course_prompt = "课程人设" * 50  # 模拟较长的 course_prompt
    common = dict(loop_system=_LS, course_prompt=course_prompt, bot_persona="",
                  always_skills=_AS, skills_manifest=_SM, tool_hint_text=_TH,
                  extended_tools_manifest=_ET)
    r1 = assemble_system_prompt(memory_context=_MC, now_text=_NT, **common)
    r2 = assemble_system_prompt(memory_context="[MC2]", session_summary=_SS, now_text="[NT2]", **common)
    shared = r1[:r1.index(_MC)]
    assert r2.startswith(shared), "同课程请求在 memory 之前的前缀应逐字一致（cache 命中）"
    assert course_prompt in shared  # 长 course_prompt 完整落在共享前缀内


def test_web_users_share_prefix_before_memory():
    """两个 web 用户（均无 persona）、同课程 → memory 之前前缀逐字一致。"""
    base = dict(loop_system=_LS, course_prompt=_CP, bot_persona="",
                always_skills=_AS, skills_manifest=_SM)
    u1 = assemble_system_prompt(memory_context="用户A记忆", now_text="t1", **base)
    u2 = assemble_system_prompt(memory_context="用户B记忆", now_text="t2", **base)
    assert u1.split("用户A记忆")[0] == u2.split("用户B记忆")[0]


def test_empty_persona_slot_does_not_fork_prefix():
    """bot_persona 空（web）不占用前缀位置 → 两 web 请求前缀仍一致，而非因 persona 槽存在而分叉。"""
    web1 = assemble_system_prompt(loop_system=_LS, course_prompt=_CP, bot_persona="",
                                  memory_context="m1", now_text="t1")
    web2 = assemble_system_prompt(loop_system=_LS, course_prompt=_CP, bot_persona="",
                                  memory_context="m2", now_text="t2")
    assert web1[:web1.index("m1")] == web2[:web2.index("m2")]
