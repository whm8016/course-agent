"""P1：矛盾检测第二级——灰区（同 kind、相似度 [0.4, 0.85)）LLM 仲裁四选一。

覆盖路径：
- 灰区对 + 仲裁开 → LLM 被调，CONTRADICT/UPDATE/DUPLICATE 剔旧、SUPPLEMENT 都保留
- 规则区（相似度 > 0.85）走规则不调 LLM；低相似（< 0.4）与跨 kind 对不进仲裁
- 默认关；LLM 失败/非法输出按 SUPPLEMENT 保留（保守不丢条）
文本对的 SequenceMatcher 相似度实测：偏好反转对 0.647 / 习惯改变对 0.48（灰区）、
版本号对 0.909（规则区）。
"""
from unittest.mock import MagicMock, patch

import pytest

from core.memory import mem0_client


def _mem_settings(monkeypatch, arbitration=False):
    from settings import get_settings

    cfg = get_settings().mem0
    monkeypatch.setattr(cfg, "conflict_detect_enabled", True)
    monkeypatch.setattr(cfg, "search_threshold", 0.0)
    monkeypatch.setattr(cfg, "rerank_enabled", False)
    monkeypatch.setattr(cfg, "conflict_llm_arbitration_enabled", arbitration)
    monkeypatch.setattr(cfg, "conflict_llm_sim_floor", 0.4)


def _mk(results):
    m = MagicMock()

    async def fake_search(query, filters=None, top_k=5, **kw):
        kind = (filters or {}).get("kind")
        return {"results": [r for r in results if r["kind"] == kind]}

    m.search = fake_search
    return m


# 灰区对（sim≈0.647）：语义矛盾但措辞不同——difflib 抓不住，正是 LLM 仲裁的目标
_REVERSAL_OLD = {"kind": "persona", "id": "p-old", "memory": "学生偏好快速的讲解风格与简洁的总结",
                 "created_at": "2026-01-01T00:00:00+00:00", "metadata": {"topic_ids": []}}
_REVERSAL_NEW = {"kind": "persona", "id": "p-new", "memory": "学生偏好缓慢的讲解风格与详细的推导",
                 "created_at": "2026-02-01T00:00:00+00:00", "metadata": {"topic_ids": []}}


async def _run(monkeypatch, results, arbitration, llm_out=None, llm_raises=None):
    """跑全链路 build_memory_context（mock mem0 检索 + mock chat_complete），返回注入文本。"""
    _mem_settings(monkeypatch, arbitration=arbitration)
    llm_calls: list[str] = []

    async def fake_chat(system_prompt, history, user_message, **kw):
        llm_calls.append(user_message)
        if llm_raises:
            raise llm_raises
        return llm_out

    with (
        patch.object(mem0_client, "get_memory", lambda: _mk(results)),
        patch("core.llm.llm.chat_complete", fake_chat),
    ):
        out = await mem0_client.build_memory_context("u1", "问", course_id="")
    return out, llm_calls


@pytest.mark.asyncio
async def test_grayzone_contradict_drops_old(monkeypatch):
    """灰区对 + 仲裁判 CONTRADICT → 旧偏好被剔除，只注入新偏好。"""
    out, calls = await _run(
        monkeypatch, [_REVERSAL_OLD, _REVERSAL_NEW], arbitration=True, llm_out="CONTRADICT"
    )
    assert len(calls) == 1                      # 恰好一次仲裁调用
    assert "缓慢" in out and "快速" not in out  # 旧条剔除
    assert "p-old" in calls[0] or "快速" in calls[0]  # prompt 里带旧条文本


@pytest.mark.asyncio
async def test_grayzone_supplement_keeps_both(monkeypatch):
    """灰区对判 SUPPLEMENT（互为补充）→ 两条都注入（persona ≤3 上限内）。"""
    out, calls = await _run(
        monkeypatch, [_REVERSAL_OLD, _REVERSAL_NEW], arbitration=True, llm_out="SUPPLEMENT"
    )
    assert len(calls) == 1
    assert "快速" in out and "缓慢" in out


@pytest.mark.asyncio
async def test_rule_zone_skips_llm(monkeypatch):
    """相似度 0.909 > 0.85（规则区）→ 走时间差规则剔旧，不调 LLM。"""
    results = [
        {"kind": "persona", "id": "v-old", "memory": "学生使用第七版电路教材",
         "created_at": "2026-01-01T00:00:00+00:00", "metadata": {"topic_ids": []}},
        {"kind": "persona", "id": "v-new", "memory": "学生使用第八版电路教材",
         "created_at": "2026-03-01T00:00:00+00:00", "metadata": {"topic_ids": []}},
    ]
    out, calls = await _run(monkeypatch, results, arbitration=True, llm_out="CONTRADICT")
    assert calls == []                    # 规则区不进仲裁
    assert "第八版" in out and "第七版" not in out  # 规则保留较新


@pytest.mark.asyncio
async def test_low_similarity_and_cross_kind_skip_llm(monkeypatch):
    """低相似（<0.4）与跨 kind 对不进仲裁——LLM 零调用，两条原样注入。"""
    results = [
        {"kind": "persona", "id": "a", "memory": "学生目标是期末八十五分",
         "created_at": "2026-01-01T00:00:00+00:00", "metadata": {"topic_ids": []}},
        {"kind": "persona", "id": "b", "memory": "学生每晚复习两小时",
         "created_at": "2026-02-01T00:00:00+00:00", "metadata": {"topic_ids": []}},
        {"kind": "event", "id": "c", "memory": "学生偏好快速的讲解风格与简洁的总结",
         "created_at": "2026-02-02T00:00:00+00:00", "metadata": {"topic_ids": []}},
    ]
    out, calls = await _run(monkeypatch, results, arbitration=True, llm_out="CONTRADICT")
    assert calls == []  # a-b 相似度过低；a-c/b-c 同文本但跨 kind（persona vs event）
    assert "八十五分" in out and "每晚" in out and "简洁" in out


@pytest.mark.asyncio
async def test_arbitration_disabled_by_default(monkeypatch):
    """仲裁关（默认）→ 灰区对不调 LLM，两条都保留（原行为）。"""
    out, calls = await _run(
        monkeypatch, [_REVERSAL_OLD, _REVERSAL_NEW], arbitration=False, llm_out="CONTRADICT"
    )
    assert calls == []
    assert "快速" in out and "缓慢" in out


@pytest.mark.asyncio
async def test_llm_failure_keeps_both(monkeypatch):
    """chat_complete 抛异常 → 该对降级保留两条，不拖垮记忆注入。"""
    out, calls = await _run(
        monkeypatch, [_REVERSAL_OLD, _REVERSAL_NEW], arbitration=True,
        llm_raises=RuntimeError("AI 服务暂时不可用"),
    )
    assert len(calls) == 1
    assert "快速" in out and "缓慢" in out


@pytest.mark.asyncio
async def test_invalid_output_treated_as_supplement(monkeypatch):
    """LLM 输出不含合法标签 → 按 SUPPLEMENT 保留（仲裁说不上来就不替学生扔记忆）。"""
    out, calls = await _run(
        monkeypatch, [_REVERSAL_OLD, _REVERSAL_NEW], arbitration=True,
        llm_out="我认为这两条都是有效的历史记录，无法判断。",
    )
    assert len(calls) == 1
    assert "快速" in out and "缓慢" in out
