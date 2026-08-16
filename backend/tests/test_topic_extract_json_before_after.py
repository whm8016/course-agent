"""主题图抽取：旧 bug vs 现在 —— 对照测试（零真实 LLM）。

模型实际返回两截：
  reasoning_content = 思考过程（散文，不是 JSON）
  content           = 最终回答（我们要的 JSON）

额度 max_tokens 是「思考 + 正文」共用的。旧代码 4096，思考写满 → content 空
→ 日志「LLM 输出非 JSON」→ 整块抽取失败。

对照三种写法（本文件复刻旧路径）和现在的 ``_llm_json``：

  版本              思考        解析哪一截              额度    思考占满时
  旧代码            开（默认）  只 content              4096    失败（当时线上）
  中间误改-读思考   开          content 空则读思考      4096    会拿到思考散文
  现在              关          只 content              8000    不发生（不思考）
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.llm.json_extract import extract_json_from_llm
from core.memory.course_topic_builder import _llm_json

# 模拟 API 两截。思考是散文；最终回答才是 JSON。
_THINKING = "先看讲义……这段在讲戴维南等效，作为先修……"
_FINAL_JSON = '{"topics": [{"label": "戴维南等效", "definition": "等效电源定理"}], "edges": []}'


def _api_resp(*, content: str, reasoning: str, finish: str):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, reasoning_content=reasoning),
        finish_reason=finish,
    )])


def _llm_that_thinks(max_tokens_budget: int):
    """思考先写。额度不够 → content 空 + finish=length（旧 4096 的线上形态）。"""

    async def create(**kwargs):
        extra = kwargs.get("extra_body") or {}
        thinking = extra.get("thinking") or {}
        if thinking.get("type") == "disabled":
            # 关思考：直接吐 JSON（中间那版误改）
            return _api_resp(content=_FINAL_JSON, reasoning="", finish="stop")
        budget = kwargs.get("max_tokens") or 0
        if budget < max_tokens_budget:
            return _api_resp(content="", reasoning=_THINKING, finish="length")
        return _api_resp(content=_FINAL_JSON, reasoning=_THINKING, finish="stop")

    llm = SimpleNamespace()
    llm.chat = SimpleNamespace()
    llm.chat.completions = SimpleNamespace()
    llm.chat.completions.create = AsyncMock(side_effect=create)
    return llm


async def _old_llm_json(llm, model: str, prompt: str):
    """旧生产代码：额度 4096，只读 content，不关思考。"""
    resp = await llm.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=4096, stream=False,
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = extract_json_from_llm(raw)
    return data or None


async def _wrong_disable_thinking(llm, model: str, prompt: str):
    """中间误改：请求里关掉思考。能出 JSON，但用户要的是带思考。"""
    resp = await llm.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=4096, stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = extract_json_from_llm(raw)
    return data or None


async def _wrong_parse_reasoning(llm, model: str, prompt: str):
    """中间误改：content 空就去读 reasoning_content。会把思考散文当结果。"""
    resp = await llm.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=4096, stream=False,
    )
    msg = resp.choices[0].message
    raw = (msg.content or "").strip() or (msg.reasoning_content or "").strip()
    data = extract_json_from_llm(raw)
    return data or None


# 思考大约要这么多 token 才会把 4096 吃光；8000 才能留下最终 JSON。
_THINKING_COST = 5000


@pytest.mark.asyncio
async def test_simulated_api_two_fields_4096_vs_8000():
    """同一模型、同一思考：额度不够时只有思考；够了两截都在，生产只用 content。"""
    llm = _llm_that_thinks(_THINKING_COST)

    tight = await llm.chat.completions.create(max_tokens=4096)
    assert tight.choices[0].finish_reason == "length"
    assert tight.choices[0].message.content == ""
    assert tight.choices[0].message.reasoning_content == _THINKING

    room = await llm.chat.completions.create(max_tokens=8000)
    assert room.choices[0].finish_reason == "stop"
    assert "戴维南等效" in (room.choices[0].message.content or "")
    assert room.choices[0].message.reasoning_content == _THINKING  # 思考仍在，解析不用


@pytest.mark.asyncio
async def test_old_code_fails_when_thinking_fills_4096():
    """旧：思考开 + 4096 → content 空 → 抽取失败（当时的「非 JSON」）。"""
    llm = _llm_that_thinks(_THINKING_COST)
    assert await _old_llm_json(llm, "deepseek-v4-flash", "抽知识点") is None
    kwargs = llm.chat.completions.create.await_args.kwargs
    assert kwargs["max_tokens"] == 4096
    assert kwargs.get("extra_body") is None  # 没关思考


@pytest.mark.asyncio
async def test_wrong_disable_thinking_gets_json_but_turns_thinking_off():
    """误改 A：关掉思考，JSON 有了，但请求带了 thinking=disabled。"""
    llm = _llm_that_thinks(_THINKING_COST)
    data = await _wrong_disable_thinking(llm, "deepseek-v4-flash", "抽知识点")
    assert data == {
        "topics": [{"label": "戴维南等效", "definition": "等效电源定理"}],
        "edges": [],
    }
    assert llm.chat.completions.create.await_args.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"},
    }


@pytest.mark.asyncio
async def test_wrong_parse_reasoning_swallows_thinking_prose():
    """误改 B：去读思考字段。思考是散文，解析仍失败——哪怕里面碰巧有 JSON 也会用错字段。"""
    llm = _llm_that_thinks(_THINKING_COST)
    # 思考散文不是 JSON → None（看起来像修好了额度，其实拿错了字段）
    assert await _wrong_parse_reasoning(llm, "deepseek-v4-flash", "抽知识点") is None

    # 若思考里碰巧是 JSON，旧误改会当成最终结果；现在不允许
    async def create_json_in_thinking(**_kwargs):
        return _api_resp(
            content="", reasoning='{"topics": [{"label": "从思考里来的", "definition": "x"}], "edges": []}',
            finish="length",
        )

    llm.chat.completions.create = AsyncMock(side_effect=create_json_in_thinking)
    stolen = await _wrong_parse_reasoning(llm, "m", "p")
    assert stolen["topics"][0]["label"] == "从思考里来的"
    assert await _llm_json(llm, "m", "p") is None  # 现在：思考里的 JSON 不用


@pytest.mark.asyncio
async def test_now_disables_thinking_parses_only_content():
    """现在：关思考，直接吐 JSON，只解析 content。"""
    llm = _llm_that_thinks(_THINKING_COST)
    llm.base_url = "https://api.deepseek.com"
    data = await _llm_json(llm, "deepseek-v4-flash", "抽知识点")
    assert data == {
        "topics": [{"label": "戴维南等效", "definition": "等效电源定理"}],
        "edges": [],
    }
    kwargs = llm.chat.completions.create.await_args.kwargs
    assert kwargs.get("extra_body") == {"thinking": {"type": "disabled"}}
