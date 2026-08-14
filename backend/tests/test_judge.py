"""阶段B：LLM 判定器单测（mock LLM，不起真实调用）。"""
import json

import pytest

from core.memory.judge import JudgeResult, _validate, judge_segment

_DIALOG = [("欧姆定律我没看懂", "欧姆定律是 U=IR，电压等于电流乘电阻")]
_TOPICS = [("ohm", "欧姆定律", "电压、电流与电阻的关系"), ("kcl", "基尔霍夫电流定律", "节点电流守恒")]
_SEGMENT_TEXT = _DIALOG[0][0] + "\n" + _DIALOG[0][1]


# ── _validate：三条硬约束 ────────────────────────────────────────────────────


def test_out_of_closed_set_topic_dropped():
    raw = {
        "judgements": [
            {"topic_id": "nonexistent", "level": "poor", "confidence": 0.9,
             "evidence_quote": "没看懂"},
            {"topic_id": "ohm", "level": "poor", "confidence": 0.9,
             "evidence_quote": "欧姆定律我没看懂"},
        ]
    }
    r = _validate(raw, [t[0] for t in _TOPICS], _SEGMENT_TEXT)
    assert len(r.judgements) == 1 and r.judgements[0].topic_id == "ohm"
    assert r.dropped_invalid_topic == 1


def test_bad_evidence_quote_dropped_and_counted():
    raw = {
        "judgements": [
            {"topic_id": "ohm", "level": "poor", "confidence": 0.9,
             "evidence_quote": "这句原文里不存在"},
        ]
    }
    r = _validate(raw, [t[0] for t in _TOPICS], _SEGMENT_TEXT)
    assert r.judgements == []
    assert r.dropped_bad_quote == 1


def test_unknown_abstains_no_judgement():
    raw = {
        "judgements": [
            {"topic_id": "kcl", "level": "unknown", "confidence": 0.3,
             "evidence_quote": ""},
        ]
    }
    r = _validate(raw, [t[0] for t in _TOPICS], _SEGMENT_TEXT)
    assert r.judgements == []
    assert r.abstained == ["kcl"]


def test_invalid_level_dropped():
    raw = {"judgements": [{"topic_id": "ohm", "level": "excellent", "confidence": 0.9,
                           "evidence_quote": "没看懂"}]}
    r = _validate(raw, [t[0] for t in _TOPICS], _SEGMENT_TEXT)
    assert r.judgements == [] and r.dropped_invalid_topic == 1


def test_confidence_clamped():
    raw = {"judgements": [{"topic_id": "ohm", "level": "good", "confidence": 5.0,
                           "evidence_quote": "欧姆定律"}]}
    r = _validate(raw, [t[0] for t in _TOPICS], _SEGMENT_TEXT)
    assert r.judgements[0].confidence == 1.0


def test_error_patterns_closed_set_and_quote():
    raw = {
        "judgements": [],
        "error_patterns": [
            {"label": "电压电流概念混淆", "related_topic_ids": ["ohm"], "evidence_quote": "没看懂"},
            {"label": "乱编的", "related_topic_ids": ["nope"], "evidence_quote": "x"},
        ],
    }
    r = _validate(raw, [t[0] for t in _TOPICS], _SEGMENT_TEXT)
    assert len(r.error_patterns) == 1
    assert r.error_patterns[0]["related_topic_ids"] == ["ohm"]


# ── judge_segment：LLM 交互（mock） ─────────────────────────────────────────


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    async def create(self, **kw):
        class _Msg:
            content = self._content
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        return _Resp()


class _FakeClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()


@pytest.fixture
def patch_llm(monkeypatch):
    def _patch(content):
        import core.llm.llm as llm_mod
        monkeypatch.setattr(llm_mod, "client", _FakeClient(content))
        return llm_mod
    return _patch


async def test_judge_segment_one_call_per_segment(patch_llm):
    patch_llm(json.dumps({
        "judgements": [
            {"topic_id": "ohm", "level": "poor", "confidence": 0.9,
             "evidence_quote": "欧姆定律我没看懂"},
        ],
        "error_patterns": [],
    }))
    r = await judge_segment(_DIALOG, _TOPICS)
    assert isinstance(r, JudgeResult)
    assert r.judgements[0].level == "poor"


async def test_judge_segment_no_topics_no_llm(patch_llm):
    """无主题段不调 LLM（空列表直接 None，mock 里放了会炸的内容也证明没被调）。"""
    patch_llm("SHOULD_NOT_BE_CALLED")
    assert await judge_segment(_DIALOG, []) is None


async def test_judge_segment_llm_failure_failsafe(monkeypatch):
    class _Boom:
        async def create(self, **kw):
            raise RuntimeError("model down")
    import core.llm.llm as llm_mod
    monkeypatch.setattr(llm_mod, "client", type("C", (), {"chat": type("S", (), {"completions": _Boom()})})())
    assert await judge_segment(_DIALOG, _TOPICS) is None


async def test_judge_segment_bad_json_failsafe(patch_llm):
    patch_llm("not json at all {{{")
    assert await judge_segment(_DIALOG, _TOPICS) is None
