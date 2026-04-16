"""Intent classification, safety guardrail and hallucination detection.

All three functions return lightweight dict results that are cheap to
serialize into SSE metadata.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Any

from openai import AsyncOpenAI

from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, TEXT_MODEL

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

# ── Intent classification ────────────────────────────────────────────────

INTENT_CHITCHAT = "chitchat"
INTENT_KNOWLEDGE = "knowledge"
INTENT_QUIZ = "quiz"
INTENT_SUMMARIZE = "summarize"

_GREETING_PATTERNS = re.compile(
    r"^(你好|嗨|hi|hello|hey|哈喽|喂|在吗|早上好|晚上好|下午好|嘿|早安|晚安"
    r"|谢谢|感谢|多谢|辛苦了|拜拜|再见|bye|好的|ok|收到|明白了|了解"
    r"|哈哈|嗯嗯|666|牛|厉害|可以|棒|对的|是的|好吧|行|okok"
    r"|你是谁|你叫什么|你能做什么|你会什么)[\s!！?？。.~～…]*$",
    re.IGNORECASE,
)

_QUIZ_KEYWORDS = {"出题", "测验", "考考我", "来几道题", "来道题", "做题", "quiz"}
_SUMMARY_KEYWORDS = {"总结", "归纳", "回顾", "小结", "知识清单", "summarize"}

# 明显在问知识点 / 求讲解：跳过意图 LLM，避免多一次网络往返（对「什么是…」类问题尤其有效）
_KNOWLEDGE_QUESTION_PATTERN = re.compile(
    r"(什么是|是啥|指什么|含义|定义|概念|原理|区别|对比|为什么|为何|怎么|如何|怎样|请问|解释一下|说明一下|"
    r"推导|证明|计算|复杂度|时间复杂度|空间复杂度|算法|数据结构|代码实现|伪代码)",
    re.I,
)


@dataclass
class IntentResult:
    intent: str  # chitchat / knowledge / quiz / summarize
    confidence: float  # 0-1
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def classify_intent(
    message: str,
    history: list[dict] | None = None,
) -> IntentResult:
    """Fast rule + optional LLM fallback intent classification."""
    text = message.strip()
    logger.info("classify_intent input=「%s」 history_len=%d", text[:80], len(history or []))

    if _GREETING_PATTERNS.match(text):
        logger.info("classify_intent result=chitchat reason=greeting_pattern")
        return IntentResult(INTENT_CHITCHAT, 1.0, "greeting_pattern")

    lower = text.lower()
    if any(kw in lower for kw in _QUIZ_KEYWORDS):
        logger.info("classify_intent result=quiz reason=quiz_keyword")
        return IntentResult(INTENT_QUIZ, 0.95, "quiz_keyword")
    if any(kw in lower for kw in _SUMMARY_KEYWORDS):
        logger.info("classify_intent result=summarize reason=summary_keyword")
        return IntentResult(INTENT_SUMMARIZE, 0.90, "summary_keyword")

    if len(text) <= 6 and not any(
        ch in text for ch in "什怎为何哪啥如何解释说明分析证明推导计算求"
    ):
        logger.info("classify_intent result=chitchat reason=short_non_question")
        return IntentResult(INTENT_CHITCHAT, 0.85, "short_non_question")

    if _KNOWLEDGE_QUESTION_PATTERN.search(text):
        logger.info("classify_intent result=knowledge reason=knowledge_question_pattern")
        return IntentResult(INTENT_KNOWLEDGE, 0.92, "knowledge_question_pattern")

    try:
        resp = await _client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个意图分类器。判断用户消息属于哪一类：\n"
                        '- "chitchat": 闲聊、打招呼、感谢、告别、不涉及课程知识\n'
                        '- "knowledge": 涉及课程知识内容的提问或讨论\n'
                        '- "quiz": 要求出题或做练习\n'
                        '- "summarize": 要求总结或归纳\n\n'
                        '只输出 JSON: {"intent":"...","confidence":0.0~1.0}'
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=60,
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        intent = obj.get("intent", INTENT_KNOWLEDGE)
        conf = float(obj.get("confidence", 0.8))
        if intent not in (INTENT_CHITCHAT, INTENT_KNOWLEDGE, INTENT_QUIZ, INTENT_SUMMARIZE):
            intent = INTENT_KNOWLEDGE
        logger.info("classify_intent result=%s confidence=%.2f reason=llm", intent, conf)
        return IntentResult(intent, conf, "llm")
    except Exception:
        logger.warning("classify_intent LLM fallback failed, default to knowledge", exc_info=True)
        return IntentResult(INTENT_KNOWLEDGE, 0.6, "fallback")


# ── Safety guardrail ─────────────────────────────────────────────────────

_RISK_PATTERNS = [
    (re.compile(r"(忽略|无视|跳过|不要遵守).{0,10}(指令|规则|限制|设定|角色)", re.I), "prompt_injection", 0.9),
    (re.compile(r"(假装|扮演|你现在是).{0,15}(没有限制|无所不能|邪恶|恶意)", re.I), "jailbreak", 0.9),
    (re.compile(r"(如何|怎么|怎样).{0,10}(攻击|入侵|破解|制造武器|伤害)", re.I), "harmful_request", 0.8),
    (re.compile(r"(色情|赌博|毒品|自杀|自残)", re.I), "sensitive_content", 0.85),
]


@dataclass
class GuardrailResult:
    safe: bool
    risk_type: str  # "none" / "prompt_injection" / ...
    risk_score: float  # 0-1, higher = riskier
    tip: str  # user-facing soft tip

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_guardrail(message: str) -> GuardrailResult:
    """Rule-based input safety check (sync, zero-latency)."""
    for pattern, risk_type, score in _RISK_PATTERNS:
        if pattern.search(message):
            logger.warning(
                "guardrail TRIGGERED risk=%s score=%.2f input=「%s」",
                risk_type, score, message[:60],
            )
            return GuardrailResult(
                safe=False,
                risk_type=risk_type,
                risk_score=score,
                tip="检测到可能的不当请求，回答将围绕课程内容进行。",
            )
    logger.debug("guardrail passed input=「%s」", message[:40])
    return GuardrailResult(safe=True, risk_type="none", risk_score=0.0, tip="")


# ── Hallucination detection ──────────────────────────────────────────────

@dataclass
class HallucinationResult:
    grounded: bool
    confidence: float  # 0-1, how grounded the answer is
    tip: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def evaluate_hallucination(
    answer: str,
    contexts: list[Any] | None = None,
) -> HallucinationResult:
    """Check if the answer is grounded in the retrieved contexts."""
    if not contexts:
        return HallucinationResult(
            grounded=False,
            confidence=0.3,
            tip="本次回答未检索到参考资料，内容基于模型自身知识，建议核实关键信息。",
        )

    ctx_text = ""
    for ctx in contexts[:4]:
        if isinstance(ctx, str):
            ctx_text += ctx[:500] + "\n"
        elif isinstance(ctx, dict):
            for key in ("content", "text", "chunk"):
                v = ctx.get(key)
                if isinstance(v, str):
                    ctx_text += v[:500] + "\n"
                    break

    if not ctx_text.strip():
        return HallucinationResult(
            grounded=False,
            confidence=0.3,
            tip="参考资料为空，回答可能包含模型自身推测。",
        )

    try:
        resp = await _client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个事实核查助手。判断【回答】是否有充分的【参考资料】支撑。\n"
                        "输出 JSON: {\"grounded\": true/false, \"confidence\": 0.0~1.0, "
                        "\"reason\": \"简短说明\"}\n"
                        "confidence 表示回答被证据支撑的程度。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"【参考资料】\n{ctx_text[:2000]}\n\n【回答】\n{answer[:1500]}",
                },
            ],
            temperature=0,
            max_tokens=120,
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        grounded = bool(obj.get("grounded", True))
        conf = float(obj.get("confidence", 0.7))
        tip = ""
        if not grounded:
            tip = "部分内容可能缺少资料支撑，建议对照课件核实。"
        elif conf < 0.6:
            tip = "回答的证据支撑度一般，仅供参考。"
        return HallucinationResult(grounded=grounded, confidence=conf, tip=tip)
    except Exception:
        logger.debug("hallucination check failed", exc_info=True)
        return HallucinationResult(grounded=True, confidence=0.5, tip="")
