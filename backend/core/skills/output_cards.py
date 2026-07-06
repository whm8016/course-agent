"""Output Cards：对话完成后生成额外的教学补充框。

（原 output_skills.py 重命名，避免与 式 skill 知识包混淆——前者是对话后
的补充内容生成器，后者是 agent 按需读取的程序性知识 playbook，两者完全不同）

功能：
- 管理员/教师可创建补充卡片（如「易错提醒」「解题思路」「竞赛延伸」）
- 每次对话回复后，对启用的卡片各生成一段补充内容
- 通过 SSE 额外发送 type="skill_output" 事件（事件名保留前端兼容）
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from settings import get_settings
OUTPUT_CARDS_PATH = get_settings().paths.output_cards_path
TEXT_MODEL = get_settings().llm.text_model
from core.llm.llm import client as async_openai_client

logger = logging.getLogger(__name__)

_STORE_PATH = Path(OUTPUT_CARDS_PATH)
_MAX_CARDS = 5

_FOLLOW_UP_SYSTEM = """你正在为课程学习助手生成一段额外的教学补充内容。
这段内容会在正常回答之后单独展示给学生。

规则：
- 严格按照补充卡片的指令风格输出
- 从不同教学角度补充，不要重复正常回答
- 输出独立的中文 Markdown 段落
- 简洁有力，150-300 字为宜"""


@dataclass
class OutputCard:
    id: str
    title: str
    description: str
    instruction: str
    enabled: bool = True
    created_at: str = ""
    course_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OutputCard":
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            instruction=str(data.get("instruction", "")),
            enabled=bool(data.get("enabled", True)),
            created_at=str(data.get("created_at", "")),
            course_id=str(data.get("course_id", "")),
        )


class OutputCardStore:
    """管理自定义补充卡片的存储。"""

    def __init__(self):
        self._ensure_file()

    def _ensure_file(self):
        if not _STORE_PATH.exists():
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _STORE_PATH.write_text(json.dumps({"cards": []}, ensure_ascii=False), encoding="utf-8")

    def list_all(self) -> list[OutputCard]:
        try:
            data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
            # 兼容旧 key "skills"（从 output_skills.json 迁移来的数据）
            cards = data.get("cards", data.get("skills", []))
            return [OutputCard.from_dict(c) for c in cards if c.get("id")]
        except Exception:
            return []

    def list_enabled(self, course_id: str = "") -> list[OutputCard]:
        cards = self.list_all()
        return [
            c for c in cards
            if c.enabled and (not c.course_id or c.course_id == course_id)
        ]

    def create(
        self, title: str, description: str, instruction: str, course_id: str = ""
    ) -> OutputCard:
        cards = self.list_all()
        if len(cards) >= _MAX_CARDS:
            raise ValueError(f"最多创建 {_MAX_CARDS} 个补充卡片")
        card = OutputCard(
            id=f"card-{uuid4().hex[:8]}",
            title=title.strip()[:20],
            description=description.strip()[:100],
            instruction=instruction.strip()[:1000],
            enabled=True,
            created_at=datetime.now().isoformat(),
            course_id=course_id,
        )
        cards.append(card)
        self._save(cards)
        return card

    def toggle(self, card_id: str, enabled: bool) -> OutputCard | None:
        cards = self.list_all()
        for c in cards:
            if c.id == card_id:
                c.enabled = enabled
                self._save(cards)
                return c
        return None

    def delete(self, card_id: str) -> bool:
        cards = self.list_all()
        filtered = [c for c in cards if c.id != card_id]
        if len(filtered) == len(cards):
            return False
        self._save(filtered)
        return True

    def _save(self, cards: list[OutputCard]):
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {"cards": [c.to_dict() for c in cards]}
        _STORE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_store = OutputCardStore()


def get_output_card_store() -> OutputCardStore:
    return _store


async def generate_output_cards(
    *,
    course_id: str,
    user_message: str,
    assistant_answer: str,
) -> list[dict[str, str]]:
    """对话结束后为每个启用的卡片生成补充内容。

    返回 [{"title": "...", "content": "..."}]
    """
    cards = _store.list_enabled(course_id)
    if not cards:
        return []

    outputs: list[dict[str, str]] = []
    for card in cards:
        content = await _generate_one(card, user_message, assistant_answer)
        if content:
            outputs.append({"title": card.title, "content": content})

    return outputs


async def _generate_one(
    card: OutputCard, user_message: str, assistant_answer: str
) -> str:
    """为单个卡片生成补充输出。"""
    user_prompt = (
        f"# 补充卡片: {card.title}\n"
        f"指令: {card.instruction}\n\n"
        f"学生问题: {user_message[:800]}\n\n"
        f"已有回答（摘要）: {assistant_answer[:800]}\n\n"
        f"请根据卡片指令，生成一段补充内容："
    )
    try:
        resp = await async_openai_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": _FOLLOW_UP_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=500,
            stream=False,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Output card generation failed card=%s: %s", card.id, e)
        return ""
