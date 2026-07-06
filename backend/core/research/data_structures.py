"""

简化版，仅保留 research pipeline 需要的最小调度状态：

- ``TopicStatus``     子主题状态枚举（pending / researching / completed / failed）
- ``ToolTrace``       一次工具调用的精简记录（tool_type / query / summary / source）
- ``TopicBlock``      队列最小调度单元（block_id / sub_topic / overview / status / sources）
- ``DynamicTopicQueue`` 主题队列：add_block / get_pending / all_done / is_full / find_similar

与 的差异：
- 不做 JSON 落盘 / state_file 持久化（内存态即可，per-turn）
- 不做 raw_answer 截断（research loop 自带上下文预算）
- find_similar 用最简单的归一化 + 子串匹配，不引 difflib（足够挡住重复主题）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TopicStatus(str, Enum):
    """子主题状态。"""

    PENDING = "pending"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    FAILED = "failed"


DEFAULT_QUEUE_MAX_LENGTH = 8


@dataclass
class ToolTrace:
    """一次工具调用的精简记录（citation 侧来源）。"""

    tool_type: str           # rag / web_search / ...
    query: str               # 本次调用发出的查询
    summary: str             # 该工具结果给到报告层的摘要 / 原文片段
    source: str = ""         # 来源标识（url / 文件名 / kb+query），用于去重
    tool_id: str = ""        # 来源编号，留给 citation_manager 填 [n]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_type": self.tool_type,
            "query": self.query,
            "summary": self.summary,
            "source": self.source,
            "tool_id": self.tool_id,
            "timestamp": self.timestamp,
        }


@dataclass
class TopicBlock:
    """队列最小调度单元。"""

    block_id: str
    sub_topic: str
    overview: str = ""
    status: TopicStatus = TopicStatus.PENDING
    sources: list[ToolTrace] = field(default_factory=list)
    knowledge: str = ""        # 本块研究 loop 的 FINISH 摘要，供报告层引用
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_source(self, trace: ToolTrace) -> None:
        self.sources.append(trace)


def _normalize(text: str) -> str:
    """归一化主题标题用于去重比较：去空白、转小写。"""
    return " ".join((text or "").split()).lower()


class DynamicTopicQueue:
    """动态主题队列：research 阶段的内存调度中心。

    DynamicTopicQueue，去掉持久化与复杂相似度，保留：
    add_block / get_pending / all_done / is_full / find_similar / 标记状态。
    """

    def __init__(self, research_id: str = "", max_length: int = DEFAULT_QUEUE_MAX_LENGTH) -> None:
        self.research_id = research_id
        self.max_length = max_length if isinstance(max_length, int) and max_length > 0 else None
        self.blocks: list[TopicBlock] = []
        self._counter = 0

    def add_block(self, sub_topic: str, overview: str = "") -> TopicBlock:
        """追加一个子主题块到队尾；队列已满抛 RuntimeError。"""
        if self.is_full():
            raise RuntimeError(f"主题队列已满（上限 {self.max_length}），无法新增子主题。")
        self._counter += 1
        block = TopicBlock(
            block_id=f"block_{self._counter}",
            sub_topic=sub_topic.strip(),
            overview=(overview or "").strip(),
        )
        self.blocks.append(block)
        return block

    def is_full(self) -> bool:
        return self.max_length is not None and len(self.blocks) >= self.max_length

    def get_block(self, block_id: str) -> TopicBlock | None:
        for b in self.blocks:
            if b.block_id == block_id:
                return b
        return None

    def get_pending(self) -> list[TopicBlock]:
        return [b for b in self.blocks if b.status == TopicStatus.PENDING]

    def all_done(self) -> bool:
        """所有块都已终结（completed / failed）。空队列视为未完成。"""
        if not self.blocks:
            return False
        terminal = {TopicStatus.COMPLETED, TopicStatus.FAILED}
        return all(b.status in terminal for b in self.blocks)

    def mark_researching(self, block_id: str) -> bool:
        block = self.get_block(block_id)
        if block is None:
            return False
        block.status = TopicStatus.RESEARCHING
        return True

    def mark_completed(self, block_id: str) -> bool:
        block = self.get_block(block_id)
        if block is None:
            return False
        block.status = TopicStatus.COMPLETED
        return True

    def mark_failed(self, block_id: str) -> bool:
        block = self.get_block(block_id)
        if block is None:
            return False
        block.status = TopicStatus.FAILED
        return True

    def find_similar(self, sub_topic: str) -> TopicBlock | None:
        """返回归一化后与 sub_topic 相同或互为子串的已有块；无则 None。

        简化去重：两种归一化各试一次——
        (1) 压缩空白后小写（保留词间单空格）；
        (2) 去除所有空白后小写（覆盖中文插入 / 删除空格的差异）。
        任一归一化下相等或互为子串即判定相似。
        """
        target = _normalize(sub_topic)
        target_compact = target.replace(" ", "")
        if not target and not target_compact:
            return None
        for b in self.blocks:
            cand = _normalize(b.sub_topic)
            cand_compact = cand.replace(" ", "")
            if not cand and not cand_compact:
                continue
            if (
                (target and (cand == target or cand in target or target in cand))
                or (
                    target_compact
                    and cand_compact
                    and (
                        cand_compact == target_compact
                        or cand_compact in target_compact
                        or target_compact in cand_compact
                    )
                )
            ):
                return b
        return None

    def list_titles(self) -> list[str]:
        return [b.sub_topic for b in self.blocks]

    def statistics(self) -> dict[str, Any]:
        pending = sum(1 for b in self.blocks if b.status == TopicStatus.PENDING)
        researching = sum(1 for b in self.blocks if b.status == TopicStatus.RESEARCHING)
        completed = sum(1 for b in self.blocks if b.status == TopicStatus.COMPLETED)
        failed = sum(1 for b in self.blocks if b.status == TopicStatus.FAILED)
        return {
            "total": len(self.blocks),
            "pending": pending,
            "researching": researching,
            "completed": completed,
            "failed": failed,
            "sources": sum(len(b.sources) for b in self.blocks),
        }


__all__ = [
    "DEFAULT_QUEUE_MAX_LENGTH",
    "DynamicTopicQueue",
    "TopicBlock",
    "TopicStatus",
    "ToolTrace",
]
