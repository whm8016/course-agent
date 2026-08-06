"""Retriever 抽象基类。

定义统一检索接口，各后端实现此接口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.rag.types import RetrievalResult


class Retriever(ABC):
    """检索器抽象基类。

    所有 RAG 后端都实现此接口，
    允许调用方通过 registry 获取具体实现，无需硬编码后端选择。
    """

    @abstractmethod
    async def retrieve(
        self,
        course_id: str,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> list[RetrievalResult]:
        """检索相关文档片段。

        Args:
            course_id: 课程 ID（知识库隔离键）
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 后端特定参数（如 mode, rerank 等）

        Returns:
            检索结果列表，按相关性降序排列
        """
        ...

    @abstractmethod
    async def retrieve_context(
        self,
        course_id: str,
        query: str,
        top_k: int = 5,
        max_chars: int = 4000,
        **kwargs,
    ) -> str:
        """检索并拼接为上下文字符串。

        Args:
            course_id: 课程 ID
            query: 查询文本
            top_k: 返回结果数量
            max_chars: 上下文最大字符数
            **kwargs: 后端特定参数

        Returns:
            拼接后的上下文字符串，供 LLM prompt 使用
        """
        ...

    async def is_available(self) -> tuple[bool, str]:
        """检查后端是否可用。

        Returns:
            (is_available, error_message) 元组
        """
        return True, ""
