"""Indexer 抽象基类。

定义统一索引接口，各后端实现此接口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.rag.types import IndexResult


class Indexer(ABC):
    """索引器抽象基类。

    所有 RAG 后端都实现此接口，
    允许调用方通过 registry 获取具体实现，无需硬编码后端选择。
    """

    @abstractmethod
    async def index(
        self,
        course_id: str,
        file_paths: list[str],
        **kwargs,
    ) -> IndexResult:
        """索引文档文件。

        Args:
            course_id: 课程 ID（知识库隔离键）
            file_paths: 待索引文件路径列表
            **kwargs: 后端特定参数（如 resume_from_chunk, kb_id 等）

        Returns:
            索引结果摘要
        """
        ...

    @abstractmethod
    async def delete(self, course_id: str) -> bool:
        """删除课程索引。

        Args:
            course_id: 课程 ID

        Returns:
            是否删除成功
        """
        ...

    async def is_available(self) -> tuple[bool, str]:
        """检查后端是否可用。

        Returns:
            (is_available, error_message) 元组
        """
        return True, ""
