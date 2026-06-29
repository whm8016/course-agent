"""LightRAG Retriever 实现。

实现 Retriever ABC，提供基于 LightRAG 知识图谱的检索能力。
"""
from __future__ import annotations

import logging
from typing import Any

from core.rag.types import RetrievalResult, ChunkMeta
from core.rag.retriever.base import Retriever
from core.rag.lightrag import (
    _get_instance,
    is_lightrag_available,
)
from core.rag.rag_config import get_safe_top_k

logger = logging.getLogger(__name__)


class LightRAGREtriever(Retriever):
    """LightRAG 检索器实现。

    使用 LightRAG 知识图谱进行语义检索，支持实体/关系推理。
    """

    async def retrieve(
        self,
        course_id: str,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> list[RetrievalResult]:
        """检索相关文档片段。

        Args:
            course_id: 课程 ID
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 后端特定参数（如 mode, rerank 等）

        Returns:
            检索结果列表，按相关性降序排列
        """
        # 检查可用性
        ok, reason = is_lightrag_available()
        if not ok:
            logger.warning("LightRAGREtriever skipped: %s", reason)
            return []

        # 获取安全 top_k
        safe_top_k = get_safe_top_k()
        actual_top_k = min(top_k, safe_top_k)

        try:
            rag = await _get_instance(course_id)

            # 构建查询参数
            query_mode = kwargs.get("mode") or "mix"
            param = self._build_query_param(query_mode)

            # 执行查询
            import time
            t0 = time.perf_counter()
            result = await rag.aquery(query, param=param)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            # 提取上下文
            contexts = self._extract_contexts(result)

            # 转换为 RetrievalResult
            results = []
            for i, ctx in enumerate(contexts[:actual_top_k]):
                text = self._extract_context_text(ctx)
                if text:
                    chunk_meta = ChunkMeta(
                        source_path="",
                        start_char=0,
                        end_char=len(text),
                        chunk_id=f"{course_id}_{i}",
                    )
                    results.append(RetrievalResult(
                        content=text,
                        score=float(actual_top_k - i),  # 模拟分数
                        source_chunk=chunk_meta,
                        metadata={"mode": query_mode},
                    ))

            logger.info(
                "LightRAGREtriever.retrieve course=%s query=%.60s top_k=%d results=%d elapsed_ms=%d",
                course_id, query[:60], actual_top_k, len(results), elapsed_ms,
            )
            return results

        except Exception as exc:
            logger.error("LightRAGREtriever.retrieve failed: %s", exc, exc_info=True)
            return []

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
        # 检查可用性
        ok, reason = is_lightrag_available()
        if not ok:
            logger.warning("LightRAGREtriever skipped: %s", reason)
            return ""

        try:
            rag = await _get_instance(course_id)

            # 构建查询参数
            query_mode = kwargs.get("mode") or "mix"
            param = self._build_query_param(query_mode)

            # 执行查询
            import time
            t0 = time.perf_counter()
            result = await rag.aquery(query, param=param)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            # 提取上下文文本
            contexts = self._extract_contexts(result)
            context_text = self._format_contexts_for_prompt(
                contexts, limit=top_k, max_chars=max_chars
            )

            logger.info(
                "LightRAGREtriever.retrieve_context course=%s query=%.60s top_k=%d chars=%d elapsed_ms=%d",
                course_id, query[:60], top_k, len(context_text), elapsed_ms,
            )
            return context_text

        except Exception as exc:
            logger.error("LightRAGREtriever.retrieve_context failed: %s", exc, exc_info=True)
            return ""

    def _build_query_param(self, mode: str) -> dict[str, Any]:
        """构建 LightRAG 查询参数。"""
        if mode == "entity":
            # 实体优先模式
            param = {
                "need_response": False,
                "only_need_context": True,
            }
        elif mode == "hybrid":
            # 混合模式（默认）
            param = {
                "need_response": True,
                "only_need_context": False,
            }
        elif mode == "keyword":
            # 关键词模式
            param = {
                "need_response": False,
                "only_need_context": True,
                "include_keywords": True,
            }
        else:
            # 默认 mix 模式
            param = {
                "need_response": True,
                "only_need_context": False,
            }
        return param

    @staticmethod
    def _extract_contexts(result: Any) -> list[Any]:
        """从 LightRAG 查询结果中提取上下文列表。"""
        if isinstance(result, list):
            return result
        if isinstance(result, str):
            text = result.strip()
            return [text] if text else []
        if isinstance(result, dict):
            for key in ("contexts", "context", "chunks", "references", "data"):
                value = result.get(key)
                if isinstance(value, list):
                    return value
                if isinstance(value, dict):
                    return [value]
                if isinstance(value, str) and value.strip():
                    return [value.strip()]
        return []

    @staticmethod
    def _extract_context_text(ctx: Any) -> str:
        """从上下文对象中提取文本。"""
        if isinstance(ctx, str):
            return ctx.strip()
        if isinstance(ctx, dict):
            for key in ("content", "text", "chunk", "passage"):
                value = ctx.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return str(ctx).strip()

    @staticmethod
    def _format_contexts_for_prompt(contexts: list[Any], limit: int = 5, max_chars: int = 4000) -> str:
        """格式化上下文为 LLM prompt 格式。"""
        rows: list[str] = []
        for idx, ctx in enumerate(contexts[:limit]):
            text = LightRAGREtriever._extract_context_text(ctx)
            if not text:
                continue
            if len(text) > max_chars:
                text = f"{text[:max_chars]}...(truncated)"
            rows.append(f"[证据{idx + 1}]\n{text}")
        return "\n\n---\n\n".join(rows)


__all__ = ["LightRAGREtriever"]