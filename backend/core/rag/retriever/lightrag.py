"""LightRAG Retriever 实现。

实现 Retriever ABC，提供基于 LightRAG 知识图谱的检索能力。
从 lightrag_engine.py 迁移的完整检索逻辑。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from settings import get_settings
LIGHTRAG_ENABLE_RERANK = get_settings().lightrag.enable_rerank
LIGHTRAG_QUERY_MODE = get_settings().lightrag.query_mode
LIGHTRAG_STREAM_CONTEXT_LIMIT = get_settings().lightrag.stream_context_limit
LIGHTRAG_STREAM_CONTEXT_MAX_CHARS = get_settings().lightrag.stream_context_max_chars
LIGHTRAG_MAX_HISTORY_MESSAGES = get_settings().lightrag.max_history_messages
LIGHTRAG_MAX_HISTORY_CHARS = get_settings().lightrag.max_history_chars
from core.observability import log_flow

from core.rag.types import RetrievalResult, ChunkMeta
from core.rag.retriever.base import Retriever
from core.rag.lightrag import (
    _get_instance,
    _release_instance,
    is_lightrag_available,
)

logger = logging.getLogger(__name__)

# ── 安全参数（模块级常量从 settings 绑定）──────────────────────────────────────

_SAFE_MAX_HISTORY_MESSAGES = LIGHTRAG_MAX_HISTORY_MESSAGES
_SAFE_MAX_HISTORY_CHARS = LIGHTRAG_MAX_HISTORY_CHARS
_STREAM_CONTEXT_LIMIT = max(1, LIGHTRAG_STREAM_CONTEXT_LIMIT)
_STREAM_CONTEXT_MAX_CHARS = max(200, LIGHTRAG_STREAM_CONTEXT_MAX_CHARS)


# ── 辅助函数──────────────────────────────────────────────────────


def _normalize_history(history: list[dict] | None) -> list[dict[str, str]]:
    """规范化对话历史。"""
    out: list[dict[str, str]] = []
    for msg in history or []:
        role = str(msg.get("role", "")).strip()
        if role not in ("user", "assistant"):
            continue
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _cap_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """截断对话历史到安全长度。"""
    if not messages:
        return []
    capped = messages[-_SAFE_MAX_HISTORY_MESSAGES:]
    total_chars = sum(len(m["content"]) for m in capped)
    while len(capped) > 1 and total_chars > _SAFE_MAX_HISTORY_CHARS:
        capped = capped[1:]
        total_chars = sum(len(m["content"]) for m in capped)
    return capped


def _build_query_param(
    mode: str,
    history: list[dict] | None,
    *,
    only_need_context: bool = False,
    top_k: int | None = None,
) -> Any:
    """构建 LightRAG QueryParam。

    top_k: 调用方显式指定的检索数量。None → 用配置默认（settings.lightrag
        .safe_top_k_value / chunk_top_k_value）；传值时仍受安全上限约束（防 API
        拒绝），但允许小于
        配置默认，使 retrieve_context / rag tool 的 top_k 真正控制 LightRAG
        检索与 rerank 的候选数量，而非仅做事后切片。
    """
    try:
        from lightrag import QueryParam
    except ImportError:
        raise RuntimeError("LightRAG 依赖不可用")

    lr = get_settings().lightrag
    if top_k is not None:
        req = max(1, int(top_k))
        safe_top_k = min(req, lr.safe_top_k_value())
        chunk_top_k = min(req, lr.chunk_top_k_value())
    else:
        safe_top_k = lr.safe_top_k_value()
        chunk_top_k = lr.chunk_top_k_value()
    max_tokens = lr.max_tokens_config()

    param = QueryParam(
        mode=mode,
        top_k=safe_top_k,
        chunk_top_k=chunk_top_k,
        max_total_tokens=max_tokens["total"],
        max_entity_tokens=max_tokens["entity"],
        max_relation_tokens=max_tokens["relation"],
        conversation_history=_cap_history(_normalize_history(history)),
        enable_rerank=LIGHTRAG_ENABLE_RERANK,
    )
    if only_need_context:
        if hasattr(param, "only_need_context"):
            setattr(param, "only_need_context", True)
        if hasattr(param, "return_context_only"):
            setattr(param, "return_context_only", True)
        if hasattr(param, "need_response"):
            setattr(param, "need_response", False)
    return param


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


def _strip_source_suffix(source: str) -> str:
    """剥离摄入时为绕过 filename 去重加的 `::chunk-<idx>` 后缀，还原真实文件名。

    见 ingestion.parse_files：每个 chunk 的 file_path 被加了 `::chunk-<node 索引>` 让其
    在 LightRAG 内部唯一。检索结果回带的 file_path 仍带此后缀，展示溯源（如「来源:
    xxx.pdf」）和 ChunkMeta.source_path 前必须剥掉，否则用户会看到 `xxx.pdf::chunk-5`。
    用 rfind 定位最后一个 `::chunk-`。不做 unknown_source 特判——剥成 unknown_source
    后由 _extract_file_path 统一过滤，避免 `unknown_source::chunk-N` 变体漏网。
    """
    marker = "::chunk-"
    idx = source.rfind(marker)
    if idx > 0:
        return source[:idx]
    return source


def _extract_file_path(ctx: Any) -> str:
    """从 LightRAG 检索结果中提取来源 file_path，过滤 unknown_source。

    LightRAG 在摄入端传 file_paths 后，query 结果的 chunk/reference 会回带 file_path。
    返回值已剥离摄入时加的 `::chunk-<idx>` 唯一后缀，还原成真实文件名。
    先剥后缀再判 unknown_source，避免 `unknown_source::chunk-5` 这类变体漏过过滤。
    """
    if isinstance(ctx, dict):
        for key in ("file_path", "source_path", "source", "file_name"):
            value = ctx.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = _strip_source_suffix(value.strip())
                if cleaned and cleaned != "unknown_source":
                    return cleaned
    return ""


def _format_contexts_for_prompt(
    contexts: list[Any],
    limit: int = _STREAM_CONTEXT_LIMIT,
    max_chars: int = _STREAM_CONTEXT_MAX_CHARS,
) -> str:
    """格式化上下文为 LLM prompt 格式，每条证据带来源文件标记。"""
    rows: list[str] = []
    for idx, ctx in enumerate(contexts[:limit]):
        text = _extract_context_text(ctx)
        if not text:
            continue
        if len(text) > max_chars:
            text = f"{text[:max_chars]}...(truncated)"
        # 仅显示文件名，避免暴露服务器绝对路径
        source = _extract_file_path(ctx)
        source_name = Path(source).name if source else ""
        header = f"[证据{idx + 1}]" if not source_name else f"[证据{idx + 1}｜来源: {source_name}]"
        rows.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(rows)


# ── LightRAGRetriever 类────────────────────────────────────────────────────


class LightRAGRetriever(Retriever):
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
        """检索相关文档片段。"""
        ok, reason = is_lightrag_available()
        if not ok:
            logger.warning("LightRAGRetriever skipped: %s", reason)
            return []

        safe_top_k = get_settings().lightrag.safe_top_k_value()
        actual_top_k = min(top_k, safe_top_k)

        try:
            rag = await _get_instance(course_id)
            query_mode = kwargs.get("mode") or "mix"
            param = _build_query_param(query_mode, None, only_need_context=True, top_k=top_k)

            t0 = time.perf_counter()
            result = await rag.aquery(query, param=param)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            contexts = _extract_contexts(result)
            results = []
            for i, ctx in enumerate(contexts[:actual_top_k]):
                text = _extract_context_text(ctx)
                if text:
                    source_path = _extract_file_path(ctx)
                    chunk_meta = ChunkMeta(
                        source_path=source_path,
                        start_char=0,
                        end_char=len(text),
                        chunk_id=f"{course_id}_{i}",
                    )
                    results.append(RetrievalResult(
                        content=text,
                        score=float(actual_top_k - i),
                        source_chunk=chunk_meta,
                        metadata={"mode": query_mode},
                    ))

            logger.info(
                "LightRAGRetriever.retrieve course=%s query=%.60s top_k=%d results=%d elapsed_ms=%d",
                course_id, query[:60], actual_top_k, len(results), elapsed_ms,
            )
            return results

        except Exception as exc:
            logger.error("LightRAGRetriever.retrieve failed: %s", exc, exc_info=True)
            return []
        finally:
            # H-10：释放引用计数，让实例可被 LRU 淘汰（_get_instance 已 +1）
            await _release_instance(course_id)

    async def retrieve_context(
        self,
        course_id: str,
        query: str,
        top_k: int = 3,
        max_chars: int = 2000,
        **kwargs,
    ) -> str:
        """检索并拼接为上下文字符串。"""
        ok, reason = is_lightrag_available()
        if not ok:
            logger.warning("LightRAGRetriever skipped: %s", reason)
            return ""

        try:
            rag = await _get_instance(course_id)
            # retrieve_context 是 agent tool call 的快速检索路径，使用 naive 模式：
            # naive = 纯向量搜索，only_need_context=True 时 0 次内部 LLM 调用。
            # 调用方可通过 kwargs["mode"] 覆盖（如需图谱推理传 "mix"）。
            query_mode = kwargs.get("mode") or "naive"
            param = _build_query_param(query_mode, None, only_need_context=True, top_k=top_k)

            t0 = time.perf_counter()
            result = await rag.aquery(query, param=param)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            contexts = _extract_contexts(result)
            context_text = _format_contexts_for_prompt(
                contexts, limit=top_k, max_chars=max_chars
            )

            logger.info(
                "LightRAGRetriever.retrieve_context course=%s mode=%s query=%.60s top_k=%d chars=%d elapsed_ms=%d",
                course_id, query_mode, query[:60], top_k, len(context_text), elapsed_ms,
            )
            return context_text

        except Exception as exc:
            logger.error("LightRAGRetriever.retrieve_context failed: %s", exc, exc_info=True)
            return ""
        finally:
            await _release_instance(course_id)

    async def query(
        self,
        course_id: str,
        message: str,
        history: list[dict] | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """执行完整查询（检索 + 生成回答）。

        Args:
            course_id: 课程 ID
            message: 用户问题
            history: 对话历史
            mode: 查询模式 (mix/entity/keyword等)

        Returns:
            dict: {"answer": str, "contexts": list, "mode": str}
        """
        ok, reason = is_lightrag_available()
        if not ok:
            return {"answer": "", "contexts": [], "mode": mode or "mix", "error": reason}

        try:
            rag = await _get_instance(course_id)
            query_mode = (mode or LIGHTRAG_QUERY_MODE).strip() or "mix"
            param = _build_query_param(query_mode, history, only_need_context=False)

            t0 = time.perf_counter()
            result = await rag.aquery(message, param=param)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            log_flow("rag.query", course_id=course_id, query_mode=query_mode,
                     elapsed_ms=elapsed_ms, query=message[:60])

            if isinstance(result, dict):
                answer = (
                    result.get("response")
                    or result.get("answer")
                    or result.get("content")
                    or ""
                )
                contexts = _extract_contexts(result)
            else:
                answer = str(result)
                contexts = _extract_contexts(result)

            return {
                "answer": answer,
                "contexts": contexts,
                "mode": query_mode,
            }

        except Exception as exc:
            logger.error("LightRAGRetriever.query failed: %s", exc, exc_info=True)
            return {"answer": "", "contexts": [], "mode": mode or "mix", "error": str(exc)}
        finally:
            await _release_instance(course_id)


# ── 向后兼容函数（deprecated，将从 lightrag_engine.py 迁移）────────────────────────

async def query_with_lightrag(
    course_id: str,
    message: str,
    history: list[dict] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """向后兼容：使用 LightRAG 查询。

    Deprecated: 请使用 get_retriever("lightrag").query() 代替。
    """
    retriever = LightRAGRetriever()
    return await retriever.query(course_id, message, history, mode)


__all__ = [
    "LightRAGRetriever",
    "query_with_lightrag",
]
