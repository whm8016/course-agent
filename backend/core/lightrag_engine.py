from __future__ import annotations
import asyncio

import logging
import os
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    KNOWLEDGE_DIR,
    LIGHTRAG_EMBEDDING_DIM,
    LIGHTRAG_ENABLED,
    LIGHTRAG_ENABLE_RERANK,
    LIGHTRAG_AUTO_INDEX_TTL_SEC,
    LIGHTRAG_AGENTIC_RAG_MAX_CHARS,
    LIGHTRAG_QUERY_MODE,
    LIGHTRAG_STREAM_CONTEXT_LIMIT,
    LIGHTRAG_STREAM_CONTEXT_MAX_CHARS,
    LIGHTRAG_TOP_K,
    LIGHTRAG_WORKDIR,
    TEXT_MODEL,
)
from core.llm import chat_stream
from core.prompts import get_course_prompt

logger = logging.getLogger(__name__)

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    LIGHTRAG_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - import availability branch
    LightRAG = None  # type: ignore[assignment]
    QueryParam = None  # type: ignore[assignment]
    openai_complete_if_cache = None  # type: ignore[assignment]
    openai_embed = None  # type: ignore[assignment]
    wrap_embedding_func_with_attrs = None  # type: ignore[assignment]
    LIGHTRAG_IMPORT_ERROR = exc

_instances: dict[str, Any] = {}
_init_locks: dict[str, asyncio.Lock] = {}
_index_locks: dict[str, asyncio.Lock] = {}
_index_signatures: dict[str, tuple[str, ...]] = {}
_last_auto_index_at: dict[str, float] = {}
_AUTO_INDEX_STATE_DIR = Path(LIGHTRAG_WORKDIR) / ".auto_index_state"
_AUTO_INDEX_LOCK_DIR = Path(LIGHTRAG_WORKDIR) / ".auto_index_locks"

# ── LLM 错误收集（LightRAG 内部会吞掉异常，这里在抛出前记录）─────────────────
_llm_error_log: list[Exception] = []


def take_llm_errors() -> list[Exception]:
    """取出并清空已记录的 LLM 错误列表（每批插入后调用）。"""
    errors = _llm_error_log.copy()
    _llm_error_log.clear()
    return errors


def clear_llm_errors() -> None:
    """清空错误缓冲（开始新索引前调用）。"""
    _llm_error_log.clear()


def _is_fatal_llm_error(exc: Exception) -> bool:
    """判断是否为致命错误（账户余额/权限问题），不可重试。"""
    s = str(exc).lower()
    fatal_keywords = ("access denied", "account", "unauthorized", "authentication",
                      "bad request", "quota", "insufficient")
    if any(kw in s for kw in fatal_keywords):
        return True
    import re
    m = re.search(r"error code[:\s]+(\d+)", s)
    if m and int(m.group(1)) in (400, 401, 403):
        return True
    return False

# DashScope compatible API rejects oversized input (>30720).
# Use conservative hard caps even if .env config is too aggressive.
_SAFE_TOP_K = min(LIGHTRAG_TOP_K, int(os.getenv("LIGHTRAG_SAFE_TOP_K", "10")))
_SAFE_CHUNK_TOP_K = min(int(os.getenv("LIGHTRAG_CHUNK_TOP_K", "8")), _SAFE_TOP_K)
_SAFE_MAX_TOTAL_TOKENS = min(int(os.getenv("LIGHTRAG_MAX_TOTAL_TOKENS", "22000")), 26000)
_SAFE_MAX_ENTITY_TOKENS = min(int(os.getenv("LIGHTRAG_MAX_ENTITY_TOKENS", "4000")), 6000)
_SAFE_MAX_RELATION_TOKENS = min(int(os.getenv("LIGHTRAG_MAX_RELATION_TOKENS", "4000")), 6000)
_SAFE_MAX_HISTORY_MESSAGES = int(os.getenv("LIGHTRAG_MAX_HISTORY_MESSAGES", "8"))
_SAFE_MAX_HISTORY_CHARS = int(os.getenv("LIGHTRAG_MAX_HISTORY_CHARS", "8000"))

_AUTO_INDEX_TTL_SEC = max(0, LIGHTRAG_AUTO_INDEX_TTL_SEC)

_STREAM_CONTEXT_LIMIT = max(1, LIGHTRAG_STREAM_CONTEXT_LIMIT)
_STREAM_CONTEXT_MAX_CHARS = max(200, LIGHTRAG_STREAM_CONTEXT_MAX_CHARS)
_AGENTIC_RAG_MAX_CHARS = max(2000, LIGHTRAG_AGENTIC_RAG_MAX_CHARS)


def is_lightrag_available() -> tuple[bool, str]:
    if not LIGHTRAG_ENABLED:
        return False, "LIGHTRAG_ENABLED 未开启"
    if LIGHTRAG_IMPORT_ERROR is not None:
        return False, f"LightRAG 依赖不可用: {LIGHTRAG_IMPORT_ERROR}"
    if not DASHSCOPE_API_KEY:
        return False, "缺少 DASHSCOPE_API_KEY"
    return True, ""


async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    assert openai_complete_if_cache is not None
    max_sys_chars = int(os.getenv("LIGHTRAG_LLM_SYSTEM_MAX_CHARS", "24000"))
    safe_system_prompt = system_prompt
    if max_sys_chars > 0 and safe_system_prompt and len(safe_system_prompt) > max_sys_chars:
        safe_system_prompt = safe_system_prompt[:max_sys_chars]
    try:
        return await openai_complete_if_cache(
            TEXT_MODEL,
            prompt,
            system_prompt=safe_system_prompt,
            history_messages=history_messages or [],
            keyword_extraction=keyword_extraction,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            **kwargs,
        )
    except Exception as exc:
        # LightRAG 内部会捕获此异常并继续，但我们在此先记录
        _llm_error_log.append(exc)
        logger.error("LLM 调用失败（已记录）: %s", exc)
        raise


if wrap_embedding_func_with_attrs is not None and openai_embed is not None:

    @wrap_embedding_func_with_attrs(
        embedding_dim=LIGHTRAG_EMBEDDING_DIM,
        max_token_size=8192,
        model_name=EMBEDDING_MODEL,
    )
    async def _embedding_func(texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts,
            model=EMBEDDING_MODEL,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
        )

else:

    async def _embedding_func(texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise RuntimeError("LightRAG embedding function unavailable")


def _workspace_name(course_id: str) -> str:
    return f"course_{course_id}"


def _resolve_source_dir(course_id: str, source_dir: str | None = None) -> Path:
    if source_dir:
        return Path(source_dir).expanduser().resolve()
    return (Path(KNOWLEDGE_DIR) / course_id).resolve()


def _build_signature(file_paths: list[str]) -> tuple[str, ...]:
    signature: list[str] = []
    for file_path in sorted(file_paths):
        path = Path(file_path)
        stat = path.stat()
        signature.append(f"{file_path}|{stat.st_mtime_ns}|{stat.st_size}")
    return tuple(signature)




def _collect_course_docs(source_dir: Path, course_id: str) -> tuple[list[str], list[str], list[str]]:
    if not source_dir.is_dir():
        return [], [], []

    docs: list[str] = []
    ids: list[str] = []
    file_paths: list[str] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError:
            continue
        if not content.strip():
            continue
        docs.append(content)
        ids.append(f"{course_id}:{path.relative_to(source_dir).as_posix()}")
        file_paths.append(str(path.resolve()))
    return docs, ids, file_paths


async def _get_instance(course_id: str):
    ok, reason = is_lightrag_available()
    if not ok:
        raise RuntimeError(reason)

    if course_id in _instances:
        return _instances[course_id]

    lock = _init_locks.setdefault(course_id, asyncio.Lock())
    async with lock:
        if course_id in _instances:
            return _instances[course_id]

        os.makedirs(LIGHTRAG_WORKDIR, exist_ok=True)
        assert LightRAG is not None
        rag = LightRAG(
            working_dir=LIGHTRAG_WORKDIR,
            workspace=_workspace_name(course_id),
            llm_model_func=_llm_model_func,
            embedding_func=_embedding_func,
        )
        await rag.initialize_storages()
        _instances[course_id] = rag
        logger.info("LightRAG initialized for course=%s workspace=%s", course_id, _workspace_name(course_id))
        return rag


async def index_course_with_lightrag(
    course_id: str,
    force: bool = False,
    source_dir: str | None = None,
) -> dict[str, Any]:
    rag = await _get_instance(course_id)
    resolved_dir = _resolve_source_dir(course_id, source_dir)
    if not resolved_dir.is_dir():
        return {"indexed_docs": 0, "indexed_files": 0, "skipped": False, "reason": "source_dir_not_found"}

    all_files = [str(p.resolve()) for p in sorted(resolved_dir.rglob("*")) if p.is_file()]
    if not all_files:
        return {"indexed_docs": 0, "indexed_files": 0, "skipped": False, "reason": "no_files"}

    signature = _build_signature(all_files)
    cache_key = f"{course_id}|{resolved_dir}"
    if not force and _index_signatures.get(cache_key) == signature:
        return {"indexed_docs": 0, "indexed_files": 0, "skipped": True, "source_dir": str(resolved_dir)}

    indexed_files = 0
    if hasattr(rag, "ainsert_files"):
        await rag.ainsert_files(all_files)
        indexed_files = len(all_files)

    docs, ids, text_file_paths = _collect_course_docs(resolved_dir, course_id)
    indexed_docs = 0
    if docs:
        await rag.ainsert(docs, ids=ids, file_paths=text_file_paths)
        indexed_docs = len(docs)

    _index_signatures[cache_key] = signature
    logger.info(
        "LightRAG indexed course=%s files=%d docs=%d source_dir=%s",
        course_id,
        indexed_files,
        indexed_docs,
        resolved_dir,
    )
    return {
        "indexed_docs": indexed_docs,
        "indexed_files": indexed_files,
        "skipped": False,
        "source_dir": str(resolved_dir),
    }


def _normalize_history(history: list[dict] | None) -> list[dict[str, str]]:
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
    if not messages:
        return []
    capped = messages[-_SAFE_MAX_HISTORY_MESSAGES:]
    total_chars = sum(len(m["content"]) for m in capped)
    while len(capped) > 1 and total_chars > _SAFE_MAX_HISTORY_CHARS:
        capped = capped[1:]
        total_chars = sum(len(m["content"]) for m in capped)
    return capped

# 替换 LightRAG 内置 rag_response：去掉强制 References 章节，保留三个必要占位符
_LIGHTRAG_RESPONSE_TEMPLATE = """---角色---

你是擅长综合知识库信息的 AI 助手，须**仅**依据提供的 **Context（上下文）** 准确回答用户问题。

---目标---

对用户问题给出全面、结构清晰的回答。
回答须整合 **Context** 中「知识图谱数据」与「文档片段」的事实。
若提供对话历史，请保持连贯并避免重复已说内容。

---说明---

1. 步骤指引：
  - 结合对话历史判断用户意图与信息需求。
  - 仔细查阅 **Context** 中的 `Knowledge Graph Data` 与 `Document Chunks`，提取与问题直接相关的全部信息。
  - 将事实组织成连贯回答；你的自身知识**仅**用于润色句子和衔接，**不得**引入上下文未出现的信息。

2. 内容与依据：
  - 严格遵循 **Context**；**不得**编造、臆测或推断未明确陈述的内容。
  - 若 **Context** 不足以回答，请说明信息不足，不要猜测。

3. 格式与语言：
  - 回答语言须与用户问题语言一致。
  - 回答须使用 Markdown（标题、加粗、列表等）以提升可读性。
  - 回答以 {response_type} 形式呈现。

4. 来源处理：
  - **不要**在答案末尾单独列出参考文献、References、参考依据等章节。
  - 若需提及来源，将其自然融入正文（如"根据实验教案..."）。

5. 附加说明：{user_prompt}


---Context（上下文）---

{context_data}
"""

def _build_query_param(
    mode: str,
    history: list[dict] | None,
    *,
    only_need_context: bool = False,
):
    assert QueryParam is not None
    param = QueryParam(
        mode=mode,
        top_k=_SAFE_TOP_K,
        chunk_top_k=_SAFE_CHUNK_TOP_K,
        max_total_tokens=_SAFE_MAX_TOTAL_TOKENS,
        max_entity_tokens=_SAFE_MAX_ENTITY_TOKENS,
        max_relation_tokens=_SAFE_MAX_RELATION_TOKENS,
        conversation_history=_cap_history(_normalize_history(history)),
        enable_rerank=LIGHTRAG_ENABLE_RERANK,
    )
    if only_need_context:
        # Compatible with multiple LightRAG versions.
        if hasattr(param, "only_need_context"):
            setattr(param, "only_need_context", True)
        if hasattr(param, "return_context_only"):
            setattr(param, "return_context_only", True)
        if hasattr(param, "need_response"):
            setattr(param, "need_response", False)
    return param


def _extract_contexts(result: Any) -> list[Any]:
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


async def query_with_lightrag(
    course_id: str,
    message: str,
    history: list[dict] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    rag = await _get_instance(course_id)
    
    await index_course_with_lightrag(course_id)

    query_mode = (mode or LIGHTRAG_QUERY_MODE).strip() or "mix"
    param = _build_query_param(query_mode, history, only_need_context=False)
    result = await rag.aquery(message, param=param) 

    logger.info('query_with_lightrag result', result)

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
    logger.info('query_with_lightrag contexts', contexts)
    return {
        "answer": answer,
        "contexts": contexts,
        "mode": query_mode,
    }


def _extract_context_text(ctx: Any) -> str:
    if isinstance(ctx, str):
        return ctx.strip()
    if isinstance(ctx, dict):
        for key in ("content", "text", "chunk", "passage"):
            value = ctx.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(ctx).strip()


def _format_contexts_for_prompt(contexts: list[Any], limit: int = _STREAM_CONTEXT_LIMIT) -> str:
    rows: list[str] = []
    for idx, ctx in enumerate(contexts[:limit]):
        text = _extract_context_text(ctx)
        if not text:
            continue
        if len(text) > _STREAM_CONTEXT_MAX_CHARS:
            text = f"{text[:_STREAM_CONTEXT_MAX_CHARS]}...(truncated)"
        rows.append(f"[证据{idx + 1}]\n{text}")
    return "\n\n---\n\n".join(rows)


async def retrieve_with_lightrag(
    course_id: str,
    message: str,
    history: list[dict] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve contexts from LightRAG first.
    Prefer context-only retrieval if supported by installed LightRAG version.
    """
    logger.info(
        "retrieve_with_lightrag course=%s mode=%s query=「%s」",
        course_id, mode, message[:80],
    )
    rag = await _get_instance(course_id)
    idx_lock = _index_locks.setdefault(course_id, asyncio.Lock())
    # async with idx_lock:
    #     now = time.monotonic()
    #     last_index = _last_auto_index_at.get(course_id, 0.0)
    #     if _AUTO_INDEX_TTL_SEC == 0 or (now - last_index) >= _AUTO_INDEX_TTL_SEC:
    #         index_t0 = time.perf_counter()
    #         index_result = await index_course_with_lightrag(course_id)
    #         _last_auto_index_at[course_id] = now
    #         logger.info(
    #             "LightRAG auto-index course=%s skipped=%s elapsed_ms=%d",
    #             course_id,
    #             index_result.get("skipped"),
    #             int((time.perf_counter() - index_t0) * 1000),
    #         )

    # query_mode = (mode or LIGHTRAG_QUERY_MODE).strip() or "mix"
    # param = _build_query_param(query_mode, history, only_need_context=False)
    # context_param = _build_query_param(query_mode, history, only_need_context=False)
    # # 打开 LightRAG 原生流式输出：aquery 返回 AsyncIterator[str]
    # if hasattr(context_param, "stream"):
    #     context_param.stream = True
    # if hasattr(param, "stream"):
    #     param.stream = True

    # logger.info("QueryParam: %s", context_param)
    # retrieve_strategy = "aquery_context_param"
    # try:
    #     result: Any = await rag.aquery(message, param=context_param)
    # except TypeError:
    #     retrieve_strategy = "aquery_fallback"
    #     result = await rag.aquery(message, param=param)
    # return result
    query_mode = (mode or LIGHTRAG_QUERY_MODE).strip() or "mix"

    course_system_prompt = await get_course_prompt(course_id)
    lightrag_system_prompt = course_system_prompt + "\n\n" + _LIGHTRAG_RESPONSE_TEMPLATE

    param = _build_query_param(query_mode, history, only_need_context=False)
    if hasattr(param, "stream"):
        param.stream = True

    logger.info("QueryParam: %s", param)
    try:
        result: Any = await rag.aquery(message, param=param, system_prompt=lightrag_system_prompt)
    except TypeError:
        # 旧版 LightRAG 不接受 system_prompt 参数，退回默认
        result = await rag.aquery(message, param=param)
    return result

async def agentic_pipeline(
    course_id: str,
    message: str,
    history: list[dict] | None = None,
    mode: str | None = None,
    enabled_tools: list[str] | None = None,
    image_path: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    四阶段 pipeline（对标 DeepTutor）：
      Thinking → Acting（执行用户选定工具） → Observing → Responding
    yield dict：
      {'type': 'stage', 'stage': 'thinking'|'retrieving'|'observing'|'responding',
       'state': 'start'|'done', 'content': str}
      {'type': 'stage_chunk', 'stage': str, 'content': str}
      {'type': 'token', 'content': str}
    """
    from core.prompts import (
        THINKING_PROMPT, THINKING_KB_HINT, THINKING_TOOL_LIST_PREFIX,
        OBSERVING_PROMPT, RESPONDING_PROMPT, _CONCISE_SUFFIX,
    )

    course_prompt = await get_course_prompt(course_id)
    safe_history = _cap_history(_normalize_history(history))
    _tools = enabled_tools or []

    # 动态拼 thinking prompt：有 rag 工具时注入 kb_hint，同时附上启用工具列表
    _kb_hint = THINKING_KB_HINT if "rag" in _tools else ""
    _tool_list_text = (THINKING_TOOL_LIST_PREFIX + "\n".join(f"- {t}" for t in _tools)) if _tools else ""
    _thinking_prompt = THINKING_PROMPT.replace("{kb_hint}", _kb_hint)
    if _tool_list_text:
        _thinking_prompt += "\n\n" + _tool_list_text

    # ── Stage 1: Thinking ────────────────────────────────────────────
    logger.info("agentic_pipeline [thinking] start course=%s tools=%s", course_id, _tools)
    yield {"type": "stage", "stage": "thinking", "state": "start", "content": ""}
    await asyncio.sleep(0)
    thinking_chunks: list[str] = []
    async for token in chat_stream(
        system_prompt=course_prompt + "\n\n" + _thinking_prompt,
        history=safe_history,
        user_message=message,
        image_path=image_path,
    ):
        thinking_chunks.append(token)
        yield {"type": "stage_chunk", "stage": "thinking", "content": token}
    thinking = "".join(thinking_chunks).strip()
    logger.info("agentic_pipeline [thinking] done chars=%d", len(thinking))
    yield {"type": "stage", "stage": "thinking", "state": "done", "content": ""}
    await asyncio.sleep(0)

    # ── Stage 2: Acting（执行用户选定的工具）──────────────────────────
    logger.info("agentic_pipeline [acting] start course=%s tools=%s", course_id, _tools)
    yield {"type": "stage", "stage": "retrieving", "state": "start", "content": ""}
    await asyncio.sleep(0)

    async def _run_rag() -> dict:
        try:
            rag = await _get_instance(course_id)
            query_mode = (mode or LIGHTRAG_QUERY_MODE).strip() or "mix"
            # 先尝试 only_need_context=True 拿纯 context 文本
            context_param = _build_query_param(query_mode, history, only_need_context=True)
            if hasattr(context_param, "stream"):
                context_param.stream = False
            raw = await rag.aquery(message, param=context_param)
            content = raw.strip() if isinstance(raw, str) else ""
            # 如果 only_need_context 模式返回的不是字符串（部分 LightRAG 版本返回结构体），
            # 则退回完整查询，用返回的答案文本作为证据
            if not content:
                full_param = _build_query_param(query_mode, history, only_need_context=False)
                if hasattr(full_param, "stream"):
                    full_param.stream = False
                raw2 = await rag.aquery(message, param=full_param)
                content = raw2.strip() if isinstance(raw2, str) else str(raw2 or "").strip()
                logger.info(
                    "agentic_pipeline [rag] only_need_context returned empty, fallback to full query chars=%d",
                    len(content),
                )
            if len(content) > _AGENTIC_RAG_MAX_CHARS:
                content = content[:_AGENTIC_RAG_MAX_CHARS] + "\n...(truncated)"
            _preview = (content[:800] + "…") if len(content) > 800 else content
            logger.info(
                "agentic_pipeline [rag] course=%s mode=%s query_chars=%d retrieved_chars=%d empty=%s\n"
                "--- RAG 检索结果预览（前 800 字）---\n%s\n--- end preview ---",
                course_id,
                query_mode,
                len(message),
                len(content),
                not bool(content),
                _preview or "（空）",
            )
            if content:
                logger.debug("agentic_pipeline [rag] full retrieved_context chars=%d:\n%s", len(content), content)
            return {"name": "rag", "query": message, "content": content, "success": bool(content)}
        except Exception:
            logger.exception("agentic_pipeline rag failed")
            return {"name": "rag", "query": message, "content": "（知识库检索失败）", "success": False}

    async def _run_llamaindex_rag() -> dict:
        try:
            from core.rag_llama import retrieve_chunks_llamaindex
            content = await retrieve_chunks_llamaindex(course_id, message)
            logger.info(
                "agentic_pipeline [llamaindex_rag] course=%s query_chars=%d result_chars=%d",
                course_id, len(message), len(content),
            )
            return {"name": "llamaindex_rag", "query": message, "content": content, "success": bool(content.strip())}
        except Exception:
            logger.exception("agentic_pipeline llamaindex_rag failed")
            return {"name": "llamaindex_rag", "query": message, "content": "（LlamaIndex 检索失败）", "success": False}

    async def _run_web_search() -> dict:
        try:
            from core.tool_registry import _execute_web_search
            result = await _execute_web_search(query=message)
            return {"name": "web_search", "query": message, "content": result.content, "success": result.success}
        except Exception:
            logger.exception("agentic_pipeline web_search failed")
            return {"name": "web_search", "query": message, "content": "（网络搜索失败）", "success": False}

    tasks = []
    if "rag" in _tools:
        yield {"type": "stage_chunk", "stage": "retrieving", "content": "检索知识库..."}
        tasks.append(_run_rag())
    if "llamaindex_rag" in _tools:
        yield {"type": "stage_chunk", "stage": "retrieving", "content": "检索 LlamaIndex 向量库..."}
        tasks.append(_run_llamaindex_rag())
    if "web_search" in _tools:
        yield {"type": "stage_chunk", "stage": "retrieving", "content": "搜索网络..."}
        tasks.append(_run_web_search())

    if tasks:
        results = await asyncio.gather(*tasks)
        tool_traces: list[dict] = list(results)
    else:
        tool_traces = []

    logger.info("agentic_pipeline [acting] done traces=%d", len(tool_traces))
    yield {"type": "stage", "stage": "retrieving", "state": "done",
           "content": f"完成 {len(tool_traces)} 个工具"}
    await asyncio.sleep(0)

    def _fmt_traces(traces: list[dict]) -> str:
        if not traces:
            return "（本轮未使用任何工具）"
        parts = []
        for t in traces:
            parts.append(f"[工具: {t['name']}]\n查询: {t['query']}\n结果:\n{t['content']}")
        return "\n\n---\n\n".join(parts)

    tool_trace_text = _fmt_traces(tool_traces)

    # ── Stage 3: Observing ───────────────────────────────────────────
    logger.info("agentic_pipeline [observing] start")
    yield {"type": "stage", "stage": "observing", "state": "start", "content": ""}
    await asyncio.sleep(0)
    obs_user = (
        f"[Thinking]\n{thinking}\n\n"
        f"[Tool Traces]\n{tool_trace_text}\n\n"
        f"用户问题：{message}\n\n"
        "请输出观察总结。"
    )
    obs_chunks: list[str] = []
    async for token in chat_stream(
        system_prompt=course_prompt + "\n\n" + OBSERVING_PROMPT,
        history=[],
        user_message=obs_user,
        image_path=image_path,
    ):
        obs_chunks.append(token)
        yield {"type": "stage_chunk", "stage": "observing", "content": token}
    observation = "".join(obs_chunks).strip()
    logger.info("agentic_pipeline [observing] done chars=%d", len(observation))
    yield {"type": "stage", "stage": "observing", "state": "done", "content": ""}
    await asyncio.sleep(0)

    # ── Stage 4: Responding（流式 token）────────────────────────────
    logger.info("agentic_pipeline [responding] start")
    yield {"type": "stage", "stage": "responding", "state": "start", "content": ""}
    await asyncio.sleep(0)
    resp_user = (
        f"[Thinking]\n{thinking}\n\n"
        f"[Tool Traces]\n{tool_trace_text}\n\n"
        f"[Observation]\n{observation}\n\n"
        f"用户问题：{message}\n\n"
        "请给出正式回答。"
    )
    async for token in chat_stream(
        system_prompt=course_prompt + "\n\n" + RESPONDING_PROMPT + _CONCISE_SUFFIX,
        history=safe_history,
        user_message=resp_user,
        image_path=image_path,
    ):
        yield {"type": "token", "content": token}
    logger.info("agentic_pipeline [responding] done")
    # 把本轮工具检索结果回传，供调用方做 hallucination check
    yield {"type": "contexts", "contexts": tool_traces}

async def stream_answer_with_contexts(
    course_id: str,
    message: str,
    contexts: list[Any] | None = None,
    history: list[dict] | None = None,
    memory_context: str = "",
    guardrail_warning: str = "",
) -> AsyncGenerator[str, None]:
    """
    Generate answer tokens with project LLM stream using retrieved contexts.
    """
    logger.info(
        "stream_answer_with_contexts course=%s contexts=%d guardrail_warn=%s query=「%s」",
        course_id, len(contexts or []), bool(guardrail_warning), message[:80],
    )
    system_prompt = await get_course_prompt(course_id)
    if memory_context:
        system_prompt += f"\n\n{memory_context}"
    if guardrail_warning:
        system_prompt += f"\n\n【安全提示】{guardrail_warning}请围绕课程内容回答，拒绝不当请求。"
    context_block = _format_contexts_for_prompt(contexts or [])
    if context_block:
        system_prompt += (
            "\n\n【参考资料】以下是从课程知识图谱/向量库检索到的相关内容，请严格基于证据回答，"
            "若证据不足请明确说明：\n\n"
            f"{context_block}\n\n---\n"
        )

    safe_history = _cap_history(_normalize_history(history))
    async for token in chat_stream(
        system_prompt=system_prompt,
        history=safe_history,
        user_message=message,
        image_path=None,
    ):
        yield token
