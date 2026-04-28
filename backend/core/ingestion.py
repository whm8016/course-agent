"""LightRAG 摄入流水线（管理端「开始索引」）。

解析与切块与 rag_llama/indexing_documents + rag_llama/llamaindex_pipeline 共用；
LightRAG 仅负责 ainsert。
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from config import (
    INGEST_CHUNK_OVERLAP,
    INGEST_CHUNK_SIZE,
    LLAMA_CLOUD_API_KEY,
    LIGHTRAG_INGEST_CHUNKS_SNAPSHOT,
    LIGHTRAG_INGEST_CHUNKS_SUBDIR,
    LIGHTRAG_SAVE_INGEST_CHUNKS,
    LIGHTRAG_WORKDIR,
)
from rag_llama.indexing_documents import (
    LLAMA_INDEX_CHUNK_OVERLAP,
    LLAMA_INDEX_CHUNK_SIZE,
    file_paths_to_llama_documents,
)

logger = logging.getLogger(__name__)


# ── 索引控制（暂停/终止）─────────────────────────────────────────────────────
#
# 控制信号通过 Redis key 跨 worker 传递：
#     indexing:ctrl:{kb_id} → "pause" | "stop"
# 任何 worker 的 pause/stop API 都写这个 key；运行索引的 worker 在每个 batch
# 边界 GET 一次。这样多 worker（gunicorn -w N）部署下也能命中目标任务。

AbortAction = Literal["pause", "stop"]

CTRL_KEY_PREFIX = "indexing:ctrl:"
# 控制信号的 TTL：足够覆盖一次索引任务的最长执行时间，防止僵尸 key
# 阻塞下次启动；结束时也会主动 clear() 一次。
_CTRL_TTL_SECONDS = 6 * 3600


class IndexingAborted(Exception):
    """由 IndexingControl 在批次检查点抛出，用于中断索引循环。"""

    def __init__(self, action: AbortAction, chunks_done: int = 0):
        self.action = action
        self.chunks_done = chunks_done
        super().__init__(f"indexing aborted: {action}")


class IndexingControl:
    """基于 Redis 的索引控制器（跨 worker 进程）。

    - request_pause()：写 "pause"（不覆盖已有 stop）；下一个 batch 边界中止，
      保留 chunks_done，可后续续传。
    - request_stop() ：写 "stop"（覆盖 pause）；下一个 batch 边界中止，
      调用方负责清零进度。
    - checkpoint()   ：在运行索引的 worker 内从 Redis 读取信号；命中则抛
      IndexingAborted。
    - clear()        ：清除残留信号（启动前 / 结束时调用）。
    """

    def __init__(self, kb_id: str) -> None:
        self.kb_id = kb_id
        self._key = f"{CTRL_KEY_PREFIX}{kb_id}"

    async def _redis(self):
        # 延迟导入，避免 ingestion 在没有 Redis 配置时也无法导入
        from core.cache import _get_pool
        return _get_pool()

    async def clear(self) -> None:
        try:
            r = await self._redis()
            await r.delete(self._key)
        except Exception:
            logger.debug("IndexingControl.clear failed kb=%s", self.kb_id, exc_info=True)

    async def request_pause(self) -> None:
        try:
            r = await self._redis()
            # nx=True：只有当 key 不存在时才设置，避免覆盖已下达的 stop
            await r.set(self._key, "pause", ex=_CTRL_TTL_SECONDS, nx=True)
        except Exception:
            logger.warning("IndexingControl.request_pause failed kb=%s", self.kb_id, exc_info=True)
            raise

    async def request_stop(self) -> None:
        try:
            r = await self._redis()
            # stop 覆盖 pause
            await r.set(self._key, "stop", ex=_CTRL_TTL_SECONDS)
        except Exception:
            logger.warning("IndexingControl.request_stop failed kb=%s", self.kb_id, exc_info=True)
            raise

    async def checkpoint(self, chunks_done: int = 0) -> None:
        try:
            r = await self._redis()
            action = await r.get(self._key)
        except Exception:
            # Redis 抖动时不阻断索引，下一个 batch 再读
            logger.debug("IndexingControl.checkpoint read failed kb=%s", self.kb_id, exc_info=True)
            return
        if action in ("pause", "stop"):
            raise IndexingAborted(action, chunks_done)

# 每个 chunk 的固定 token 估算开销（LightRAG 实体抽取提示词约 2000 token）
_TOKEN_OVERHEAD_PER_CHUNK = 2000

# ── LlamaIndex（可选）────────────────────────────────────────────────────────
try:
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import Document as LlamaDocument
    logger.info("LlamaIndex 可用，将使用智能文档解析")
except ImportError as e:
    raise ImportError(
        "llama-index-core 未安装，请执行: pip install llama-index-core"
    ) from e

# ── LlamaParse（可选，图像 PDF 专用）─────────────────────────────────────────
try:
    from llama_cloud_services import LlamaParse
    _llamaparse_available = True
except ImportError:
    try:
        from llama_parse import LlamaParse  # type: ignore[no-redef]
        _llamaparse_available = True
    except ImportError:
        _llamaparse_available = False
        logger.warning("llama-cloud-services 未安装，图像 PDF 将无法解析；请执行: pip install llama-cloud-services")
    


# ── 降级文本读取 ────────────────────────────────────────────────────────────

def _read_text_fallback(file_path: str) -> str:
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("无法读取文件 %s: %s", file_path, e)
        return ""


def _split_text(
    text: str,
    chunk_size: int = INGEST_CHUNK_SIZE,
    overlap: int = INGEST_CHUNK_OVERLAP,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ── 核心解析函数 ─────────────────────────────────────────────────────────────

_IMAGE_PDF_THRESHOLD = 0.3  # 空文档占比超过此值时认定为图像 PDF（扫描件）

def _llamaparse_pdfs(pdf_paths: list[str]) -> list[LlamaDocument]:
    """用 LlamaParse 解析图像 PDF，返回 LlamaIndex Document 列表。"""
    if not _llamaparse_available or not LLAMA_CLOUD_API_KEY:
        logger.warning(
            "LlamaParse 不可用（llama-parse 未安装或 LLAMA_CLOUD_API_KEY 未配置），"
            "图像 PDF 将跳过：%s",
            pdf_paths,
        )
        return []

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="text",
        language="ch_sim",
    )
    logger.info("LlamaParse 开始解析 %d 个 PDF...", len(pdf_paths))
    docs = parser.load_data(pdf_paths)
    logger.info("LlamaParse 解析完成: %d 个文档", len(docs))
    return docs


def parse_files(file_paths: list[str]) -> list[str]:
    """
    解析文件列表，返回文本 chunk 列表（供 LightRAG 摄入）。
    路由与 llamaindex_pipeline 一致；SentenceSplitter 固定 1200/120 与 Settings 对齐。
    图像 PDF 仍可能走 LlamaParse（需 LLAMA_CLOUD_API_KEY）。
    """
    if not file_paths:
        return []

    documents, classification = file_paths_to_llama_documents(
        file_paths, log=logger
    )

    empty_count = sum(1 for d in documents if not d.get_content().strip())
    if (
        documents
        and empty_count / len(documents) > _IMAGE_PDF_THRESHOLD
        and classification.parser_files
    ):
        logger.info(
            "检测到图像 PDF（%d/%d 个文档内容为空），尝试 LlamaParse 重新解析...",
            empty_count,
            len(documents),
        )
        pdf_paths = list(classification.parser_files)
        non_pdf_docs = [d for d in documents if d.get_content().strip()]
        if pdf_paths:
            parsed_docs = _llamaparse_pdfs(pdf_paths)
            documents = non_pdf_docs + parsed_docs

    if not documents:
        logger.warning("摄入解析结果为空（无有效文档）")
        return []

    nodes = SentenceSplitter(
        chunk_size=LLAMA_INDEX_CHUNK_SIZE,
        chunk_overlap=LLAMA_INDEX_CHUNK_OVERLAP,
    ).get_nodes_from_documents(documents)
    chunks = [n.get_content() for n in nodes if n.get_content().strip()]
    logger.info(
        "摄入解析完成: %d 个输入文件 → %d 个文档 → %d 个 chunk",
        len(file_paths),
        len(documents),
        len(chunks),
    )
    return chunks


# ── 完整摄入流水线 ───────────────────────────────────────────────────────────

ProgressCallback = Optional[Callable[..., Awaitable[None]]]


def _lightrag_ingest_chunks_dir(course_id: str) -> Path:
    """lightrag_store/course_{course_id}/ingest_chunks/（与 core.lightrag_engine workspace 命名一致）。"""
    return Path(LIGHTRAG_WORKDIR) / f"course_{course_id}" / LIGHTRAG_INGEST_CHUNKS_SUBDIR


def _persist_lightrag_ingest_chunks(
    course_id: str,
    file_paths: list[str],
    all_chunks: list[str],
    resume_from_chunk: int,
) -> Path | None:
    """
    将摄入前切分好的文本块写入 JSON（供排查/审计；不参与 LightRAG 加载）。
    返回写入的 latest.json 路径；失败时记日志并返回 None。
    """
    if not LIGHTRAG_SAVE_INGEST_CHUNKS:
        return None
    out_dir = _lightrag_ingest_chunks_dir(course_id)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "course_id": course_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "source_files": [str(Path(p).name) for p in file_paths],
            "source_paths": [str(Path(p).resolve()) for p in file_paths],
            "chunk_count": len(all_chunks),
            "resume_from_chunk_at_save": resume_from_chunk,
            "chunks": all_chunks,
        }
        latest = out_dir / "latest.json"
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        latest.write_text(text, encoding="utf-8")
        if LIGHTRAG_INGEST_CHUNKS_SNAPSHOT:
            ts_name = f"chunks_{int(datetime.now(timezone.utc).timestamp())}.json"
            (out_dir / ts_name).write_text(text, encoding="utf-8")
        logger.info(
            "已保存 LightRAG 摄入切块 course=%s dir=%s chunks=%d",
            course_id,
            out_dir,
            len(all_chunks),
        )
        return latest
    except OSError as e:
        logger.warning("保存摄入切块失败 course=%s: %s", course_id, e)
        return None


async def ingest_to_lightrag(
    course_id: str,
    file_paths: list[str],
    batch_size: int = 4,
    on_progress: ProgressCallback = None,
    resume_from_chunk: int = 0,
    control: Optional[IndexingControl] = None,
) -> dict:
    """
    完整摄入流水线：与 LlamaIndex 同策略解析切块 → LightRAG ainsert。

    Args:
        course_id:         课程 ID
        file_paths:        待摄入文件列表
        batch_size:        每批插入 LightRAG 的 chunk 数
        on_progress:       异步进度回调
        resume_from_chunk: 断点续传：从第 N 个 chunk 开始（跳过前 N 个）
        control:           可选的 IndexingControl，用于在 batch 边界中止任务
    """
    from core.lightrag_engine import (
        _get_instance, is_lightrag_available,
        take_llm_errors, clear_llm_errors, _is_fatal_llm_error,
    )

    async def _emit(progress: int, msg: str, chunks_done: int, chunks_total: int, token_estimate: int):
        logger.info("进度 %d%% | %s | chunk %d/%d | token≈%d",
                    progress, msg, chunks_done, chunks_total, token_estimate)
        if on_progress:
            await on_progress(
                progress=progress,
                msg=msg,
                chunks_done=chunks_done,
                chunks_total=chunks_total,
                token_estimate=token_estimate,
            )

    ok, reason = is_lightrag_available()
    if not ok:
        raise RuntimeError(f"LightRAG 不可用: {reason}")

    # 清空上次遗留的错误记录
    clear_llm_errors()

    # Step 1: 解析文件（CPU 密集，放线程池）
    is_resume = resume_from_chunk > 0
    parse_label = f"续传解析 {len(file_paths)} 个文件（将跳过前 {resume_from_chunk} 个文本块）…" \
        if is_resume else f"开始解析 {len(file_paths)} 个文件…"
    await _emit(5, parse_label, resume_from_chunk, 0, 0)
    logger.info("开始解析 %d 个文件 course=%s resume_from=%d", len(file_paths), course_id, resume_from_chunk)
    all_chunks = await asyncio.to_thread(parse_files, file_paths)

    if not all_chunks:
        logger.warning("解析结果为空 course=%s", course_id)
        await _emit(100, "解析结果为空，无可索引内容", 0, 0, 0)
        return {"status": "empty", "chunks": 0, "files": len(file_paths)}

    await asyncio.to_thread(
        _persist_lightrag_ingest_chunks,
        course_id,
        file_paths,
        all_chunks,
        resume_from_chunk,
    )

    total = len(all_chunks)
    avg_chars = sum(len(c) for c in all_chunks) / total
    token_per_chunk = int(_TOKEN_OVERHEAD_PER_CHUNK + avg_chars / 3.5)

    # 断点续传：跳过已处理的 chunk
    start = min(resume_from_chunk, total)
    chunks = all_chunks[start:]
    already_done = start

    resume_note = f"（已跳过 {already_done} 个，续传）" if is_resume else ""
    await _emit(
        10,
        f"解析完成：{len(file_paths)} 个文件 → {total} 个文本块（均长 {int(avg_chars)} 字符）{resume_note}",
        already_done, total, already_done * token_per_chunk,
    )

    if not chunks:
        await _emit(100, "所有文本块均已索引完毕", total, total, total * token_per_chunk)
        return {"status": "done", "chunks": total, "files": len(file_paths)}

    # Step 2: 分批写入 LightRAG
    logger.info("开始写入 LightRAG: %d 个 chunk（跳过 %d），course=%s", len(chunks), already_done, course_id)
    rag = await _get_instance(course_id)

    for i in range(0, len(chunks), batch_size):
        # 暂停/终止检查点：在每个 batch 写入前从 Redis 读一次控制信号
        if control is not None:
            await control.checkpoint(chunks_done=already_done + i)

        batch = chunks[i : i + batch_size]
        await rag.ainsert(batch)

        # ── 检查 LLM 错误（LightRAG 内部吞掉了异常，我们在 _llm_model_func 中记录）──
        errors = take_llm_errors()
        if errors:
            fatal = [e for e in errors if _is_fatal_llm_error(e)]
            if fatal:
                err_msg = str(fatal[0])
                done_so_far = already_done + min(i + batch_size, len(chunks))
                await _emit(
                    int(10 + done_so_far / total * 85),
                    f"遇到致命错误，索引中止（已完成 {done_so_far}/{total} 个文本块）",
                    done_so_far, total, done_so_far * token_per_chunk,
                )
                raise RuntimeError(f"LLM API 致命错误，索引中止: {err_msg[:300]}")
            else:
                # 非致命错误（如网络超时）仅记录日志，继续
                logger.warning("非致命 LLM 错误（继续）: %s", errors[0])

        done = already_done + min(i + batch_size, len(chunks))
        progress = 10 + int(done / total * 85)
        token_estimate = done * token_per_chunk
        await _emit(
            progress,
            f"构建知识图谱：{done}/{total} 个文本块",
            done, total, token_estimate,
        )
        logger.info("LightRAG 摄入进度 course=%s %d/%d", course_id, done, total)

    final_tokens = total * token_per_chunk
    await _emit(
        100,
        f"索引完成：{len(file_paths)} 个文件，{total} 个文本块，估算消耗 {final_tokens:,} tokens",
        total, total, final_tokens,
    )
    logger.info("摄入完成 course=%s files=%d chunks=%d", course_id, len(file_paths), total)
    return {"status": "done", "chunks": total, "files": len(file_paths)}


def llama_available() -> bool:
    """返回 LlamaIndex 是否可用（供 API 展示）。"""
    return True
