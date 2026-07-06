"""LightRAG 摄入流水线（管理端「开始索引」）。

解析与切块与 rag_llama/indexing_documents + rag_llama/llamaindex_pipeline 共用；
LightRAG 仅负责 ainsert。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from settings import get_settings
INGEST_CHUNK_OVERLAP = get_settings().chunking.ingest_overlap
INGEST_CHUNK_SIZE = get_settings().chunking.ingest_size
LLAMA_CLOUD_API_KEY = get_settings().llamaparse.cloud_api_key.get_secret_value()
LIGHTRAG_INGEST_BATCH_SIZE = get_settings().lightrag.ingest_batch_size
LIGHTRAG_INGEST_CHUNKS_SNAPSHOT = get_settings().lightrag.ingest_chunks_snapshot
LIGHTRAG_INGEST_CHUNKS_SUBDIR = get_settings().lightrag.ingest_chunks_subdir
LIGHTRAG_SAVE_INGEST_CHUNKS = get_settings().lightrag.save_ingest_chunks
LIGHTRAG_WORKDIR = get_settings().paths.lightrag_workdir
from core.rag.llamaindex.indexing_documents import (
    LLAMA_INDEX_CHUNK_OVERLAP,
    LLAMA_INDEX_CHUNK_SIZE,
    file_paths_to_llama_documents,
)

logger = logging.getLogger(__name__)

from core.observability import log_flow  # noqa: E402


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
        from core.db.cache import _get_pool
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


def _build_source_prefix(section: str = "", file_name: str = "") -> str:
    """构建来源前缀，注入到 chunk 文本开头，供 LightRAG 实体抽取 LLM 可见章节/文件来源。

    DOCX/PPTX 有 section → 注入章节名；PDF/TXT/MD 无 section → 退化用文件名。
    无任何来源信息时返回空串（不注入）。
    """
    if section:
        return f"【章节: {section}】\n"
    if file_name:
        return f"【来源: {file_name}】\n"
    return ""


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


def parse_files(file_paths: list[str]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """
    解析文件列表，返回 (文本 chunk 列表, chunk 来源路径列表, 按文件分段的全文 dict)。

    chunk_sources 与 chunks 等长，第 i 个元素 = 第 i 个 chunk 的来源 file_path，
    供 ingest_to_lightrag 传给 rag.ainsert(file_paths=...) 做来源溯源。
    每个 chunk 文本开头已注入【章节: xxx】/【来源: filename】前缀，供实体抽取 LLM 可见来源。

    doc_texts 供图片 VLM 摄入复用，避免重复提取：
      dict[绝对路径, list[str]]  — PDF 每页 / DOCX 每 section / PPTX 每 slide / 纯文本单元素
    """
    if not file_paths:
        return [], [], {}

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
        return [], [], {}

    # 构建 doc_texts：按文件路径分组，复用已解析的 Document 内容
    doc_texts: dict[str, list[str]] = {}
    for doc in documents:
        fp = doc.metadata.get("file_path", "")
        if not fp:
            continue
        content = doc.get_content().strip()
        if content:
            doc_texts.setdefault(fp, []).append(content)

    nodes = SentenceSplitter(
        chunk_size=LLAMA_INDEX_CHUNK_SIZE,
        chunk_overlap=LLAMA_INDEX_CHUNK_OVERLAP,
    ).get_nodes_from_documents(documents)

    # TextNode 继承 source Document 的 metadata（file_name/file_path/section）。
    # 这里保留来源信息：chunk_sources 与 chunks 等长，供 rag.ainsert(file_paths=...)；
    # 同时把章节/来源前缀注入 chunk 文本，让实体抽取 LLM 可见来源。
    chunks: list[str] = []
    chunk_sources: list[str] = []
    for idx, n in enumerate(nodes):
        content = n.get_content().strip()
        if not content:
            continue
        meta = getattr(n, "metadata", None) or {}
        file_path = str(meta.get("file_path", "") or "")
        section = str(meta.get("section", "") or "")
        file_name = str(meta.get("file_name", "") or "")

        prefix = _build_source_prefix(section=section, file_name=file_name)
        chunks.append(f"{prefix}{content}" if prefix else content)
        # 给每个 chunk 的来源加全局唯一后缀（::chunk-<node 索引>），规避 LightRAG 的
        # filename 去重：旧版 LightRAG 会按 file_path 判重，同一个 PDF 切出的多 chunk
        # 贴同一个文件名 → 一个 batch 里只有第 1 个进库，其余被标 DUPLICATE:filename 丢弃
        # （实测 16 个 chunk 的 batch 15 个被扔）。用 enumerate(nodes) 的全局序保证唯一，
        # 跳过空 node 不影响。Windows 文件名禁用 ':'、Linux 路径也不含 '::'，故 '::chunk-'
        # 不会与真实文件名冲突。检索端 _strip_source_suffix 会剥掉它还原真实文件名。
        base_src = file_path if file_path else "unknown_source"
        chunk_sources.append(f"{base_src}::chunk-{idx}")

    logger.info(
        "摄入解析完成: %d 个输入文件 → %d 个文档 → %d 个 chunk",
        len(file_paths),
        len(documents),
        len(chunks),
    )
    return chunks, chunk_sources, doc_texts


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
    all_sources: list[str] | None = None,
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
            "chunk_sources": all_sources if all_sources is not None else [],
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


async def _await_with_polling(
    task: asyncio.Task,
    poll_fn,  # Callable[[], Awaitable[None]] —— 每 poll_interval 秒调一次，可抛异常（如 IndexingAborted）中断等待
    poll_interval: float = 3.0,
) -> None:
    """等待 task 完成，期间定期调 poll_fn（借此检查暂停/终止信号）。

    asyncio 标准惯用法：wait_for 超时时默认会取消被等待的协程，这里用 shield 把
    task 包起来，让超时只中断"等待"、不取消 task 本身——这样 poll_fn 抛出后，取消
    与否的决定权留给调用方（见 _consumer 的 IndexingAborted 分支）。
    """
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=poll_interval)
        except asyncio.TimeoutError:
            await poll_fn()


async def ingest_to_lightrag(
    course_id: str,
    file_paths: list[str],
    batch_size: int = LIGHTRAG_INGEST_BATCH_SIZE,
    on_progress: ProgressCallback = None,
    resume_from_chunk: int = 0,
    control: Optional[IndexingControl] = None,
) -> dict:
    """
    完整摄入流水线：与 LlamaIndex 同策略解析切块 → LightRAG ainsert。

    支持生产者-消费者模式：图片提取与文本解析并行启动，
    文本 chunk 产出后立即开始分批写入 LightRAG。

    Args:
        course_id:         课程 ID
        file_paths:        待摄入文件列表
        batch_size:        每批插入 LightRAG 的 chunk 数
        on_progress:       异步进度回调
        resume_from_chunk: 断点续传：从第 N 个 chunk 开始（跳过前 N 个）
        control:           可选的 IndexingControl，用于在 batch 边界中止任务
    """
    from core.rag.lightrag import (
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

    _t0 = time.perf_counter()
    log_flow("index.start", course_id=course_id, files=len(file_paths),
             resume_from_chunk=resume_from_chunk)
    ok, reason = is_lightrag_available()
    if not ok:
        raise RuntimeError(f"LightRAG 不可用: {reason}")

    clear_llm_errors()

    # Step 1: 解析文件（CPU 密集，放线程池）——同时返回 doc_texts 供图片阶段复用
    is_resume = resume_from_chunk > 0
    parse_label = f"续传解析 {len(file_paths)} 个文件（将跳过前 {resume_from_chunk} 个文本块）…" \
        if is_resume else f"开始解析 {len(file_paths)} 个文件…"
    await _emit(5, parse_label, resume_from_chunk, 0, 0)
    logger.info("开始解析 %d 个文件 course=%s resume_from=%d", len(file_paths), course_id, resume_from_chunk)
    all_chunks, all_sources, doc_texts = await asyncio.to_thread(parse_files, file_paths)

    # Step 1b: 图片摄入（复用 doc_texts，不再重复提取文本）
    images_processed = 0
    try:
        from core.rag.llamaindex.image_extractor import ingest_images_from_files

        rag_for_images = await _get_instance(course_id)
        img_cache = _lightrag_ingest_chunks_dir(course_id) / "image_desc_cache.json"
        await _emit(
            8,
            f"开始提取文档中的图片并写入知识图谱（{len(file_paths)} 个源文件）…",
            resume_from_chunk,
            max(len(all_chunks), 1),
            0,
        )

        async def _on_image_progress(done: int, total: int) -> None:
            if total <= 0:
                return
            pct = 8 + int(done / total * 2)
            await _emit(
                pct,
                f"图片知识图谱：{done}/{total} 张",
                resume_from_chunk,
                max(len(all_chunks), total),
                0,
            )

        images_processed = await ingest_images_from_files(
            file_paths,
            rag_for_images,
            cache_path=str(img_cache),
            doc_texts=doc_texts,
            on_image_done=_on_image_progress,
            control=control,
        )
        if images_processed:
            logger.info(
                "图片知识图谱摄入 course=%s images=%d",
                course_id,
                images_processed,
            )
    except ImportError:
        logger.warning("raganything 未安装，跳过图片知识图谱摄入 course=%s", course_id)
    except IndexingAborted:
        raise
    except Exception as exc:
        logger.warning(
            "图片知识图谱摄入失败（继续文本索引）course=%s: %s",
            course_id,
            exc,
            exc_info=True,
        )

    if not all_chunks and images_processed == 0:
        logger.warning("解析结果为空 course=%s", course_id)
        await _emit(100, "解析结果为空，无可索引内容", 0, 0, 0)
        return {"status": "empty", "chunks": 0, "files": len(file_paths), "images": 0}

    await asyncio.to_thread(
        _persist_lightrag_ingest_chunks,
        course_id,
        file_paths,
        all_chunks,
        resume_from_chunk,
        all_sources,
    )

    total = len(all_chunks)
    avg_chars = sum(len(c) for c in all_chunks) / total
    token_per_chunk = int(_TOKEN_OVERHEAD_PER_CHUNK + avg_chars / 3.5)

    start = min(resume_from_chunk, total)
    chunks = all_chunks[start:]
    sources = all_sources[start:]  # 与 chunks 同步偏移，保持一一配对
    already_done = start

    resume_note = f"（已跳过 {already_done} 个，续传）" if is_resume else ""
    await _emit(
        10,
        f"解析完成：{len(file_paths)} 个文件 → {total} 个文本块（均长 {int(avg_chars)} 字符）{resume_note}",
        already_done, total, already_done * token_per_chunk,
    )

    if not chunks:
        img_note = f"，{images_processed} 张图片已写入知识图谱" if images_processed else ""
        await _emit(
            100,
            f"所有文本块均已索引完毕{img_note}",
            total,
            total,
            total * token_per_chunk,
        )
        return {
            "status": "done",
            "chunks": total,
            "files": len(file_paths),
            "images": images_processed,
        }

    # Step 2: 生产者-消费者模式写入 LightRAG
    # 生产者把 chunk 按 batch_size 分组放入队列，消费者从队列取出并 ainsert。
    # 当前 parse_files 已完成（同步），生产者只是快速切分；
    # 真正的并行收益来自：消费者 ainsert 某批时，下一批已在队列中就绪，
    # 且 LightRAG 内部的 max_async 可以跨批次流水线化 LLM 调用。
    logger.info(
        "开始写入 LightRAG: %d 个 chunk（跳过 %d），batch_size=%d，course=%s",
        len(chunks), already_done, batch_size, course_id,
    )
    rag = await _get_instance(course_id)

    _QUEUE_MAXSIZE = 3
    # 队列元素：(batch_chunks, batch_sources)，两者等长配对；None 为哨兵
    chunk_queue: asyncio.Queue[tuple[list[str], list[str]] | None] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)

    async def _producer() -> None:
        """将 chunks/sources 按 batch_size 同步切分后放入队列，结束时放入 None 作为哨兵。"""
        try:
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_sources = sources[i : i + batch_size]
                await chunk_queue.put((batch_chunks, batch_sources))
        finally:
            await chunk_queue.put(None)

    async def _consumer() -> None:
        """从队列取出 batch 写入 LightRAG；batch 边界响应暂停/终止信号 + 错误检测。"""
        nonlocal already_done
        consumed = 0
        while True:
            item = await chunk_queue.get()
            if item is None:
                break
            batch_chunks, batch_sources = item

            batch_start_idx = consumed
            consumed += len(batch_chunks)

            # ainsert 是原子的：一个 batch 要么整批写入要么全不计入。因此 checkpoint 的进度
            # 恒记"本 batch 起点前"的已完成数（already_done + batch_start_idx），续传时从该
            # 起点重跑整个 batch——不重不漏。control 为 None 时 _checkpoint 退化为 no-op，
            # _await_with_polling 仍正常等待 task 完成（仅每 3s 多一次空轮询，可忽略）。
            done_at_batch_start = already_done + batch_start_idx

            async def _checkpoint() -> None:
                if control is not None:
                    await control.checkpoint(chunks_done=done_at_batch_start)
                # 实时回写 LightRAG 内部进度。一个 batch 的 ainsert 内部要逐 chunk 调 LLM 做
                # 实体/关系抽取（每 chunk 一个 "Extracting stage"，常达数分钟）；若不回写，这段
                # 时间前端只能看到上一帧"解析完成 10%"，表现为卡死。每 3s 从 pipeline_status 读
                # cur_batch/batchs（LightRAG 按 workspace 隔离的 multiprocessing.Manager 共享 dict，
                # worker 进程内可直接读；web 进程读不到，故必须由 worker 写 DB）换算成全局进度。
                # 读取失败绝不影响停止信号检查与 ainsert 本身。
                try:
                    from lightrag.kg.shared_storage import get_namespace_data
                    ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
                    if ps.get("busy"):
                        cur_batch = int(ps.get("cur_batch") or 0)
                        batchs = int(ps.get("batchs") or 0)
                        live_done = done_at_batch_start + cur_batch
                        live_progress = 10 + int(live_done / total * 85) if total else 10
                        live_msg = f"构建知识图谱：实体抽取 {live_done}/{total}"
                        if batchs:
                            live_msg += f"（本批 {cur_batch}/{batchs}）"
                        await _emit(
                            live_progress, live_msg, live_done, total,
                            live_done * token_per_chunk,
                        )
                except Exception:
                    logger.debug("读取 LightRAG 实时进度失败 course=%s", course_id, exc_info=True)

            # 起点先查一次信号（若已暂停则不启动本 batch），再异步跑 + 每 3 秒轮询
            await _checkpoint()
            _insert_task = asyncio.create_task(
                rag.ainsert(batch_chunks, file_paths=batch_sources)
            )
            try:
                await _await_with_polling(_insert_task, _checkpoint, poll_interval=3.0)
            except IndexingAborted:
                _insert_task.cancel()
                # LightRAG ainsert 对 cancel 响应不一致：给至多 3s 收尾，超时则放弃等待。
                # 孤儿 task 至多把当前 batch 跑完即结束，_consumer 已退出不再消费后续 batch
                # ——这是"跨进程取消一个不可取消的第三方调用"的务实妥协。
                try:
                    await asyncio.wait_for(_insert_task, timeout=3.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
                raise
            except Exception:
                _insert_task.cancel()
                raise

            errors = take_llm_errors()
            if errors:
                fatal = [e for e in errors if _is_fatal_llm_error(e)]
                if fatal:
                    err_msg = str(fatal[0])
                    done_so_far = already_done + consumed
                    await _emit(
                        int(10 + done_so_far / total * 85),
                        f"遇到致命错误，索引中止（已完成 {done_so_far}/{total} 个文本块）",
                        done_so_far, total, done_so_far * token_per_chunk,
                    )
                    raise RuntimeError(f"LLM API 致命错误，索引中止: {err_msg[:300]}")
                else:
                    logger.warning("非致命 LLM 错误（继续）: %s", errors[0])

            done = already_done + consumed
            progress = 10 + int(done / total * 85)
            token_estimate = done * token_per_chunk
            await _emit(
                progress,
                f"构建知识图谱：{done}/{total} 个文本块",
                done, total, token_estimate,
            )
            logger.info("LightRAG 摄入进度 course=%s %d/%d", course_id, done, total)

    producer_task = asyncio.create_task(_producer())
    try:
        await _consumer()
    except (IndexingAborted, RuntimeError):
        producer_task.cancel()
        try:
            await producer_task
        except (asyncio.CancelledError, Exception):
            pass
        raise
    await producer_task

    final_tokens = total * token_per_chunk
    img_note = f"，{images_processed} 张图片" if images_processed else ""
    await _emit(
        100,
        f"索引完成：{len(file_paths)} 个文件，{total} 个文本块{img_note}，估算消耗 {final_tokens:,} tokens",
        total, total, final_tokens,
    )
    logger.info(
        "摄入完成 course=%s files=%d chunks=%d images=%d",
        course_id,
        len(file_paths),
        total,
        images_processed,
    )
    log_flow("index.complete", course_id=course_id,
             elapsed_ms=int((time.perf_counter() - _t0) * 1000),
             chunks=total, files=len(file_paths), images=images_processed)
    return {
        "status": "done",
        "chunks": total,
        "files": len(file_paths),
        "images": images_processed,
    }


def llama_available() -> bool:
    """返回 LlamaIndex 是否可用（供 API 展示）。"""
    return True
