"""LightRAG 摄入流水线（管理端「开始索引」）。

解析与切块与 rag_llama/indexing_documents + rag_llama/llamaindex_pipeline 共用；
LightRAG 仅负责 ainsert。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from settings import get_settings
INGEST_CHUNK_OVERLAP = get_settings().chunking.ingest_overlap
INGEST_CHUNK_SIZE = get_settings().chunking.ingest_size
LIGHTRAG_INGEST_BATCH_SIZE = get_settings().lightrag.ingest_batch_size
LIGHTRAG_INGEST_CHUNKS_SNAPSHOT = get_settings().lightrag.ingest_chunks_snapshot
LIGHTRAG_INGEST_CHUNKS_SUBDIR = get_settings().lightrag.ingest_chunks_subdir
LIGHTRAG_SAVE_INGEST_CHUNKS = get_settings().lightrag.save_ingest_chunks
LIGHTRAG_WORKDIR = get_settings().paths.lightrag_workdir
from core.rag.chunking.registry import get_chunk_strategy, register_chunk_strategy
from core.rag.llamaindex.indexing_documents import file_paths_to_llama_documents
from core.rag.llamaindex.file_routing import FileClassification
from core.rag.source_utils import strip_chunk_suffix

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
    from llama_index.core.schema import Document as _LIDoc
    logger.info("LlamaIndex 可用，将使用智能文档解析")
except ImportError as e:
    raise ImportError(
        "llama-index-core 未安装，请执行: pip install llama-index-core"
    ) from e


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


def _build_source_prefix(section: str = "", file_name: str = "", page: int = 0) -> str:
    """构建来源前缀，注入到 chunk 文本开头，供 LightRAG 实体抽取 LLM 可见章节/文件来源。

    DOCX/PPTX 有 section → 注入章节名；PDF/TXT/MD 无 section → 退化用文件名。
    page>0（PDF 独有页码）时追加 | 第N页。page 默认 0 → 现有 docx/sentence_splitter
    调用不传 page，输出零变化（向后兼容）。
    """
    parts: list[str] = []
    if section:
        parts.append(f"章节: {section}")
    elif file_name:
        parts.append(f"来源: {file_name}")
    if page > 0:
        parts.append(f"第{page}页")
    if not parts:
        return ""
    return f"【{' | '.join(parts)}】\n"


# ── 切块策略分发（可插拔）─────────────────────────────────────────────────────


def _chunk_by_sentence_splitter(documents: list) -> tuple[list[str], list[str]]:
    """默认切块策略：LlamaIndex SentenceSplitter → chunks + sources。

    SentenceSplitter 的 chunk_size 默认按 tiktoken **token** 计数
    （``_token_size = len(self._tokenizer(text))``，默认 tokenizer 是 cl100k_base），
    与 ``.env`` 的 ``INGEST_SIZE``「字符数」直觉不符——传 ``tokenizer=lambda t: t``
    让其按**字符**计数，chunk_size/overlap 真正以字符为单位（对齐 .env.example 注释
    「LlamaIndex 按字数切」的本意）。

    PDF/MinerU markdown 正文含内嵌 HTML ``<table>``：按表格边界拆段，正文走
    SentenceSplitter，表格原子化（``_atomize_table``），避免数值从单元格内部被锯断
    （实测 AutoAct 论文 Table 1 被切成 4 块、``49.09`` 断成 ``49.``/``09``）。
    每个 chunk 注入【章节/来源】前缀，source 加 ``::chunk-<idx>`` 全局唯一后缀，
    顺序保留文档内 text/table 交错次序。
    """
    splitter = SentenceSplitter(
        chunk_size=INGEST_CHUNK_SIZE,
        chunk_overlap=INGEST_CHUNK_OVERLAP,
        # 按字符计数：默认 tiktoken token 计数会让 chunk 实际字符数远超 INGEST_SIZE
        tokenizer=lambda t: t,
    )

    chunks: list[str] = []
    chunk_sources: list[str] = []
    # 给每个 chunk 的来源加全局唯一后缀（::chunk-<node 索引>），规避 LightRAG 的
    # filename 去重：同文件多 chunk 贴同一 file_path 会被标 DUPLICATE:filename 丢弃。
    # Windows 文件名禁用 ':'、Linux 路径不含 '::'，故不与真实文件名冲突；
    # 检索端 strip_chunk_suffix 会剥掉它还原真实文件名。
    for doc in documents:
        meta = getattr(doc, "metadata", None) or {}
        file_path = str(meta.get("file_path", "") or "")
        section = str(meta.get("section", "") or "")
        file_name = str(meta.get("file_name", "") or "")
        page = int(meta.get("page", 0) or 0)
        prefix = _build_source_prefix(section=section, file_name=file_name, page=page)
        base_src = file_path if file_path else "unknown_source"

        for seg, is_table in _split_markdown_by_tables(doc.get_content()):
            if is_table:
                pieces = _atomize_table(seg, INGEST_CHUNK_SIZE)
            elif seg.strip():
                pieces = [
                    n.get_content()
                    for n in splitter.get_nodes_from_documents(
                        [_LIDoc(text=seg, metadata=meta)]
                    )
                ]
            else:
                continue
            for content in pieces:
                content = content.strip()
                if not content:
                    continue
                chunks.append(f"{prefix}{content}" if prefix else content)
                chunk_sources.append(f"{base_src}::chunk-{len(chunks) - 1}")
    return chunks, chunk_sources


# ── PDF markdown 表格原子化 ───────────────────────────────────────────────────
# MinerU markdown 表格是内嵌 HTML ``<table><tr><td rowspan/colspan>…</td></tr></table>``
# （实测 AutoAct 论文 7 张表全 HTML、0 markdown 表）。SentenceSplitter 把它当纯文本
# 会从单元格内部锯断数值，故按 ``<table>`` 边界拆段，表格整块（与 DOCX 路径
# ``_flush_table``+``serialize_table`` 同理）。
_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL)
_TR_RE = re.compile(r"<tr[^>]*>.*?</tr>", re.DOTALL)


def _split_markdown_by_tables(text: str) -> list[tuple[str, bool]]:
    """按 ``<table>...</table>`` 边界把 markdown 拆成有序 ``[(段, 是否表格), ...]``。

    表格外的正文段 ``is_table=False``；表格段 ``is_table=True``（含完整 ``<table>`` 标签）。
    不匹配任何表格时返回整段正文（``[(text, False)]``），行为等价原直通。
    """
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in _TABLE_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((m.group(0), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def _atomize_table(table_html: str, max_chars: int) -> list[str]:
    """表格原子化：整表一块；超过 ``max_chars`` 才按 ``<tr>`` 边界分组，每组前置首行表头。

    绝不在 ``<tr>`` 中间切（保证单元格数值完整）。单行即超阈值时整行保留
    （无法再切，宁超不断）。多行表头（rowspan/colspan 跨多 ``<tr>``）只重复**首个**
    ``<tr>``——优先级是「不断单元格」，多行表头可读性次之（已知近似）。
    """
    if max_chars <= 0 or len(table_html) <= max_chars:
        return [table_html]
    rows = _TR_RE.findall(table_html)
    if len(rows) <= 1:  # 无 body 行可分组，整表保留
        return [table_html]
    header = rows[0]
    groups: list[str] = []
    cur = f"<table>{header}"
    head_only = f"<table>{header}"
    for row in rows[1:]:
        # 加入本行就超阈值、且当前组已有 body 行 → 落盘当前组、起新组（带表头）
        if len(cur) + len(row) + len("</table>") > max_chars and cur != head_only:
            groups.append(f"{cur}</table>")
            cur = head_only
        cur += row
    groups.append(f"{cur}</table>")
    return groups


def _chunk_documents(
    documents: list,
    classification: FileClassification,
    strategy: str,
) -> tuple[list[str], list[str]]:
    """切块策略分发（settings.chunking.strategy 扩展点）。

    查 chunking registry 注册表分发（与 core/rag/registry.py 的检索器后端注册表同构）：
    get_chunk_strategy(strategy) 取策略函数，以统一签名
    (documents, classification, INGEST_CHUNK_SIZE) 调用。新增策略只需
    register_chunk_strategy 一行，无需改本函数的分发代码（消除原先硬编码的 if/else）。

    产出的 chunks/chunk_sources 格式与默认策略完全一致（前缀 + ::chunk-<idx>），
    下游 _ingest_body 的 rag.ainsert(file_paths=...) 零改动。
    """
    return get_chunk_strategy(strategy)(documents, classification, INGEST_CHUNK_SIZE)


def _chunk_sentence_splitter_strategy(
    documents: list, classification: FileClassification, ingest_size: int
) -> tuple[list[str], list[str]]:
    """默认策略：全部 documents 走 _chunk_by_sentence_splitter（忽略 classification/ingest_size）。"""
    return _chunk_by_sentence_splitter(documents)


def _chunk_ragflow_manual_strategy(
    documents: list, classification: FileClassification, ingest_size: int
) -> tuple[list[str], list[str]]:
    """ragflow_manual_docx 策略：DOCX 走 chunk_docx_structured（移植 RAGFlow Manual：
    标题层级栈 + 表格原子化），非 DOCX documents 回退 _chunk_by_sentence_splitter。
    混合策略——只对 DOCX 生效，PDF/TXT/PPTX 行为不变。"""
    from core.rag.chunking.ragflow_manual_docx import chunk_docx_structured

    # documents 里哪些来自 DOCX：用 resolved 绝对路径匹配（indexing_documents 存的
    # file_path 是 resolved 绝对路径，classification.docx_files 是原始字符串）
    docx_resolved = {str(Path(fp).resolve()) for fp in classification.docx_files}
    docx_doc_indices = {
        i for i, d in enumerate(documents)
        if str(d.metadata.get("file_path", "")) in docx_resolved
    }

    chunks: list[str] = []
    chunk_sources: list[str] = []

    # DOCX：结构化切块（表格原子化 + 标题层级栈）
    for fp in classification.docx_files:
        ck_list, sec_list = chunk_docx_structured(fp, max_section_chars=ingest_size)
        file_name = Path(fp).name
        resolved_fp = str(Path(fp).resolve())
        for i, (ck, sec) in enumerate(zip(ck_list, sec_list)):
            prefix = _build_source_prefix(section=sec, file_name=file_name)
            chunks.append(f"{prefix}{ck}" if prefix else ck)
            # 同文件内 i 唯一 + resolved file_path 前缀 → 全局唯一，与默认策略等价
            chunk_sources.append(f"{resolved_fp}::chunk-{i}")

    # 非 DOCX：原 SentenceSplitter（排除 DOCX documents，避免 DOCX 被切两次）
    non_docx_docs = [d for i, d in enumerate(documents) if i not in docx_doc_indices]
    if non_docx_docs:
        nd_chunks, nd_sources = _chunk_by_sentence_splitter(non_docx_docs)
        chunks.extend(nd_chunks)
        chunk_sources.extend(nd_sources)

    return chunks, chunk_sources


# 注册切块策略（模块加载时执行；新增策略在此加一行即可，_chunk_documents 分发结构不变）
register_chunk_strategy("sentence_splitter", _chunk_sentence_splitter_strategy)
register_chunk_strategy("ragflow_manual_docx", _chunk_ragflow_manual_strategy)


# ── 核心解析函数 ─────────────────────────────────────────────────────────────

def parse_files(file_paths: list[str]) -> tuple[list[str], list[str], dict[str, list[str]], list[str]]:
    """
    解析文件列表，返回 (文本 chunk 列表, chunk 来源路径列表, 按文件分段的全文 dict, 解析失败原因列表)。

    chunk_sources 与 chunks 等长，第 i 个元素 = 第 i 个 chunk 的来源 file_path，
    供 ingest_to_lightrag 传给 rag.ainsert(file_paths=...) 做来源溯源。
    每个 chunk 文本开头已注入【章节: xxx】/【来源: filename】前缀，供实体抽取 LLM 可见来源。

    doc_texts 供图片 VLM 摄入复用，避免重复提取：
      dict[绝对路径, list[str]]  — PDF 每页 / DOCX 每 section / PPTX 每 slide / 纯文本单元素
    """
    if not file_paths:
        return [], [], {}, []

    documents, classification, parse_errors = file_paths_to_llama_documents(
        file_paths, log=logger
    )

    if not documents:
        logger.warning("摄入解析结果为空（无有效文档）")
        return [], [], {}, parse_errors

    # 构建 doc_texts：按文件路径分组，复用已解析的 Document 内容
    doc_texts: dict[str, list[str]] = {}
    for doc in documents:
        fp = doc.metadata.get("file_path", "")
        if not fp:
            continue
        content = doc.get_content().strip()
        if content:
            doc_texts.setdefault(fp, []).append(content)

    # 切块策略分发（默认 sentence_splitter；ragflow_manual_docx 时 DOCX 走结构化切块，
    # 非 DOCX 仍走 sentence_splitter）。详见 _chunk_documents。
    strategy = get_settings().chunking.strategy
    chunks, chunk_sources = _chunk_documents(documents, classification, strategy)

    logger.info(
        "摄入解析完成: %d 个输入文件 → %d 个文档 → %d 个 chunk（strategy=%s）",
        len(file_paths),
        len(documents),
        len(chunks),
        strategy,
    )
    return chunks, chunk_sources, doc_texts, parse_errors


# ── Phase 2: Contextual Chunking 接入辅助 ────────────────────────────────────

def _load_contextual_cache(path: Path) -> dict[str, str]:
    """加载磁盘缓存 {chunk_hash: enriched}；损坏/缺失返回空 dict。"""
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, json.JSONDecodeError):
        logger.debug("contextual cache 读取失败，重新开始: %s", path)
    return {}


def _save_contextual_cache(path: Path, cache: dict[str, str]) -> None:
    """持久化缓存，供重复索引复用（避免对同一 chunk 重复调 LLM）。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning("contextual cache 保存失败: %s", e)


async def _apply_contextual_enrichment(
    course_id: str,
    chunks: list[str],
    sources: list[str],
    doc_texts: dict[str, list[str]],
    *,
    model: str = "",
) -> list[str]:
    """对 chunks 做文档背景前缀注入，返回等长 enriched 列表（绝不抛异常）。

    自门控：``chunking.contextual_enrichment`` 关闭或无 chunk 时直接返回原文，调用方无需
    再包 gate + try/except。整体异常也降级返回原文 chunks（单 chunk LLM 失败已在更内层降级）。
    按来源文件分组：同一文件的 chunks 共享该文件全文作 document_text。磁盘 cache 跨索引复用。
    """
    if not chunks:
        return chunks
    _chunk_cfg = get_settings().chunking
    if not _chunk_cfg.contextual_enrichment:
        return chunks
    try:
        from core.rag.contextual_chunking import contextualize_chunks, summarize_document
        from core.llm.llm import chat_complete

        fast_model = (
            model or _chunk_cfg.contextual_model
            or get_settings().llm.fast_model or get_settings().llm.text_model
        )

        async def _llm_func(prompt: str) -> str:
            return await chat_complete(
                system_prompt="",
                history=[],
                user_message=prompt,
                model=fast_model,
                temperature=0.3,
                max_tokens=200,
            )

        cache_path = _lightrag_ingest_chunks_dir(course_id) / "contextual_cache.json"
        cache = _load_contextual_cache(cache_path)

        # 按来源文件分组 chunk 索引，每组共享对应文件全文
        groups: dict[str, list[int]] = {}
        for i, src in enumerate(sources):
            groups.setdefault(strip_chunk_suffix(src), []).append(i)

        enriched = list(chunks)
        for fp, idxs in groups.items():
            doc_parts = doc_texts.get(fp) or []
            document_text = "\n\n".join(doc_parts) if isinstance(doc_parts, list) else str(doc_parts)
            group_chunks = [chunks[i] for i in idxs]
            # 文档级摘要：每篇文档 1 次 LLM，本组所有 chunk 共享（对标 dsRAG AutoContext）。
            # 摘要按 md5(document_text) 缓存，重复索引不重复调用。
            doc_summary = await summarize_document(document_text, _llm_func, cache=cache)
            enc = await contextualize_chunks(
                group_chunks, document_text, _llm_func, cache=cache, doc_summary=doc_summary,
            )
            for i, text in zip(idxs, enc):
                enriched[i] = text

        _save_contextual_cache(cache_path, cache)
        logger.info(
            "Contextual enrichment 完成 course=%s files=%d chunks=%d",
            course_id, len(groups), len(chunks),
        )
        return enriched
    except Exception as exc:
        logger.warning(
            "Contextual enrichment 整体失败，降级原文 chunks course=%s: %s", course_id, exc,
        )
        return chunks


async def _index_batch_to_es(es_store, course_id: str, chunks: list[str], sources: list[str]) -> None:
    """Phase 3: 把一批 chunk 写入 ES BM25 索引（chunk_id=content hash，作两系统 join key）。

    失败仅告警不阻断 LightRAG 索引（ES 是增强路径，不可用则检索降级纯 dense）。
    """
    docs = [
        {
            "chunk_id": hashlib.md5(text.encode("utf-8")).hexdigest(),
            "content": text,
            "file_path": strip_chunk_suffix(src),
        }
        for text, src in zip(chunks, sources)
        if text and text.strip()
    ]
    if docs:
        ok = await es_store.index_chunks(docs, course_id=course_id)
        if not ok:
            logger.warning("ES 双写失败（不影响 LightRAG 索引）course=%s batch=%d", course_id, len(docs))


# ── 完整摄入流水线 ───────────────────────────────────────────────────────────

ProgressCallback = Optional[Callable[..., Awaitable[None]]]


def _lightrag_ingest_chunks_dir(course_id: str) -> Path:
    """lightrag_store/course_{course_id}/ingest_chunks/（与 core.lightrag_engine workspace 命名一致）。"""
    return Path(LIGHTRAG_WORKDIR) / f"course_{course_id}" / LIGHTRAG_INGEST_CHUNKS_SUBDIR


def persist_ingest_chunks(
    course_id: str,
    file_paths: list[str],
    all_chunks: list[str],
    resume_from_chunk: int,
    all_sources: list[str] | None = None,
    *,
    backend: str = "lightrag",
    node_ids: list[str] | None = None,
) -> Path | None:
    """
    将摄入前切分好的文本块写入审计 JSON（供排查/审计；不参与检索加载）。

    两个后端共用一个 writer——审计的价值正在于对照同一份 ``parse_files`` 输出在
    LightRAG / pgvector 两边吃到的 chunk 是否完全一致（schema 分裂就失去这个作用）：

    - ``backend="lightrag"``：写 ``lightrag_store/course_{id}/ingest_chunks/``（含可选
      snapshot），gate 读 ``lightrag.save_ingest_chunks``；行为与历史一致，仅 payload
      多 ``backend`` 字段、``version`` 升 2。
    - ``backend="llamaindex_pg"``：写独立根 ``data/ingest_chunks/course_{id}/``（无
      snapshot），gate 读 ``chunking.save_pg_ingest_chunks``；额外记 ``node_ids``
      （= ``data_kb_chunks.node_id`` 主键），把审计文本直接 join 回 PG 行。

    返回写入的 latest.json 路径；失败时记日志并返回 None。
    """
    settings = get_settings()
    if backend == "lightrag":
        if not LIGHTRAG_SAVE_INGEST_CHUNKS:
            return None
        out_dir = _lightrag_ingest_chunks_dir(course_id)
        snapshot = LIGHTRAG_INGEST_CHUNKS_SNAPSHOT
    elif backend == "llamaindex_pg":
        if not settings.chunking.save_pg_ingest_chunks:
            return None
        out_dir = Path(settings.paths.ingest_chunks_dir) / f"course_{course_id}"
        snapshot = False
    else:
        logger.warning("persist_ingest_chunks 未知 backend=%s，跳过", backend)
        return None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "version": 2,
            "backend": backend,
            "course_id": course_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "source_files": [str(Path(p).name) for p in file_paths],
            "source_paths": [str(Path(p).resolve()) for p in file_paths],
            "chunk_count": len(all_chunks),
            "resume_from_chunk_at_save": resume_from_chunk,
            "chunks": all_chunks,
            "chunk_sources": all_sources if all_sources is not None else [],
        }
        if node_ids is not None:
            payload["node_ids"] = node_ids
        latest = out_dir / "latest.json"
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        latest.write_text(text, encoding="utf-8")
        if snapshot:
            ts_name = f"chunks_{int(datetime.now(timezone.utc).timestamp())}.json"
            (out_dir / ts_name).write_text(text, encoding="utf-8")
        logger.info(
            "已保存摄入切块 backend=%s course=%s dir=%s chunks=%d",
            backend,
            course_id,
            out_dir,
            len(all_chunks),
        )
        return latest
    except OSError as e:
        logger.warning("保存摄入切块失败 backend=%s course=%s: %s", backend, course_id, e)
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
    # 可用性先校验，避免无谓地 lease 一个实例
    from core.rag.lightrag import is_lightrag_available, lease_instance

    ok, reason = is_lightrag_available()
    if not ok:
        raise RuntimeError(f"LightRAG 不可用: {reason}")

    # H-10：整个摄入期间持有一个实例引用（lease），离开 with 块自动释放计数。
    # 图片摄入(Step 1b)与文本摄入(Step 2)共用同一 rag，避免重复 +1。
    async with lease_instance(course_id) as rag:
        return await _ingest_body(
            course_id, file_paths, batch_size, on_progress,
            resume_from_chunk, control, rag,
        )


# DOCX 切块阶段埋的图片位置占位符：[[IMG:原图blob sha256 前16位]]
_IMG_PLACEHOLDER_RE = re.compile(r"\[\[IMG:([0-9a-f]{16})\]\]")


def _resolve_image_placeholders(
    chunks: list[str],
    img_cache: Path,
    *,
    fill: bool = True,
) -> tuple[list[str], set[str]]:
    """把 chunk 里的 [[IMG:sha16]] 占位符按位置清单回填成 [图: desc]。

    位置清单 image_desc_by_blob.json 与 image_desc_cache.json 同目录，key 为原图 blob
    sha256 全量（占位符取其前 16 位）。清单里查不到的占位符降级为空串（移除），不再制造
    "[图: 电路图]" 噪声。

    fill=False（``chunking.inline_image_descriptions`` 关）时只清理占位符、不回填描述，
    返回空 inlined 集合——因为 LightRAG 路径的 KG 摄入总会产出清单，需用 fill 门控避免
    描述在开关关时仍漏进正文。

    Returns:
        (回填后的 chunks, 已成功回填的 sha16 集合)。后者供 _append_image_desc_chunks
        去重——已内联进正文的图不再追加孤儿 chunk（GPT-RAG：figures never duplicated）。
        位置清单缺失/损坏 → 占位符全部降级为空串、返回空集，不抛异常。
    """
    inlined: set[str] = set()
    if not fill:
        return [_IMG_PLACEHOLDER_RE.sub("", c) for c in chunks], inlined

    from core.rag.llamaindex.image_extractor import (
        _blob_manifest_path, _load_blob_manifest,
    )

    manifest = _load_blob_manifest(_blob_manifest_path(img_cache))
    by16: dict[str, str] = {
        k[:16]: v["desc"] for k, v in manifest.items() if v.get("desc")
    }

    def _replace(match: "re.Match[str]") -> str:
        sha16 = match.group(1)
        desc = by16.get(sha16)
        if desc:
            inlined.add(sha16)
            return f"[图: {desc}]"
        return ""  # 清单无此图（如公式碎片被过滤）→ 移除占位符，不造噪声

    resolved = [_IMG_PLACEHOLDER_RE.sub(_replace, c) for c in chunks]
    return resolved, inlined


def _append_image_desc_chunks(
    all_chunks: list[str],
    all_sources: list[str],
    img_cache: Path,
    inlined: set[str] | None = None,
) -> int:
    """Phase 4：把未内联进正文的图片描述作为独立 chunk 追加（带来源前缀、去重）。

    读位置清单 image_desc_by_blob.json（key=原图 blob sha256, value={desc, source}）。
    凡已通过 [[IMG:sha16]] 占位符内联进正文（sha16 ∈ inlined）的图，不再重复追加孤儿
    chunk（GPT-RAG：figures never duplicated）；其余（页眉/页脚图、PDF 图等正文无占位符
    的）追加为带【来源: 文件名】前缀的独立 chunk——前缀在 chunk 文本里，LLM 可见、不再悬空。
    source 用 `{file_path}::image-{sha16}`（真实路径，可溯源/可过滤，替原有的假 image_desc::img-N）。

    chunks/sources 严格配对 append，返回追加条数。
    位置清单缺失/空/读失败 → 降级返回 0，不抛异常（绝不阻断索引）。
    """
    try:
        from core.rag.llamaindex.image_extractor import (
            _blob_manifest_path, _load_blob_manifest,
        )

        manifest = _load_blob_manifest(_blob_manifest_path(img_cache))
    except Exception as exc:
        logger.warning("图片描述回填失败（降级跳过）: %s", exc)
        return 0

    skip = inlined or set()
    added = 0
    for blob_sha, entry in manifest.items():
        desc = (entry.get("desc") or "").strip()
        if not desc:
            continue
        sha16 = blob_sha[:16]
        if sha16 in skip:
            continue  # 已内联进正文，不重复追加
        source_path = entry.get("source") or ""
        file_name = Path(source_path).name if source_path else ""
        prefix = _build_source_prefix(file_name=file_name)
        all_chunks.append(f"{prefix}【图片描述】\n{desc}")
        src = (
            f"{source_path}::image-{sha16}"
            if source_path
            else f"image_desc::image-{sha16}"
        )
        all_sources.append(src)
        added += 1
    return added


async def _ingest_body(
    course_id: str,
    file_paths: list[str],
    batch_size: int,
    on_progress: ProgressCallback,
    resume_from_chunk: int,
    control: Optional[IndexingControl],
    rag,
) -> dict:
    """ingest_to_lightrag 的函数体（lease_instance 已保证引用计数配对）。

    拆出本函数仅为让 lease_instance 的 async with 能干净包裹整个摄入流程；
    rag 由调用方注入，本函数不再自行 _get_instance。
    """
    from core.rag.lightrag import (
        is_lightrag_available,
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
    # 可用性已在调用方（ingest_to_lightrag）校验过，但防御性再查一次
    ok, reason = is_lightrag_available()
    if not ok:
        raise RuntimeError(f"LightRAG 不可用: {reason}")

    # 索引开始前清空 LLM 错误缓冲（避免上一轮残留误判为致命错误）
    clear_llm_errors()

    # Step 1: 解析文件（CPU 密集，放线程池）——同时返回 doc_texts 供图片阶段复用
    is_resume = resume_from_chunk > 0
    parse_label = f"续传解析 {len(file_paths)} 个文件（将跳过前 {resume_from_chunk} 个文本块）…" \
        if is_resume else f"开始解析 {len(file_paths)} 个文件…"
    await _emit(5, parse_label, resume_from_chunk, 0, 0)
    logger.info("开始解析 %d 个文件 course=%s resume_from=%d", len(file_paths), course_id, resume_from_chunk)
    all_chunks, all_sources, doc_texts, parse_errors = await asyncio.to_thread(parse_files, file_paths)

    # Phase 2: Contextual Chunking（可选）——给每个 chunk 注入文档背景前缀，提升
    # embedding/BM25 命中率。自门控 + 整体降级已下沉到 _apply_contextual_enrichment 内部，
    # 开关默认关（需 .env 显式开启）。_chunk_cfg 后续图片摄入也用。
    _chunk_cfg = get_settings().chunking
    all_chunks = await _apply_contextual_enrichment(
        course_id, all_chunks, all_sources, doc_texts,
    )

    # Step 1b: 图片摄入（复用 doc_texts，不再重复提取文本）
    # rag 由调用方通过 lease_instance 注入，Step 1b 与 Step 2 共用同一实例（H-10）。
    images_processed = 0
    img_cache = _lightrag_ingest_chunks_dir(course_id) / "image_desc_cache.json"

    # M-28：断点续传时图片在首次索引已写入知识图谱（raganything 按 entity_name
    # ainsert，重复跑会产生重复实体/边）。续传场景下跳过图片摄入，仅继续文本 chunk。
    # 判据：resume_from_chunk>0 即为续传（与文本续传同源触发）。
    skip_images_on_resume = resume_from_chunk > 0
    if skip_images_on_resume:
        logger.info(
            "续传模式：跳过图片知识图谱重摄入（首次索引已写入）course=%s",
            course_id,
        )

    if not skip_images_on_resume:
        try:
            from core.rag.llamaindex.image_extractor import ingest_images_from_files

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
                rag,
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
        # parse_errors 透传给索引层写 kb_builds.error_msg（如「MinerU 解析失败: 超过 200 页上限」）
        return {"status": "empty", "chunks": 0, "files": len(file_paths), "images": 0, "parse_errors": parse_errors}

    # Phase 4: 图片位置回填 + 描述追加。
    # 回填无条件跑：ragflow_manual_docx 切块已埋 [[IMG:sha16]] 占位符，无论开关与否都要
    # 处理——开关关（fill=False）时占位符降级为空串被清理，避免索引留下字面量；
    # 开关开时按 image_desc_by_blob 清单回填真实描述。inlined = 已内联进正文的 sha16。
    # （LightRAG 路径的 KG 摄入总会产出清单，故必须用 fill 门控，否则关开关时描述仍漏进正文。）
    all_chunks, inlined = _resolve_image_placeholders(
        all_chunks, img_cache, fill=_chunk_cfg.inline_image_descriptions,
    )

    # 追加孤儿 chunk 仍由开关门控（默认关，需 CHUNKING__INLINE_IMAGE_DESCRIPTIONS=true）：
    # 把未内联进正文的图片描述作为独立 chunk 追加，让纯向量检索也能召回图片内容。
    if _chunk_cfg.inline_image_descriptions and not skip_images_on_resume:
        added = _append_image_desc_chunks(all_chunks, all_sources, img_cache, inlined)
        if added:
            logger.info("图片描述回填 course=%s 追加 %d 条", course_id, added)

    await asyncio.to_thread(
        persist_ingest_chunks,
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
    # rag 由 lease_instance 注入，无需再 _get_instance（H-10）

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
                # M-30：abort 时当前 batch 可能已有部分 chunk 写入图谱（LightRAG 无事务回滚）。
                # 不主动回滚——续传从本 batch 起点重跑整个 batch（done_at_batch_start），
                # LightRAG 按 chunk content 去重，已写入的 chunk 重跑时被识别为 duplicate 跳过，
                # partial batch 在续传中幂等收敛，不会产生重复实体。
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
                    # M-29：图片摄入在文本之前（Step 1b）已写入知识图谱。文本致命失败时，
                    # 图谱里已存在图片实体但缺少文本知识，处于"不完整"状态。
                    # raganything 的 process_multimodal_content 是"边处理边 ainsert"，
                    # 无法低成本回滚已写入的图片实体。此处显式告警，提示需排查/重索引。
                    if images_processed:
                        logger.error(
                            "文本索引致命失败，但 %d 张图片已先写入知识图谱（图谱可能不完整），"
                            "course=%s。修复 LLM 配置后建议重新索引以补齐文本知识。",
                            images_processed, course_id,
                        )
                    raise RuntimeError(f"LLM API 致命错误，索引中止: {err_msg[:300]}")
                else:
                    logger.warning("非致命 LLM 错误（继续）: %s", errors[0])

            # Phase 3: ES 双写（仅 settings.elasticsearch.enabled；失败仅告警不阻断）
            from core.rag.es_client import get_es_store
            es_store = get_es_store()
            if es_store is not None:
                await _index_batch_to_es(es_store, course_id, batch_chunks, batch_sources)

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
