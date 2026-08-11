"""LightRAG 摄入与 LlamaIndex 建库共用的文档加载与切块常量。

file_paths_to_llama_documents：统一 PDF（PyMuPDF）/ 文本 / DOCX（H1 章节）/ PPTX（逐页）→ LlamaIndex Document 列表。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from llama_index.core.schema import Document

from core.rag.llamaindex.file_routing import FileClassification, FileTypeRouter
from settings import get_settings

logger = logging.getLogger(__name__)

# 与 LlamaIndex Settings.chunk_size / chunk_overlap 及 LightRAG parse_files 中 SentenceSplitter 一致
LLAMA_INDEX_CHUNK_SIZE = 1200
LLAMA_INDEX_CHUNK_OVERLAP = 120


def file_paths_to_llama_documents(
    file_paths: list[str],
    *,
    log: Optional[logging.Logger] = None,
) -> tuple[list[Document], FileClassification, list[str]]:
    """
    将文件路径列表转为 LlamaIndex Document（供 LightRAG 摄入流水线使用）。

    返回第 3 个元素 parse_errors：逐文件解析失败原因（如「x.pdf: MinerU 解析失败: 超过 200 页上限」），
    供索引层在 0 chunk 时写进 kb_builds.error_msg，避免空索引被误判 ready 时毫无错误线索。
    """
    lg = log or logger
    classification = FileTypeRouter.classify_files(file_paths)
    documents: list[Document] = []
    parse_errors: list[str] = []

    _parsing_engine = get_settings().parsing.engine
    pdf_backend = get_settings().pdf.backend
    drop_references = get_settings().parsing.drop_references
    for file_path_str in classification.parser_files:
        file_path = Path(file_path_str).resolve()
        if _parsing_engine:
            # 解析层（MinerU 托管 API 等）：parse_document → markdown → sections
            # opt-in：settings.parsing.engine 非空才走；失败即跳过该文件不降级（单引擎哲学）。
            # 走 DeepTutor 式 markdown 路线（MarkdownNodeParser 按标题分节），不再用
            # blocks_to_sections——MinerU 标题是 type=text+text_level，后者分不出节。
            try:
                from core.rag.parsing import (
                    ParserError,
                    markdown_to_sections,
                    parse_document,
                    resolve_image_refs,
                )

                parsed = parse_document(file_path, engine=_parsing_engine)
                markdown = resolve_image_refs(parsed.markdown, parsed.asset_dir)
                sections = markdown_to_sections(
                    markdown, blocks=parsed.blocks, drop_refs=drop_references
                )
                eng_name = parsed.engine or _parsing_engine
            except ParserError as exc:
                lg.error(
                    "解析失败（跳过该文件）%s（引擎 %s）: %s",
                    file_path.name, _parsing_engine, exc,
                )
                parse_errors.append(f"{file_path.name}: {exc}")
                continue
        else:
            # 原 file_routing（docling/mupdf），settings.parsing.engine 空时行为零变化
            sections = FileTypeRouter.extract_pdf_sections(
                str(file_path), backend=pdf_backend
            )
            eng_name = pdf_backend
        if sections:
            for sec in sections:
                documents.append(
                    Document(
                        text=sec["content"],
                        metadata={
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "section": sec["title"],
                            "page": sec["page"],
                        },
                    )
                )
            lg.info("Loaded: %s → %d sections（引擎 %s）", file_path.name, len(sections), eng_name)
        else:
            lg.warning("Skipped empty PDF: %s", file_path.name)

    for file_path_str in classification.text_files:
        file_path = Path(file_path_str).resolve()
        lg.info("Parsing text: %s", file_path.name)
        text = FileTypeRouter.read_text_file_sync(str(file_path))
        if text.strip():
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                    },
                )
            )
            lg.info("Loaded: %s (%d chars)", file_path.name, len(text))
        else:
            lg.warning("Skipped empty document: %s", file_path.name)

    for file_path_str in classification.docx_files:
        file_path = Path(file_path_str).resolve()
        lg.info("Parsing DOCX (section-aware): %s", file_path.name)
        sections = FileTypeRouter.extract_docx_sections(str(file_path))
        if sections:
            for sec in sections:
                documents.append(
                    Document(
                        text=sec["content"],
                        metadata={
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "section": sec["title"],
                        },
                    )
                )
            lg.info("Loaded: %s → %d sections", file_path.name, len(sections))
        else:
            lg.warning("Skipped empty or unreadable DOCX: %s", file_path.name)

    for file_path_str in classification.pptx_files:
        file_path = Path(file_path_str).resolve()
        lg.info("Parsing PPTX (slide-aware): %s", file_path.name)
        sections = FileTypeRouter.extract_pptx_sections(str(file_path))
        if sections:
            for sec in sections:
                documents.append(
                    Document(
                        text=sec["content"],
                        metadata={
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "section": sec["title"],
                        },
                    )
                )
            lg.info("Loaded: %s → %d slides", file_path.name, len(sections))
        else:
            lg.warning("Skipped empty or unreadable PPTX: %s", file_path.name)

    for file_path_str in classification.unsupported:
        lg.warning("Skipped unsupported file: %s", Path(file_path_str).name)

    return documents, classification, parse_errors
