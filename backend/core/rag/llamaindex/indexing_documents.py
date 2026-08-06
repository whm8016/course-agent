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
) -> tuple[list[Document], FileClassification]:
    """
    将文件路径列表转为 LlamaIndex Document（供 LightRAG 摄入流水线使用）。
    """
    lg = log or logger
    classification = FileTypeRouter.classify_files(file_paths)
    documents: list[Document] = []

    _parsing_engine = get_settings().parsing.engine
    pdf_backend = get_settings().pdf.backend
    for file_path_str in classification.parser_files:
        file_path = Path(file_path_str).resolve()
        if _parsing_engine:
            # 解析层（MinerU 托管 API 等）：parse_document → ParsedDocument.to_sections()
            # opt-in：settings.parsing.engine 非空才走；失败即跳过该文件不降级（plan 单引擎哲学）
            try:
                from core.rag.parsing import ParserError, parse_document

                parsed = parse_document(file_path, engine=_parsing_engine)
                sections = parsed.to_sections()
                eng_name = parsed.engine or _parsing_engine
                content_list = parsed.blocks  # 供 type_routed 策略按结构分块
            except ParserError as exc:
                lg.error(
                    "解析失败（跳过该文件）%s（引擎 %s）: %s",
                    file_path.name, _parsing_engine, exc,
                )
                continue
        else:
            # 原 file_routing（docling/mupdf），settings.parsing.engine 空时行为零变化
            sections = FileTypeRouter.extract_pdf_sections(
                str(file_path), backend=pdf_backend
            )
            eng_name = pdf_backend
            content_list = None
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
                            "content_list": content_list,
                        },
                        # content_list 是整份 PDF 的 MinerU block JSON（可能几万 token），
                        # 只供 type_routed 策略读取，不该算进 embedding/LLM 的元数据长度——
                        # 否则 sentence_splitter 策略下 SentenceSplitter.split_text_metadata_aware
                        # 会把它计入 metadata_len，超过 chunk_size 直接抛 ValueError。
                        excluded_embed_metadata_keys=["content_list"],
                        excluded_llm_metadata_keys=["content_list"],
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

    return documents, classification
