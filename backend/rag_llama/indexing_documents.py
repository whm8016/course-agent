"""LightRAG 摄入与 LlamaIndex 建库共用的文档加载与切块常量。

file_paths_to_llama_documents：统一 PDF（PyMuPDF）/ 文本 / DOCX（H1 章节）→ LlamaIndex Document 列表。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from llama_index.core.schema import Document

from rag_llama.file_routing import FileClassification, FileTypeRouter

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
    将文件路径列表转为 LlamaIndex Document（与 LlamaIndexPipeline.initialize / add_documents 原逻辑一致）。
    """
    lg = log or logger
    classification = FileTypeRouter.classify_files(file_paths)
    documents: list[Document] = []

    for file_path_str in classification.parser_files:
        file_path = Path(file_path_str).resolve()
        lg.info("Parsing PDF: %s", file_path.name)
        text = FileTypeRouter.extract_pdf_text(str(file_path))
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

    for file_path_str in classification.unsupported:
        lg.warning("Skipped unsupported file: %s", Path(file_path_str).name)

    return documents, classification
