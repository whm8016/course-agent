"""RAGAS Testset Generator —— 针对课程知识库自动合成 QA 评测集。

把"手写标准答案"换成"工具基于知识库自动出题"，这是 RAG 评测的专业做法：
合成的题保证答案都来自知识库文档（题就是从文档生成的），所以即便
faithfulness（防幻觉）不依赖标准答案，整套评测也有据可依。

输入（二选一）：
  - course_id：连库读该课程的 kb_files，自动定位文档
  - docs_dir：直接扫描一个目录下的文档

流程：
  原文件 → extract_text_from_path 提取纯文本 → RecursiveCharacterTextSplitter 切段
  → LangChain Document → TestsetGenerator.from_langchain(llm, emb).generate_with_langchain_docs(docs, n)
  → Testset → to_pandas() → 对齐 qa_dataset.json 结构落盘

输出 synthetic_dataset.json，结构与 qa_dataset.json 一致（id/question/ground_truth/...），
可直接喂给 run_eval 的现有评测链路（rag_runner + ragas_evaluator）。

复用 ragas_evaluator._build_llm / _build_embeddings（同一套 LLM/embedding 配置）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import config
from .ragas_evaluator import _build_embeddings, _build_llm

logger = logging.getLogger(__name__)

# 预切参数：控制喂给 TestsetGenerator 的 Document 数量与粒度
# （Document 太多会让 RAGAS 的 transforms 变慢 + 合成调 LLM 成本爆炸）
_CHUNK_SIZE = 1200
_CHUNK_OVERLAP = 120
_MAX_DOCS = 40

# knowledge_dir 兜底：kb_files.file_path 可能是相对路径，拼 BASE_DIR 试试
try:
    from settings import BASE_DIR  # type: ignore
except Exception:  # pragma: no cover
    BASE_DIR = Path(__file__).resolve().parents[2]  # backend/


def _supported_exts() -> set[str]:
    """复用 chat 附件的文档提取器支持的扩展名集合。"""
    try:
        from utils.document_extractor import SUPPORTED_DOC_EXTENSIONS
        return set(SUPPORTED_DOC_EXTENSIONS)
    except Exception:  # pragma: no cover
        return {".pdf", ".docx", ".pptx", ".txt", ".md"}


def _collect_dir_files(docs_dir: str | Path) -> list[Path]:
    """递归扫描目录下受支持的文档文件。"""
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"docs_dir 不是目录: {docs_dir}")
    exts = _supported_exts()
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


async def _collect_course_files(course_id: str) -> list[Path]:
    """连库读该课程 kb_files 的 file_path，转成可读 Path。"""
    from sqlalchemy import select

    from core.db.database import AsyncSessionLocal, KBFile, KnowledgeBase

    paths: list[Path] = []
    async with AsyncSessionLocal() as db:
        # 找该 course_id 对应的 KB
        kb = (
            await db.execute(
                select(KnowledgeBase).where(KnowledgeBase.course_id == course_id)
            )
        ).scalar_one_or_none()
        if kb is None:
            raise FileNotFoundError(f"课程 {course_id} 没有对应的知识库记录")
        rows = (
            await db.execute(
                select(KBFile.file_path, KBFile.original_name)
                .where(KBFile.kb_id == kb.id)
                .order_by(KBFile.created_at)
            )
        ).all()

    for fp, _name in rows:
        p = Path(fp)
        if not p.is_absolute():
            # 相对路径 → 拼 BASE_DIR 兜底
            cand = Path(BASE_DIR) / p
            if cand.exists():
                p = cand
        paths.append(p)
    return paths


def _load_documents(file_paths: list[Path]) -> list[Any]:
    """提取每个文件文本 → 切段 → 包装成 LangChain Document。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document as LCDocument

    from utils.document_extractor import extract_text_from_path

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    docs: list[Any] = []
    for p in file_paths:
        try:
            text = extract_text_from_path(p, max_bytes=20 * 1024 * 1024, max_chars=None)
        except Exception as exc:
            logger.warning("跳过文件 %s（提取失败）: %s", p.name, exc)
            continue
        if not text or not text.strip():
            logger.warning("跳过文件 %s（无文本）", p.name)
            continue
        for chunk in splitter.split_text(text):
            docs.append(LCDocument(page_content=chunk, metadata={"source": p.name}))
        if len(docs) >= _MAX_DOCS:
            logger.info("已达到 _MAX_DOCS=%d，停止读更多文件", _MAX_DOCS)
            break
    return docs[:_MAX_DOCS]


def _testset_to_items(testset: Any, source_label: str) -> list[dict]:
    """把 ragas Testset 转成 qa_dataset.json 结构的 list[dict]。"""
    df = testset.to_pandas()
    items: list[dict] = []
    for i, row in df.iterrows():
        # ragas 0.4 列名：user_input / reference / evolution_type / difficulty / ...
        question = str(row.get("user_input") or row.get("question") or "").strip()
        gt = str(row.get("reference") or row.get("ground_truth") or "").strip()
        if not question:
            continue
        items.append({
            "id": f"{source_label}{int(i) + 1:02d}",
            "question": question,
            "ground_truth": gt,
            "category": str(row.get("evolution_type") or "synthetic"),
            "difficulty": str(row.get("difficulty") or "medium"),
            "source": source_label,
        })
    return items


async def generate_dataset(
    *,
    course_id: str | None = None,
    docs_dir: str | None = None,
    n: int = 15,
    output_path: str | Path | None = None,
) -> list[dict]:
    """合成评测集，落盘并返回 list[dict]。

    course_id 与 docs_dir 至少传一个；都传时 course_id 优先。
    """
    import json

    # 1. 收集文件
    if course_id:
        file_paths = await _collect_course_files(course_id)
        source_label = "s"
        logger.info("课程 %s 收集到 %d 个文件", course_id, len(file_paths))
    elif docs_dir:
        file_paths = _collect_dir_files(docs_dir)
        source_label = "s"
        logger.info("目录 %s 收集到 %d 个文件", docs_dir, len(file_paths))
    else:
        raise ValueError("必须提供 course_id 或 docs_dir")

    if not file_paths:
        raise FileNotFoundError("没有找到可用的文档文件，无法合成评测集")

    # 2. 提取+切段
    docs = _load_documents(file_paths)
    if not docs:
        raise RuntimeError("所有文件都未能提取到文本，无法合成评测集")
    logger.info("切成 %d 个 Document 段（上限 %d）", len(docs), _MAX_DOCS)

    # 3. RAGAS 合成
    from ragas.testset import TestsetGenerator

    llm = _build_llm()
    embeddings = _build_embeddings()
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
    logger.info("开始合成 %d 道 QA（会调用 LLM，请耐心等待）...", n)
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=n,
        raise_exceptions=True,
    )

    # 4. 落盘
    items = _testset_to_items(testset, source_label)
    out = Path(output_path) if output_path else config.CACHE_DIR.parent / "synthetic_dataset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("合成完成：%d 道题 → %s", len(items), out)
    return items
