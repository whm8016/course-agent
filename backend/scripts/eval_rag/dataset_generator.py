"""RAGAS Testset Generator —— 针对课程知识库自动合成 QA 评测集。

把"手写标准答案"换成"工具基于知识库自动出题"，这是 RAG 评测的专业做法：
合成的题保证答案都来自知识库文档（题就是从文档生成的），所以即便
faithfulness（防幻觉）不依赖标准答案，整套评测也有据可依。

输入（二选一）：
  - course_id：连库读该课程的 kb_files，自动定位文档
  - docs_dir：直接扫描一个目录下的文档

流程（切割对齐生产摄入链路 core/rag/ingestion.parse_files）：
  file_paths_to_llama_documents（章节感知：DOCX按H1/PPTX按页/PDF全文）
  → SentenceSplitter(LLAMA_INDEX_CHUNK_SIZE/OVERLAP) → _build_source_prefix 注入【章节/来源】前缀
  → TestsetGenerator.from_langchain(llm, emb).generate_with_chunks(chunks, n)
  → Testset → to_pandas() → 对齐 qa_dataset.json 结构落盘

输出 synthetic_dataset.json，结构与 qa_dataset.json 一致（id/question/ground_truth/...），
可直接喂给 run_eval 的现有评测链路（rag_runner + ragas_evaluator）。

用 ragas 原生 _build_ragas_llm(GEN_LLM_MODEL) / _build_ragas_embeddings(async)——绕开 langchain async 与千问/DeepSeek 的兼容问题。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import config
from .ragas_evaluator import _build_ragas_embeddings, _build_ragas_llm

logger = logging.getLogger(__name__)

# chunk 数量上限：防止合成题成本失控（切割参数复用生产 LLAMA_INDEX_CHUNK_SIZE/OVERLAP）
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
    """连库读该课程 kb_files 的 file_path，转成可读 Path。

    file_path 可能是历史/生产环境写入的绝对路径（如 C:\\app\\kb_store\\... 或 /app/kb_store/...），
    换到本地开发机上该绝对路径往往不存在（库与文件分属不同环境）。故除原路径外，额外用当前
    KB_STORE_DIR 按 (course_id/raw/basename) 重映射兜底，选第一个真实存在的候选路径——
    生产环境原路径命中则行为不变，本地跨环境场景自动定位到真实文件。
    """
    from sqlalchemy import select

    try:
        from settings import get_settings

        kb_store = Path(get_settings().paths.kb_store_dir)
    except Exception:  # pragma: no cover
        kb_store = Path(BASE_DIR) / "kb_store"

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
        # 候选：原路径（绝对/相对）→ BASE_DIR 兜底 → KB_STORE_DIR 重映射兜底
        # 选第一个真实存在的；都不在则跳过并告警（不让坏路径进 _load_documents 导致整体崩）
        candidates: list[Path] = [p]
        if not p.is_absolute():
            candidates.append(Path(BASE_DIR) / p)
        candidates.append(kb_store / course_id / "raw" / p.name)
        resolved = next((c for c in candidates if c.exists()), None)
        if resolved is not None:
            paths.append(resolved)
        else:
            logger.warning(
                "课程 %s 文件未找到，跳过 %s（已尝试: %s）",
                course_id, p.name, [str(c) for c in candidates],
            )
    return paths


def _load_documents(file_paths: list[Path]) -> list[str]:
    """复用生产摄入切割，返回带章节前缀的 chunk 字符串列表。

    与 core/rag/ingestion.parse_files 完全一致：file_paths_to_llama_documents（PDF/文本/DOCX按H1章节/
    PPTX按页）→ SentenceSplitter(LLAMA_INDEX_CHUNK_SIZE/OVERLAP) → _build_source_prefix 注入
    【章节: xxx】/【来源: filename】前缀。保证 RAGAS 合成题看到的文档结构与生产索引一致，
    避免章节边界处出跨章节混淆的题（尤其电路课这种章节强相关教材）。
    """
    from llama_index.core.node_parser import SentenceSplitter

    from core.rag.ingestion import _build_source_prefix
    from core.rag.llamaindex.indexing_documents import (
        LLAMA_INDEX_CHUNK_OVERLAP,
        LLAMA_INDEX_CHUNK_SIZE,
        file_paths_to_llama_documents,
    )

    documents, _classification, _parse_errors = file_paths_to_llama_documents(
        [str(p) for p in file_paths], log=logger
    )
    if not documents:
        return []

    nodes = SentenceSplitter(
        chunk_size=LLAMA_INDEX_CHUNK_SIZE,
        chunk_overlap=LLAMA_INDEX_CHUNK_OVERLAP,
        # 按字符计数，与生产摄入 _chunk_by_sentence_splitter 保持一致
        tokenizer=lambda t: t,
    ).get_nodes_from_documents(documents)

    chunks: list[str] = []
    for n in nodes:
        content = n.get_content().strip()
        if not content:
            continue
        meta = getattr(n, "metadata", None) or {}
        prefix = _build_source_prefix(
            section=str(meta.get("section", "") or ""),
            file_name=str(meta.get("file_name", "") or ""),
        )
        chunks.append(f"{prefix}{content}" if prefix else content)
        if len(chunks) >= _MAX_DOCS:
            logger.info("已达到 _MAX_DOCS=%d，停止切更多 chunk", _MAX_DOCS)
            break
    return chunks[:_MAX_DOCS]


def _patch_ragas_persona_dedup() -> None:
    """修 ragas 0.4.3 重名 persona 导致 multi_hop 出题 KeyError。

    现象：generate_with_chunks 跑完 NER/Themes 等全部 transform 后，在 Generating Scenarios
    阶段崩 KeyError "No persona found with name 'X (2)'"。

    根因：generate_personas_from_kg 对内容同质的文档（如电路课教案高度相似）会生成**重名**
    persona（如两个 "Electronics Engineering Student"）。multi_hop synthesizer 把重名 persona
    喂给 LLM 做 theme-persona 匹配时，LLM 为区分自行在返回的 mapping key 上加 (N) 后缀；但
    PersonaList（multi_hop/base.py:62）存的是原始重名 → prepare_combinations 按 "X (2)" 查
    PersonaList 查不到 → KeyError。

    两层修复（治本 + 兜底）：
      1. 去重源头：patch generate 模块里的 generate_personas_from_kg（generate.py 是
         from...import 直接绑定，须 patch 它模块内的引用而非 persona 模块），给重名 persona
         加 (N) 后缀。name 唯一后 LLM 不再自行加后缀，mapping key 与 PersonaList 天然一致。
      2. 兜底：patch PersonaList.__getitem__，精确名查不到时 strip 末尾 (N) 再匹配。
    幂等（_patched 标记防重复）；ragas 若日后修了该 bug，精确匹配先成功，兜底不触发，无害。
    """
    import re as _re

    import ragas.testset.synthesizers.generate as _gen
    from ragas.testset.persona import PersonaList

    # 1. 去重 persona name（generate.py 是 from...import，须 patch 它模块内的引用）
    if not getattr(_gen.generate_personas_from_kg, "_dedup_patched", False):
        _orig_gen = _gen.generate_personas_from_kg

        def _dedup_gen(*args: Any, **kwargs: Any):
            personas = _orig_gen(*args, **kwargs)
            seen: dict[str, int] = {}
            for p in personas:
                base = p.name
                seen[base] = seen.get(base, 0) + 1
                if seen[base] > 1:
                    p.name = f"{base} ({seen[base]})"
            return personas

        _dedup_gen._dedup_patched = True  # type: ignore[attr-defined]
        _gen.generate_personas_from_kg = _dedup_gen

    # 2. 兜底：PersonaList.__getitem__ 容错 strip 末尾 (N)
    if not getattr(PersonaList.__getitem__, "_tolerant_patched", False):
        _orig_get = PersonaList.__getitem__

        def _tolerant_get(self: Any, key: str):
            try:
                return _orig_get(self, key)
            except KeyError:
                stripped = _re.sub(r"\s*\(\d+\)\s*$", "", key).strip()
                if stripped and stripped != key:
                    for p in self.personas:
                        if p.name == stripped:
                            return p
                raise

        _tolerant_get._tolerant_patched = True  # type: ignore[attr-defined]
        PersonaList.__getitem__ = _tolerant_get  # type: ignore[assignment]


def _chinese_persona_list() -> list[Any]:
    """固定的中文 persona 列表，替代 ragas 默认的 LLM 生成 persona。

    根因：generate_personas_from_kg 用的 PersonaGenerationPrompt 只有一条英文 few-shot
    示例（"Digital Marketing Specialist"），LLM 即便读到中文教材摘要也常常照着示例的
    英文风格编 persona（如 "electronics student focusing on..."）；后续出题 prompt 又是
    照抄 persona 的语言组织问题，于是整份合成题一半中文一半英文。
    直接给固定的中文 persona，从源头掐掉这条英文污染链路，顺带省一次 LLM 调用。
    """
    from ragas.testset.persona import Persona

    return [
        Persona(
            name="认真预习复习的学生",
            role_description="课前预习、课后复习课程内容，关心概念原理和实验步骤是否吃透。",
        ),
        Persona(
            name="缺乏基础的自学者",
            role_description="自学该课程，基础较弱，希望用通俗易懂的中文重新理解重点概念。",
        ),
        Persona(
            name="核对细节的助教",
            role_description="批改作业和实验报告，需要核对讲义中的技术细节和标准答案是否准确。",
        ),
    ]


async def _adapt_query_distribution_to_chinese(query_distribution: Any, llm: Any) -> None:
    """把出题/主题匹配等 prompt 的 few-shot 示例整体翻译成中文（就地替换 synthesizer 的 prompt）。

    根因同上：single_hop/multi_hop 的 QueryAnswerGenerationPrompt 示例也是英文
    （"Software Engineer" / "Historian" persona + 英文 query/answer），LLM 会照着示例的
    语言续写。用 ragas 官方的 PromptMixin.adapt_prompts（内部调用 PydanticPrompt.adapt）
    把该 synthesizer 名下所有 prompt 的示例和 instruction 整体翻译成中文，
    而不是猜哪个字段该改——比手写中文示例更不容易漏改某个 prompt。
    """
    for synthesizer, _weight in query_distribution:
        adapted = await synthesizer.adapt_prompts("chinese", llm=llm, adapt_instruction=True)
        synthesizer.set_prompts(**adapted)


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
    logger.info("切成 %d 个 chunk（对齐生产切割，上限 %d）", len(docs), _MAX_DOCS)

    # 3. RAGAS 合成
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.synthesizers import default_query_distribution
    from ragas.testset.transforms import apply_transforms, default_transforms_for_prechunked

    # 修 ragas 0.4.3 重名 persona 导致 multi_hop 出题 KeyError（见 _patch_ragas_persona_dedup）
    _patch_ragas_persona_dedup()

    # ragas 原生 llm(embedding)，绕开 langchain async 与千问/DeepSeek 的兼容问题
    # max_tokens 传 GEN_LLM_MAX_TOKENS(8192)：NER 抽实体输出长，ragas 默认 1024 会截断拉崩
    generator = TestsetGenerator(
        llm=_build_ragas_llm(config.GEN_LLM_MODEL, max_tokens=config.GEN_LLM_MAX_TOKENS),
        embedding_model=_build_ragas_embeddings(async_client=True),
    )
    generator.persona_list = _chinese_persona_list()

    # 手动展开 generate_with_chunks 内部的两步（建知识图谱 → 出题），是为了能在建完图谱、
    # 知道真实可用的出题方式后再做中文适配，避免像之前那样跳过用真实图谱过滤
    # single_hop/multi_hop 是否可用——某些出题方式在图谱里找不到可用节点/关系时会直接报错，
    # 白白浪费已经花掉的 LLM tokens。这里改成：找不到就跳过并打日志提醒，不报错、不崩。
    nodes = [
        Node(type=NodeType.CHUNK, properties={"page_content": chunk, "document_metadata": {}})
        for chunk in docs
    ]
    kg = KnowledgeGraph(nodes=nodes)
    transforms = default_transforms_for_prechunked(
        llm=generator.llm, embedding_model=generator.embedding_model
    )
    logger.info("构建知识图谱（NER/摘要等抽取，会调用 LLM，请耐心等待）...")
    apply_transforms(kg, transforms, run_config=RunConfig())
    generator.knowledge_graph = kg

    all_synthesizer_names = {s.name for s, _ in default_query_distribution(generator.llm)}
    query_distribution = default_query_distribution(generator.llm, kg)
    skipped = all_synthesizer_names - {s.name for s, _ in query_distribution}
    if skipped:
        logger.warning("知识图谱中没有可用的节点/关系，已跳过这些出题方式: %s", skipped)

    # 把出题 prompt 的英文 few-shot 示例翻译成中文，否则合成题会一半中文一半英文
    # （见 _adapt_query_distribution_to_chinese）
    await _adapt_query_distribution_to_chinese(query_distribution, generator.llm)

    logger.info("开始合成 %d 道 QA（会调用 LLM，请耐心等待）...", n)
    testset = generator.generate(
        testset_size=n,
        query_distribution=query_distribution,
        raise_exceptions=True,
    )

    # 4. 落盘
    items = _testset_to_items(testset, source_label)
    out = Path(output_path) if output_path else config.CACHE_DIR.parent / "synthetic_dataset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("合成完成：%d 道题 → %s", len(items), out)
    return items
