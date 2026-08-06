"""type_routed 分块：按 MinerU ``content_list`` 的块结构分块。

先 ``blocks_to_sections``（title 开新 section、table 原子化，保语义边界），再对超
``max_chars`` 的 section 递归切（保 section 名）。比 ``sentence_splitter`` 盲切保留
结构边界——Vecta 2026 七策略对比：递归 512 准确率 69% 第一；结构化技术文档用结构
分块达 87%（premai 2026 benchmark）。

依赖第二期 parsing 层产出的 ``content_list``（blocks）。无 blocks 时由 ingestion 的
``_chunk_type_routed_strategy`` 退化 ``sentence_splitter``。**opt-in**：默认策略仍是
``sentence_splitter``（行为零变化），plan 明确 chunk_size 改默认要 eval_rag 验证不直接上线。
"""
from __future__ import annotations

from core.rag.parsing.types import blocks_to_sections


def chunk_blocks_structured(
    blocks: list[dict],
    max_chars: int = 512,
) -> tuple[list[str], list[str]]:
    """blocks → (chunks, section_names)。

    先 blocks_to_sections（保 title/table 边界），再对超 max_chars 的 section 递归切
    （保 section 名，无 overlap——结构边界已保语义，overlap 收益小且增碎片）。
    返回等长的 chunks 与 section_names（供 ``_build_source_prefix`` 注入来源前缀）。
    """
    sections = blocks_to_sections(blocks)
    chunks: list[str] = []
    sec_names: list[str] = []
    for sec in sections:
        content = (sec.get("content") or "").strip()
        title = sec.get("title") or ""
        if not content:
            continue
        if len(content) <= max_chars:
            chunks.append(content)
            sec_names.append(title)
        else:
            # 递归切（保 section 名）
            for i in range(0, len(content), max_chars):
                piece = content[i : i + max_chars].strip()
                if piece:
                    chunks.append(piece)
                    sec_names.append(title)
    return chunks, sec_names


__all__ = ["chunk_blocks_structured"]
