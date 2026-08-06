"""解析层 IR：engine-agnostic 的 ParsedDocument（所有引擎产出同一份）。

介于"输入文件"与消费者（RAG 索引 ``file_paths_to_llama_documents``）之间。MinerU /
docling 都产出 ParsedDocument，消费者无需判断哪个引擎跑过。借鉴 DeepTutor
``services/parsing/types.py``。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class ParserError(RuntimeError):
    """解析失败或被 gate（引擎缺失/未配置/格式不支持）。携带用户可读消息。

    调用方（摄入链路）捕获它，写 KBFile.status=error + error_msg 给前端展示。
    """


@dataclass(frozen=True)
class ParsedDocument:
    """解析结果 IR。

    ``markdown`` 永远存在（所有引擎的最低公约数）；``blocks`` 是 MinerU ``content_list``
    形状的富结构（每块 type 为 text/title/table/equation/image、带 page_idx/text_level），
    为第三期按类型路由分块提供输入。``blocks`` 为 None 时消费者退化为切 markdown。
    """

    markdown: str
    blocks: Optional[list[dict[str, Any]]] = None
    source_hash: str = ""
    parser_signature: str = ""
    engine: str = ""
    workdir: Optional[Path] = None
    asset_dir: Optional[Path] = None  # images/ 目录（若有，供图片引用解析）

    @property
    def has_structure(self) -> bool:
        return bool(self.blocks)

    def to_sections(self) -> list[dict]:
        """blocks → sections（对齐 file_routing ``extract_pdf_sections``）。

        委托模块函数 ``blocks_to_sections``——与 ``chunking.type_routed`` 共用同一分组逻辑，
        避免 IR 转换与分块两处各写一份 blocks→sections。
        """
        return blocks_to_sections(self.blocks, self.markdown)


def blocks_to_sections(
    blocks: Optional[list[dict[str, Any]]],
    markdown_fallback: str = "",
) -> list[dict]:
    """blocks → sections（``[{title, content, page}]``），对齐 file_routing
    ``extract_pdf_sections`` 输出格式。

    无 blocks 时退化为单 section（markdown_fallback 整篇）。title 块开新 section，
    table 块原子化为独立 section，其他（text/equation/caption/list）累积进当前 section。
    平移自 file_routing ``_extract_pdf_sections_docling`` 的分组逻辑。
    ``ParsedDocument.to_sections`` 与 ``chunking.type_routed`` 共用本函数。
    """
    if not blocks:
        md = markdown_fallback.strip()
        return [{"title": "", "content": md, "page": 0}] if md else []

    sections: list[dict] = []
    cur_title = ""
    cur_parts: list[str] = []
    cur_page = 0

    def _flush() -> None:
        nonlocal cur_parts, cur_page
        if cur_parts:
            content = "\n\n".join(cur_parts)
            cur_parts = []
            if content.strip():
                sections.append({"title": cur_title, "content": content, "page": cur_page})
        cur_page = 0

    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        btype = str(blk.get("type") or blk.get("block_type") or "text").lower()
        page = int(blk.get("page_idx") or blk.get("page") or 0) or cur_page
        text = str(blk.get("text") or "").strip()
        if btype in ("title", "section_header"):
            _flush()
            cur_title = text or cur_title
            cur_page = page or cur_page
        elif btype == "table":
            _flush()
            if text:
                title = f"{cur_title}（表格）" if cur_title else "表格"
                sections.append({"title": title, "content": text, "page": page})
        else:  # text / list / caption / equation / image_desc ...
            if text:
                cur_parts.append(text)
                if not cur_page:
                    cur_page = page
    _flush()

    # 无标题无表格但 blocks 有内容 → 整篇 markdown 兜底单 section
    if not sections and markdown_fallback.strip():
        sections.append({"title": "", "content": markdown_fallback.strip(), "page": 0})
    return sections


__all__ = ["ParsedDocument", "ParserError", "blocks_to_sections"]
