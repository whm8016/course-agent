"""Docling 自托管解析引擎（可选，装 parse-docling extra）。

平移 file_routing._extract_pdf_sections_docling 的 docling 集成，作为 parsing 层可选引擎
（数据不出域的部署用）。复用 file_routing._get_docling_converter 单例（避免重复加载
torch/DocLayNet 模型，2.5GB 常驻）。产出 markdown + content_list.json（blocks），
与 MinerU 引擎 IR 一致——消费者无需判断哪个引擎跑过。

默认不装：云端用 mineru_api（去 torch）。装 ``pip install -e .[parse-docling]`` 启用。
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from core.rag.parsing.signature import ParserSignature
from core.rag.parsing.types import ParserError
from settings import get_settings

logger = logging.getLogger(__name__)


class DoclingEngine:
    """Docling 自托管解析（PDF → markdown + content_list blocks）。"""

    name = "docling"

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("docling") is not None

    def resolve_config(self) -> dict[str, Any]:
        cfg = get_settings().pdf
        return {"do_ocr": bool(cfg.do_ocr), "ocr_provider": cfg.ocr_provider}

    def supported_formats(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def signature(self, config: dict[str, Any]) -> ParserSignature:
        try:
            ver = importlib.metadata.version("docling")
        except importlib.metadata.PackageNotFoundError:
            ver = ""
        return ParserSignature.build(
            "docling",
            ver,
            {"do_ocr": config["do_ocr"], "ocr_provider": config["ocr_provider"]},
        )

    def is_ready(self, config: dict[str, Any]) -> tuple[bool, str]:
        if not self.is_available():
            return False, "docling 未安装（装 parse-docling extra：pip install -e .[parse-docling]）"
        return True, ""

    def parse(
        self,
        source_path: Path,
        workdir: Path,
        *,
        config: dict[str, Any],
        on_output: Optional[Callable[[str], None]] = None,
    ) -> None:
        try:
            from core.rag.llamaindex.file_routing import (
                _docling_item_page,
                _get_docling_converter,
            )
        except ImportError as exc:
            raise ParserError(f"file_routing 模块不可用: {exc}") from exc
        try:
            converter = _get_docling_converter()
        except ImportError as exc:
            raise ParserError(f"docling 依赖未装: {exc}") from exc

        if on_output:
            try:
                on_output(f"docling: 解析 {Path(source_path).name}")
            except Exception:
                pass
        try:
            ddoc = converter.convert(str(source_path)).document
        except Exception as exc:
            raise ParserError(f"docling 解析失败 {Path(source_path).name}: {exc}") from exc

        # items → blocks（与 MinerU content_list 同构：type/text/page_idx）
        blocks: list[dict[str, Any]] = []
        for item, _level in ddoc.iterate_items():
            label = getattr(item, "label", "")
            page = _docling_item_page(item) or 0
            if label == "section_header":
                text = (getattr(item, "text", "") or "").strip()
                if text:
                    blocks.append({"type": "title", "text": text, "page_idx": page})
            elif label == "table":
                try:
                    tbl_md = item.export_to_markdown(doc=ddoc)
                except Exception:
                    tbl_md = getattr(item, "text", "") or ""
                if tbl_md and tbl_md.strip():
                    blocks.append({"type": "table", "text": tbl_md, "page_idx": page})
            else:  # paragraph / list_item / caption / ...
                text = (getattr(item, "text", "") or "").strip()
                if text:
                    blocks.append({"type": "text", "text": text, "page_idx": page})

        markdown = ddoc.export_to_markdown() or ""
        if not markdown and not blocks:
            raise ParserError(f"docling 对 {Path(source_path).name} 未产出内容")

        stem = Path(source_path).stem
        (workdir / f"{stem}.md").write_text(markdown, encoding="utf-8")
        if blocks:
            (workdir / f"{stem}_content_list.json").write_text(
                json.dumps(blocks, ensure_ascii=False), encoding="utf-8"
            )
        logger.info("docling 解析完成: %s → %d blocks", Path(source_path).name, len(blocks))


__all__ = ["DoclingEngine"]
