"""extract_pdf_sections(mupdf backend) 单测：章节切分 + 页码 + 无书签兜底 + 路由分发。

锁定的行为（file_routing._extract_pdf_sections_mupdf）：
  1. 有 toc 书签 → 按 get_toc() 切章节，每节 page = 章节起始页(1-based)，content 为该页范围文本。
  2. 无 toc → 整篇单 section，title=""、page=1。
  3. 空文档（无文字）→ 返回 []（不产出空 section）。
  4. extract_pdf_sections(backend="mupdf") 路由到 mupdf 路径，与直接调等价。
全部用英文构造 PDF（fontname=helv = fitz 内置 Helvetica），避开 CJK 字体坑
（默认 Helvetica 不支持中文，get_text 读回空）。
"""
from __future__ import annotations

import fitz


def _make_pdf(path, pages_text: list[str], toc: list | None = None) -> str:
    """构造测试 PDF：每页一段文字，可选书签 toc=[[level, title, page_1based], ...]。"""
    doc = fitz.open()
    for txt in pages_text:
        page = doc.new_page()
        if txt:
            page.insert_text((72, 72), txt, fontname="helv", fontsize=12)
    if toc:
        doc.set_toc(toc)
    doc.save(str(path))
    doc.close()
    return str(path)


def test_mupdf_with_toc_splits_chapters(tmp_path):
    """有书签 → 按章节切，每节带正确起始页与该页文本。"""
    from core.rag.llamaindex.file_routing import FileTypeRouter

    pdf = _make_pdf(
        tmp_path / "ch.pdf",
        pages_text=["Alpha content one", "Beta content two"],
        toc=[[1, "Chapter One", 1], [1, "Chapter Two", 2]],
    )
    secs = FileTypeRouter._extract_pdf_sections_mupdf(pdf)
    assert len(secs) == 2
    assert secs[0]["title"] == "Chapter One"
    assert secs[0]["page"] == 1
    assert "Alpha content one" in secs[0]["content"]
    assert secs[1]["title"] == "Chapter Two"
    assert secs[1]["page"] == 2
    assert "Beta content two" in secs[1]["content"]


def test_mupdf_no_toc_single_section(tmp_path):
    """无书签 → 整篇单 section，title 为空、page=1（退化路径）。"""
    from core.rag.llamaindex.file_routing import FileTypeRouter

    pdf = _make_pdf(tmp_path / "notoc.pdf", pages_text=["Plain text body"])
    secs = FileTypeRouter._extract_pdf_sections_mupdf(pdf)
    assert len(secs) == 1
    assert secs[0]["title"] == ""
    assert secs[0]["page"] == 1
    assert "Plain text body" in secs[0]["content"]


def test_mupdf_empty_pdf_returns_empty(tmp_path):
    """无文字的空文档 → 返回 []（不产出空内容 section）。"""
    from core.rag.llamaindex.file_routing import FileTypeRouter

    pdf = _make_pdf(tmp_path / "empty.pdf", pages_text=[""])
    assert FileTypeRouter._extract_pdf_sections_mupdf(pdf) == []


def test_extract_pdf_sections_routes_to_mupdf(tmp_path):
    """extract_pdf_sections(backend='mupdf') 与直接调 _extract_pdf_sections_mupdf 等价。"""
    from core.rag.llamaindex.file_routing import FileTypeRouter

    pdf = _make_pdf(
        tmp_path / "route.pdf",
        pages_text=["X"],
        toc=[[1, "Only", 1]],
    )
    routed = FileTypeRouter.extract_pdf_sections(pdf, backend="mupdf")
    direct = FileTypeRouter._extract_pdf_sections_mupdf(pdf)
    assert [s["title"] for s in routed] == [s["title"] for s in direct]
    assert [s["page"] for s in routed] == [s["page"] for s in direct]
