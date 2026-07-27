"""extract_pdf_sections(docling backend) 单测：组装逻辑 + 兜底契约（mock，不依赖真实模型）。

为什么 mock 而非真实跑：docling 真实 convert 要下 DocLayNet/TableFormer 模型（几百 MB），
且本机中文路径触发 docling-parse issue #115（C++ 后端找不到 .dat 资源）→ convert 失败。
故这里用假 converter/doc/item 覆盖 **我们的组装逻辑**（file_routing._extract_pdf_sections_docling），
不测 docling 模型本身的识别能力（那是第三方能力，非本项目代码）。

锁定的行为（file_routing._extract_pdf_sections_docling:589-658）：
  1. section_header 开新 section；其后段落累积进当前 title。
  2. table 原子化为独立 section（title 追加「表格」），不混入正文，page 取表格所在页。
  3. 无任何产出 → 回退整文档 export_to_markdown 单 section（page=0）。
  4. docling 未装（_get_docling_converter 抛 ImportError）→ 返回 []，不降级、不崩溃。
  5. converter.convert 抛异常 → 返回 []。
（真实端到端 smoke 待中文路径阻塞解决后补，将标 @pytest.mark.slow。）
"""
from __future__ import annotations


# ── 假对象：最小化复现 docling DocumentConverter / DoclingDocument / item 的被用 API ──


class _FakeProv:
    def __init__(self, page_no: int):
        self.page_no = page_no


class _FakeItem:
    """复刻 docling item 被 _extract_pdf_sections_docling 用到的属性。"""

    def __init__(self, label: str, text: str = "", page: int = 0, table_md: str = ""):
        self.label = label
        self.text = text
        self._page = page
        self._table_md = table_md

    @property
    def prov(self):
        return [_FakeProv(self._page)] if self._page else []

    def export_to_markdown(self, doc=None) -> str:
        return self._table_md


class _FakeDoc:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        for it in self._items:
            yield (it, 0)

    def export_to_markdown(self) -> str:
        return "FALLBACK_MD"


class _FakeResult:
    def __init__(self, doc):
        self.document = doc


class _FakeConverter:
    def __init__(self, doc):
        self._doc = doc

    def convert(self, path):
        return _FakeResult(self._doc)


def _patch_converter(monkeypatch, items):
    """把 file_routing._get_docling_converter 换成返回装着 items 的假 converter。"""
    from core.rag.llamaindex import file_routing

    conv = _FakeConverter(_FakeDoc(items))
    monkeypatch.setattr(file_routing, "_get_docling_converter", lambda: conv)
    return file_routing.FileTypeRouter


def test_docling_assembles_sections_headers_and_tables(monkeypatch, tmp_path):
    """section_header 切章 + table 原子化 + 表格后段落归当前 title + 页码注入。"""
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 mock")  # converter 已 mock，不会真读此文件
    Router = _patch_converter(monkeypatch, [
        _FakeItem("section_header", "Intro", page=1),
        _FakeItem("paragraph", "hello world", page=1),
        _FakeItem("table", table_md="| a | b |\n|---|---|\n| 1 | 2 |", page=2),
        _FakeItem("paragraph", "after table", page=2),
    ])
    secs = Router._extract_pdf_sections_docling(str(pdf))

    # 三个 section：Intro(hello) / Intro（表格） / Intro(after)
    assert len(secs) == 3
    # 正文段「hello world」归入 Intro，page=1
    assert secs[0]["title"] == "Intro"
    assert "hello world" in secs[0]["content"]
    assert secs[0]["page"] == 1
    # 表格原子化为独立 section，title 追加「表格」，page=2，内容是 markdown 表
    assert "表格" in secs[1]["title"]
    assert secs[1]["page"] == 2
    assert "| a | b |" in secs[1]["content"]
    # 表格后的段落仍归当前 title（Intro），page=2
    assert secs[2]["title"] == "Intro"
    assert "after table" in secs[2]["content"]
    assert secs[2]["page"] == 2


def test_docling_fallback_whole_doc_when_no_items(monkeypatch, tmp_path):
    """无任何产出 → 回退整文档 export_to_markdown 单 section（page=0）。"""
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-1.4 mock")
    Router = _patch_converter(monkeypatch, [])
    secs = Router._extract_pdf_sections_docling(str(pdf))
    assert len(secs) == 1
    assert secs[0]["content"] == "FALLBACK_MD"
    assert secs[0]["page"] == 0


def test_docling_import_error_returns_empty(monkeypatch, tmp_path):
    """docling 未装（_get_docling_converter 抛 ImportError）→ 返回 []，不降级、不崩溃。"""
    from core.rag.llamaindex import file_routing

    monkeypatch.setattr(file_routing, "_get_docling_converter", lambda: (_ for _ in ()).throw(ImportError("no docling")))
    secs = file_routing.FileTypeRouter._extract_pdf_sections_docling(str(tmp_path / "x.pdf"))
    assert secs == []


def test_docling_convert_error_returns_empty(monkeypatch, tmp_path):
    """converter.convert 抛异常（如 PDF 损坏）→ 返回 []，不崩溃。"""
    from core.rag.llamaindex import file_routing

    class _BoomConv:
        def convert(self, path):
            raise RuntimeError("boom")

    monkeypatch.setattr(file_routing, "_get_docling_converter", lambda: _BoomConv())
    secs = file_routing.FileTypeRouter._extract_pdf_sections_docling(str(tmp_path / "x.pdf"))
    assert secs == []


def test_extract_pdf_sections_routes_to_docling_by_default(monkeypatch, tmp_path):
    """extract_pdf_sections(backend='docling') 与显式 'docling' 都走 docling 路径（非 mupdf）。"""
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4 mock")
    Router = _patch_converter(monkeypatch, [_FakeItem("paragraph", "routed", page=3)])
    # 默认 backend（不传）= docling
    secs_default = Router.extract_pdf_sections(str(pdf))
    secs_explicit = Router.extract_pdf_sections(str(pdf), backend="docling")
    assert [s["content"] for s in secs_default] == [s["content"] for s in secs_explicit]
    assert secs_default[0]["page"] == 3
