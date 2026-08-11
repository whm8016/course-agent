"""解析层（parsing/）单测：IR 转换 / 签名 / 内容寻址缓存 / registry / service 调度 / MinerU 引擎配置。

纯 mock + tmp_path 真实文件系统——不调真实 MinerU API（留 Docker）。验证：
- ParsedDocument.to_sections 把 MinerU content_list 转成 file_routing 的 sections 格式
- ParserSignature 稳定 + api_token 不进签名
- cache 命中/未命中/manifest ready 语义
- service 缓存命中不重复 parse、格式不支持直接报错不换引擎、readiness gate
- MinerU 引擎 config/signature/is_ready（parse 的 httpx 流程留集成测试）
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


# ── ParsedDocument.to_sections ────────────────────────────────────────────────


class TestToSections:
    def test_blocks_to_sections(self):
        from core.rag.parsing.types import ParsedDocument

        doc = ParsedDocument(
            markdown="",
            blocks=[
                {"type": "title", "text": "第一章", "page_idx": 0},
                {"type": "text", "text": "内容A", "page_idx": 0},
                {"type": "text", "text": "内容B", "page_idx": 1},
                {"type": "table", "text": "| a | b |", "page_idx": 2},
                {"type": "title", "text": "第二章", "page_idx": 3},
                {"type": "text", "text": "内容C", "page_idx": 3},
            ],
        )
        sections = doc.to_sections()
        # title 开新 section，table 原子化，text 累积
        assert len(sections) == 3
        assert sections[0]["title"] == "第一章"
        assert "内容A" in sections[0]["content"] and "内容B" in sections[0]["content"]
        assert sections[1]["title"] == "第一章（表格）"
        assert sections[1]["content"] == "| a | b |"
        assert sections[2]["title"] == "第二章"
        assert sections[2]["content"] == "内容C"

    def test_no_blocks_degrades_to_single_section(self):
        from core.rag.parsing.types import ParsedDocument

        doc = ParsedDocument(markdown="整篇 markdown 内容")
        sections = doc.to_sections()
        assert len(sections) == 1
        assert sections[0]["content"] == "整篇 markdown 内容"
        assert sections[0]["page"] == 0

    def test_empty(self):
        from core.rag.parsing.types import ParsedDocument

        assert ParsedDocument(markdown="").to_sections() == []
        assert ParsedDocument(markdown="", blocks=[]).to_sections() == []


# ── ParserSignature ───────────────────────────────────────────────────────────


class TestSignature:
    def test_stable_hash(self):
        from core.rag.parsing.signature import ParserSignature

        s1 = ParserSignature.build("mineru_api", "v1", {"a": 1, "b": 2})
        s2 = ParserSignature.build("mineru_api", "v1", {"b": 2, "a": 1})  # 顺序不同
        assert s1.hash() == s2.hash()  # 与顺序无关

    def test_different_config_different_hash(self):
        from core.rag.parsing.signature import ParserSignature

        s1 = ParserSignature.build("mineru_api", "v1", {"model": "vlm"})
        s2 = ParserSignature.build("mineru_api", "v1", {"model": "pipeline"})
        assert s1.hash() != s2.hash()

    def test_token_not_in_signature(self):
        # api_token 不该进 signature（换 token 不让缓存失效）
        from core.rag.parsing.signature import ParserSignature

        s1 = ParserSignature.build("mineru_api", "v1", {"model": "vlm"})
        s2 = ParserSignature.build("mineru_api", "v1", {"model": "vlm"})
        assert s1.hash() == s2.hash()  # 无 token 字段，相同


# ── cache（tmp_path 真实文件系统）─────────────────────────────────────────────


class TestCache:
    def test_roundtrip(self, tmp_path):
        from core.rag.parsing import cache

        root = tmp_path / "parse_cache"
        sh, sig = "abc123def456ghij", "sig123sig456sige"
        wd = cache.reserve(root, sh, sig)
        assert not cache.is_ready(wd)  # 无 manifest

        (wd / "full.md").write_text("# T\ncontent", encoding="utf-8")
        (wd / "content_list.json").write_text(
            json.dumps([{"type": "title", "text": "T", "page_idx": 0}]), encoding="utf-8"
        )
        cache.write_manifest(wd, {"engine": "test"})
        assert cache.is_ready(wd)

        hit = cache.lookup(root, sh, sig)
        assert hit == wd
        md, blocks, asset = cache.load_ir(wd)
        assert "content" in md
        assert blocks and blocks[0]["type"] == "title"

    def test_miss_returns_none(self, tmp_path):
        from core.rag.parsing import cache

        assert cache.lookup(tmp_path, "nonexist1234567", "sig") is None

    def test_source_hash_by_bytes_not_name(self, tmp_path):
        from core.rag.parsing import cache

        a = tmp_path / "a.pdf"
        b = tmp_path / "different_name.pdf"
        a.write_bytes(b"same content")
        b.write_bytes(b"same content")
        # 同字节不同文件名 → 同 hash（重传命中缓存）
        assert cache.source_hash_from_path(a) == cache.source_hash_from_path(b)

    def test_cleanup_failed_removes_incomplete(self, tmp_path):
        from core.rag.parsing import cache

        wd = cache.reserve(tmp_path, "h1" * 8, "s1")
        (wd / "full.md").write_text("partial")  # 无 manifest
        assert not cache.is_ready(wd)
        cache.cleanup_failed(wd)
        assert not wd.exists()


# ── registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_mineru_engine(self):
        from core.rag.parsing.registry import get_engine
        from core.rag.parsing.engines.mineru_api import MinerUApiEngine

        eng = get_engine("mineru_api")
        assert isinstance(eng, MinerUApiEngine)

    def test_get_default_when_unknown(self):
        from core.rag.parsing.registry import DEFAULT_ENGINE, get_engine

        eng = get_engine("unknown_xyz")
        assert eng.name == DEFAULT_ENGINE  # 未知回退默认

    def test_is_engine_available(self):
        from core.rag.parsing.registry import is_engine_available

        assert is_engine_available("mineru_api") is True


# ── service.parse_document（mock engine）──────────────────────────────────────


class TestParseDocument:
    def _make_pdf(self, tmp_path) -> Path:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")
        return pdf

    def test_cache_miss_then_hit(self, tmp_path, monkeypatch):
        from core.rag.parsing import service
        from core.rag.parsing.signature import ParserSignature

        monkeypatch.setattr(service, "_cache_root", lambda: tmp_path / "cache")

        calls = {"parse": 0}

        class FakeEngine:
            name = "fake"

            def resolve_config(self):
                return {}

            def supported_formats(self):
                return frozenset({".pdf"})

            def signature(self, _c):
                return ParserSignature.build("fake", "v1", {})

            def is_ready(self, _c):
                return True, ""

            def parse(self, src, wd, *, config, on_output=None):
                calls["parse"] += 1
                (wd / "full.md").write_text("# T\nbody", encoding="utf-8")
                (wd / "fake_content_list.json").write_text(
                    json.dumps([{"type": "title", "text": "T", "page_idx": 0}]),
                    encoding="utf-8",
                )

        monkeypatch.setattr(service, "get_engine", lambda name=None: FakeEngine())

        pdf = self._make_pdf(tmp_path)
        doc1 = service.parse_document(pdf)
        assert calls["parse"] == 1
        assert doc1.markdown == "# T\nbody"
        assert doc1.engine == "fake"

        doc2 = service.parse_document(pdf)  # 第二次命中缓存，不重复 parse
        assert calls["parse"] == 1
        assert doc2.markdown == "# T\nbody"

    def test_unsupported_format_raises_no_fallback(self, tmp_path, monkeypatch):
        from core.rag.parsing import service
        from core.rag.parsing.signature import ParserSignature
        from core.rag.parsing.types import ParserError

        monkeypatch.setattr(service, "_cache_root", lambda: tmp_path / "cache")

        class FakeEngine:
            name = "fake"

            def resolve_config(self):
                return {}

            def supported_formats(self):
                return frozenset({".pdf"})

            def signature(self, _c):
                return ParserSignature.build("fake", "v1", {})

            def is_ready(self, _c):
                return True, ""

            def parse(self, *a, **k):
                raise AssertionError("不该调 parse（格式不支持应先报错）")

        monkeypatch.setattr(service, "get_engine", lambda name=None: FakeEngine())

        docx = tmp_path / "notpdf.docx"
        docx.write_bytes(b"fake docx")
        with pytest.raises(ParserError, match="不支持"):
            service.parse_document(docx)  # .docx 不在 supported_formats → 直接报错

    def test_not_ready_raises(self, tmp_path, monkeypatch):
        from core.rag.parsing import service
        from core.rag.parsing.signature import ParserSignature
        from core.rag.parsing.types import ParserError

        monkeypatch.setattr(service, "_cache_root", lambda: tmp_path / "cache")

        class FakeEngine:
            name = "fake"

            def resolve_config(self):
                return {}

            def supported_formats(self):
                return frozenset({".pdf"})

            def signature(self, _c):
                return ParserSignature.build("fake", "v1", {})

            def is_ready(self, _c):
                return False, "token 未配置"

            def parse(self, *a, **k):
                raise AssertionError("未就绪不该调 parse")

        monkeypatch.setattr(service, "get_engine", lambda name=None: FakeEngine())

        pdf = self._make_pdf(tmp_path)
        with pytest.raises(ParserError, match="token 未配置"):
            service.parse_document(pdf)


# ── MinerU 引擎配置（不调 parse / 不发 HTTP）──────────────────────────────────


class TestMinerUEngineConfig:
    def test_signature_excludes_token(self):
        from core.rag.parsing.engines.mineru_api import MinerUApiEngine

        eng = MinerUApiEngine()
        cfg_with_token = {
            "api_base_url": "https://mineru.net",
            "api_token": "secret1",
            "model_version": "vlm",
            "language": "ch",
            "enable_formula": True,
            "enable_table": True,
            "poll_interval": 5,
            "poll_timeout": 1800,
            "max_file_mb": 200,
        }
        cfg_diff_token = {**cfg_with_token, "api_token": "secret2"}
        # 换 token → 签名相同（token 不进 signature）
        assert eng.signature(cfg_with_token).hash() == eng.signature(cfg_diff_token).hash()

    def test_is_ready_no_token(self):
        from core.rag.parsing.engines.mineru_api import MinerUApiEngine

        eng = MinerUApiEngine()
        cfg = {
            "api_base_url": "https://mineru.net",
            "api_token": "",
            "model_version": "vlm",
            "language": "ch",
            "enable_formula": True,
            "enable_table": True,
            "poll_interval": 5,
            "poll_timeout": 1800,
            "max_file_mb": 200,
        }
        ok, reason = eng.is_ready(cfg)
        assert ok is False
        assert "token" in reason

    def test_is_ready_with_token(self):
        from core.rag.parsing.engines.mineru_api import MinerUApiEngine

        eng = MinerUApiEngine()
        cfg = {
            "api_base_url": "https://mineru.net",
            "api_token": "valid_token",
            "model_version": "vlm",
            "language": "ch",
            "enable_formula": True,
            "enable_table": True,
            "poll_interval": 5,
            "poll_timeout": 1800,
            "max_file_mb": 200,
        }
        ok, _ = eng.is_ready(cfg)
        assert ok is True

    def test_supported_formats_pdf_only(self):
        from core.rag.parsing.engines.mineru_api import MinerUApiEngine

        assert MinerUApiEngine().supported_formats() == frozenset({".pdf"})

    def test_parse_oversize_file_raises(self, tmp_path):
        from core.rag.parsing.engines.mineru_api import MinerUApiEngine, MinerUError

        eng = MinerUApiEngine()
        cfg = {
            "api_base_url": "https://mineru.net",
            "api_token": "x",
            "model_version": "vlm",
            "language": "ch",
            "enable_formula": True,
            "enable_table": True,
            "poll_interval": 5,
            "poll_timeout": 1800,
            "max_file_mb": 1,  # 1MB 上限
        }
        big = tmp_path / "big.pdf"
        big.write_bytes(b"0" * (2 * 1024 * 1024))  # 2MB > 1MB
        with pytest.raises(MinerUError, match="超过 MinerU 上限"):
            eng.parse(big, tmp_path / "wd", config=cfg)


# ── engine-ui 能力探测 ────────────────────────────────────────────────────────


class TestEngineUI:
    async def test_list_rag_engines(self):
        from api.admin import list_rag_engines

        with (
            patch("core.rag.registry.is_backend_available") as ba,
            patch("core.rag.parsing.registry.is_engine_available") as ea,
            patch("api.admin.get_settings") as gs,
        ):
            ba.side_effect = lambda b: (b == "lightrag", "" if b == "lightrag" else "依赖未装")
            ea.return_value = True
            gs.return_value.parsing.mineru_api_key.get_secret_value.return_value = "fake_key"

            result = await list_rag_engines(_={})

        assert len(result["index_backends"]) == 2
        ids = [b["id"] for b in result["index_backends"]]
        assert ids == ["lightrag", "llamaindex_pg"]
        # lightrag configured=True，llamaindex_pg configured=False（mock 返回 False）
        assert result["index_backends"][0]["configured"] is True
        assert result["index_backends"][1]["configured"] is False
        assert "reason" in result["index_backends"][1]  # 未配置带原因

        assert len(result["parse_engines"]) == 2
        # mineru_api configured 看 api_key（mock 配了 fake_key → True）
        assert result["parse_engines"][0]["configured"] is True
        assert result["parse_engines"][0]["requires_api_key"] is True


# ── MinerU 大 PDF 分片（超 max_file_pages 自动切→逐片解析→合并）──────────────


def _mineru_cfg(**overrides) -> dict:
    base = {
        "api_base_url": "https://mineru.net",
        "api_token": "x",
        "model_version": "vlm",
        "language": "ch",
        "enable_formula": True,
        "enable_table": True,
        "poll_interval": 5,
        "poll_timeout": 1800,
        "max_file_pages": 200,
        "max_file_mb": 200,
    }
    base.update(overrides)
    return base


class TestMinerUPdfSplit:
    """分片机制：纯 fitz 造真 PDF + mock _parse_one（不调真实 MinerU API）。"""

    @staticmethod
    def _make_pdf(path: Path, pages: int) -> Path:
        import fitz  # noqa: PLC0415

        doc = fitz.open()
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"page {i + 1}", fontsize=12)
        doc.save(str(path))
        doc.close()
        return path

    def test_count_pages_real_pdf(self, tmp_path):
        from core.rag.parsing.engines.mineru_api import _count_pages

        p = self._make_pdf(tmp_path / "a.pdf", 7)
        assert _count_pages(p) == 7

    def test_count_pages_invalid_returns_none(self, tmp_path):
        from core.rag.parsing.engines.mineru_api import _count_pages

        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf")
        assert _count_pages(bad) is None

    def test_split_ranges_cover_no_overlap(self, tmp_path):
        from core.rag.parsing.engines.mineru_api import _split_pdf

        import fitz  # noqa: PLC0415

        src = self._make_pdf(tmp_path / "src.pdf", 500)
        parts, tmp_dir = _split_pdf(src, 200, 500)
        try:
            assert len(parts) == 3
            # 区间覆盖 [0,500)，无重叠无空洞，每片 ≤ 阈值
            assert (parts[0][1], parts[0][2]) == (0, 200)
            assert (parts[1][1], parts[1][2]) == (200, 400)
            assert (parts[2][1], parts[2][2]) == (400, 500)
            for (part_path, _start, _end), expect_pages in zip(parts, [200, 200, 100]):
                d = fitz.open(str(part_path))
                assert d.page_count == expect_pages
                d.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_split_exact_multiple(self, tmp_path):
        from core.rag.parsing.engines.mineru_api import _split_pdf

        src = self._make_pdf(tmp_path / "src.pdf", 400)
        parts, _ = _split_pdf(src, 200, 400)
        assert len(parts) == 2
        assert parts[0][2] == 200 and parts[1][1] == 200 and parts[1][2] == 400

    def test_parse_under_threshold_single_call(self, tmp_path):
        from core.rag.parsing.engines import mineru_api as M

        eng = M.MinerUApiEngine()
        src = self._make_pdf(tmp_path / "src.pdf", 100)  # < 200 → 不拆
        wd = tmp_path / "wd"
        wd.mkdir()
        calls = []

        def fake_one(pdf_path, workdir, config, on_output=None, *, label=""):
            calls.append((Path(pdf_path).name, Path(workdir).name, label))
            (Path(workdir) / "full.md").write_text("# T", encoding="utf-8")

        with patch.object(M, "_parse_one", side_effect=fake_one):
            eng.parse(src, wd, config=_mineru_cfg(max_file_pages=200))

        assert len(calls) == 1
        assert calls[0][0] == "src.pdf"
        assert calls[0][1] == "wd"  # 直接用 workdir，无 _part 子目录
        assert calls[0][2] == ""  # 无分片标签
        assert not any(p.name.startswith("_part") for p in wd.iterdir())  # 无分片残留

    def test_parse_over_threshold_splits_and_merges(self, tmp_path):
        from core.rag.parsing.engines import mineru_api as M

        eng = M.MinerUApiEngine()
        src = self._make_pdf(tmp_path / "src.pdf", 450)  # > 200 → 3 片
        wd = tmp_path / "wd"
        wd.mkdir()

        def fake_one(pdf_path, workdir, config, on_output=None, *, label=""):
            wdir = Path(workdir)
            idx = int(wdir.name.replace("_part", "")) - 1  # _part1 → 0
            (wdir / "full.md").write_text(f"# Part {idx}\nbody{idx}", encoding="utf-8")
            (wdir / "content_list.json").write_text(
                json.dumps(
                    [
                        {"type": "title", "text": f"Part {idx}", "page_idx": 0},
                        {"type": "text", "text": "x", "page_idx": 5},
                    ]
                ),
                encoding="utf-8",
            )
            img_dir = wdir / "images"
            img_dir.mkdir()
            (img_dir / f"img{idx}.png").write_bytes(b"PNG" + bytes([idx]))

        with patch.object(M, "_parse_one", side_effect=fake_one):
            eng.parse(src, wd, config=_mineru_cfg(max_file_pages=200))

        # 合并 markdown = 三片按页序拼接
        merged_md = (wd / "full.md").read_text(encoding="utf-8")
        assert "# Part 0" in merged_md and "# Part 1" in merged_md and "# Part 2" in merged_md
        # page_idx 按片起始偏移（片0 +0 / 片1 +200 / 片2 +400）还原全局页码
        blocks = json.loads((wd / "content_list.json").read_text(encoding="utf-8"))
        assert len(blocks) == 6
        assert sorted(b["page_idx"] for b in blocks) == [0, 5, 200, 205, 400, 405]
        # 图片合并
        assert sorted(p.name for p in (wd / "images").iterdir()) == [
            "img0.png",
            "img1.png",
            "img2.png",
        ]
        # 分片子目录已清理
        assert not (wd / "_part1").exists()

    def test_parse_threshold_zero_never_splits(self, tmp_path):
        from core.rag.parsing.engines import mineru_api as M

        eng = M.MinerUApiEngine()
        src = self._make_pdf(tmp_path / "src.pdf", 500)  # 本该拆，但阈值=0 关闭分片
        wd = tmp_path / "wd"
        wd.mkdir()
        labels = []

        def fake_one(pdf_path, workdir, config, on_output=None, *, label=""):
            labels.append(label)
            (Path(workdir) / "full.md").write_text("x", encoding="utf-8")

        with patch.object(M, "_parse_one", side_effect=fake_one):
            eng.parse(src, wd, config=_mineru_cfg(max_file_pages=0))

        assert labels == [""]  # 单片直递，不分片
