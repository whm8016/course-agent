"""markdown 路线解析单测（parsing/markdown_sections.py + cache.load_ir v2 排除）。

DeepTutor 式「解析归解析、切块归切块」：MinerU full.md → MarkdownNodeParser → sections。
纯函数 + tmp_path 真实文件系统——不调真实 VLM（fallback 用 monkeypatch / desc_cache 命中）。

验证：
- markdown_to_sections 按标题分节、叶子标题正确、参考文献过滤、页码回填
- resolve_image_refs：有图注删 marker 留图注；无图注查 desc_cache / fallback 文件名
- cache.load_ir 显式排除 content_list_v2（v2 是 list[list[dict]]，误选会 blocks 非空但形状错）
"""
from __future__ import annotations

import hashlib
import json


# ── markdown_to_sections ──────────────────────────────────────────────────────


class TestMarkdownToSections:
    def test_basic_sectioning_and_leaf_titles(self):
        from core.rag.parsing.markdown_sections import markdown_to_sections

        md = (
            "# My Paper Title\nauthors\n\n"
            "## Abstract\nsome abstract\n\n"
            "## Methods\nwe do X\n\n"
            "### Sub-method\nmore detail\n"
        )
        secs = markdown_to_sections(md, drop_refs=False)
        # 4 节：title / Abstract / Methods / Sub-method
        assert [s["title"] for s in secs] == [
            "My Paper Title",
            "Abstract",
            "Methods",
            "Sub-method",
        ]
        # 内容首行保留叶子标题（MarkdownNodeParser 行为）
        assert secs[1]["content"].startswith("## Abstract")
        assert "some abstract" in secs[1]["content"]
        # 形状与 extract_pdf_sections 同构
        for s in secs:
            assert set(s.keys()) == {"title", "content", "page"}

    def test_drop_references_filters_references_section(self):
        from core.rag.parsing.markdown_sections import markdown_to_sections

        md = (
            "# Paper\nbody\n\n"
            "## Introduction\nintro\n\n"
            "## References\n[1] Foo, 2023\n[2] Bar, 2024\n"
        )
        kept = [s["title"] for s in markdown_to_sections(md, drop_refs=True)]
        assert "References" not in kept
        assert "Introduction" in kept
        # drop_refs=False 保留
        all_titles = [s["title"] for s in markdown_to_sections(md, drop_refs=False)]
        assert "References" in all_titles

    def test_drop_references_chinese_and_bibliography(self):
        from core.rag.parsing.markdown_sections import _is_reference_section

        assert _is_reference_section("References")
        assert _is_reference_section("参考文献")
        assert _is_reference_section("Bibliography")
        assert _is_reference_section("Works Cited")
        # 非参考文献章节不误杀
        assert not _is_reference_section("Introduction")
        assert not _is_reference_section("Related Work")
        assert not _is_reference_section("")

    def test_page_backfill_from_text_level_blocks(self):
        from core.rag.parsing.markdown_sections import markdown_to_sections

        md = "# Title\ntitle body\n\n## Intro\nintro body\n\n## Methods\nm body\n"
        # MinerU 标题是 type=text + text_level（非 type=title）
        blocks = [
            {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "Intro", "text_level": 2, "page_idx": 3},
            {"type": "text", "text": "Methods", "text_level": 2, "page_idx": 7},
        ]
        secs = {s["title"]: s["page"] for s in markdown_to_sections(md, blocks=blocks)}
        assert secs["Title"] == 0
        assert secs["Intro"] == 3
        assert secs["Methods"] == 7

    def test_page_falls_back_to_zero_when_blocks_missing_heading(self):
        from core.rag.parsing.markdown_sections import markdown_to_sections

        md = "# A\nx\n\n## B\ny\n"
        # blocks 无 "B" 标题 → B 的页码回填 0（不报错）
        blocks = [{"type": "text", "text": "A", "text_level": 1, "page_idx": 2}]
        secs = {s["title"]: s["page"] for s in markdown_to_sections(md, blocks=blocks)}
        assert secs["A"] == 2
        assert secs["B"] == 0

    def test_empty_markdown_returns_empty(self):
        from core.rag.parsing.markdown_sections import markdown_to_sections

        assert markdown_to_sections("") == []
        assert markdown_to_sections("   \n  ") == []

    def test_docling_title_blocks_also_indexed(self):
        """docling 产 type=title（无 text_level）也应进页码查找表。"""
        from core.rag.parsing.markdown_sections import _heading_page_index

        blocks = [
            {"type": "title", "text": "Intro", "page_idx": 1},
            {"type": "title", "text": "Methods", "page_idx": 4},
            {"type": "text", "text": "正文段落", "page_idx": 1},  # 非标题，跳过
        ]
        assert _heading_page_index(blocks) == {"Intro": 1, "Methods": 4}


# ── resolve_image_refs ────────────────────────────────────────────────────────


class TestResolveImageRefs:
    def test_marker_deleted_when_caption_present(self):
        from core.rag.parsing.markdown_sections import resolve_image_refs

        md = (
            "## Figure section\n\n"
            "![](images/abc123.jpg)\n\n"
            "Figure 1: An illustration of the method.\n\n"
            "Body text continues.\n"
        )
        out = resolve_image_refs(md, asset_dir=None)
        assert "![](images/" not in out
        assert "Figure 1: An illustration of the method." in out  # 图注保留
        assert "Body text continues." in out

    def test_caption_variants_recognized(self):
        from core.rag.parsing.markdown_sections import resolve_image_refs

        for caption in ("Figure 2: x", "Fig. 3: y", "Table 1: z", "图 1 流程", "表 2 数据"):
            md = f"![](images/x.jpg)\n\n{caption}\n"
            assert "![](images/" not in resolve_image_refs(md, asset_dir=None)

    def test_fallback_filename_when_no_caption_no_asset(self, monkeypatch):
        from core.rag.parsing.markdown_sections import resolve_image_refs

        # 无 asset_dir 且无图注 → [图: 文件名 stem]，绝不调 VLM
        md = "before\n\n![](images/abc123.jpg)\n\nafter\n"
        out = resolve_image_refs(md, asset_dir=None)
        assert "[图: abc123]" in out
        assert "![](images/" not in out
        assert "before" in out and "after" in out

    def test_fallback_uses_desc_cache_hit(self, tmp_path):
        """无图注 + asset_dir 内 desc_cache 命中 → 用缓存描述，不调 VLM。"""
        from core.rag.parsing.markdown_sections import resolve_image_refs

        images = tmp_path / "images"
        images.mkdir()
        img_bytes = b"\x89PNG\r\n\x1a\nfake-image-bytes"
        (images / "pic.jpg").write_bytes(img_bytes)
        # 按 image_extractor 缓存格式：key = sha256(图片字节)
        key = hashlib.sha256(img_bytes).hexdigest()
        (images / "desc_cache.json").write_text(
            json.dumps({key: "一张电路示意图"}), encoding="utf-8"
        )

        md = "![](images/pic.jpg)\n"  # 无图注
        out = resolve_image_refs(md, asset_dir=images)
        assert "[图: 一张电路示意图]" in out

    def test_fallback_filename_when_vlm_unavailable(self, tmp_path, monkeypatch):
        """无图注 + desc_cache 未命中 + VLM 不可用 → 降级文件名，不抛异常。"""
        from core.rag.parsing import markdown_sections

        monkeypatch.setattr(markdown_sections, "_vlm_caption_sync", lambda _b: "")
        images = tmp_path / "images"
        images.mkdir()
        (images / "pic.jpg").write_bytes(b"fake")

        md = "![](images/pic.jpg)\n"
        out = markdown_sections.resolve_image_refs(md, asset_dir=images)
        assert "[图: pic]" in out

    def test_no_image_markers_returns_unchanged(self):
        from core.rag.parsing.markdown_sections import resolve_image_refs

        md = "## Intro\nno images here\n"
        assert resolve_image_refs(md, asset_dir=None) == md

    # ── vlm_always 模式（image_vlm_always opt-in）──────────────────────────────

    def test_default_false_keeps_caption_no_vlm(self, tmp_path):
        """vlm_always=False（默认）有图注 → 删 marker 留图注，行为零变化（回归护栏）。"""
        from core.rag.parsing.markdown_sections import resolve_image_refs

        images = tmp_path / "images"
        images.mkdir()
        (images / "abc.jpg").write_bytes(b"fake")  # 有 asset 也不该被调

        md = "![](images/abc.jpg)\n\n图 1 股权架构图\n"
        out = resolve_image_refs(md, asset_dir=images, vlm_always=False)
        assert "![](images/" not in out
        assert "图 1 股权架构图" in out
        assert "[图:" not in out  # 有图注 + 非 always → 不生成描述

    def test_always_mode_cache_hit_keeps_caption(self, tmp_path):
        """vlm_always=True + desc_cache 命中 → [图: 描述] 与图注并存、marker 消失。"""
        from core.rag.parsing.markdown_sections import resolve_image_refs

        images = tmp_path / "images"
        images.mkdir()
        img_bytes = b"\x89PNG\r\n\x1a\nfake-image-bytes"
        (images / "pic.jpg").write_bytes(img_bytes)
        key = hashlib.sha256(img_bytes).hexdigest()
        (images / "desc_cache.json").write_text(
            json.dumps({key: "龙女士持股 49%，虎公司持股 51%"}), encoding="utf-8"
        )

        md = "![](images/pic.jpg)\n\n图 1-1 神兽公司股权架构图\n"
        out = resolve_image_refs(md, asset_dir=images, vlm_always=True)
        assert "[图: 龙女士持股 49%，虎公司持股 51%]" in out  # 描述在前
        assert "图 1-1 神兽公司股权架构图" in out  # 图注仍保留
        assert "![](images/" not in out

    def test_always_mode_vlm_unavailable_keeps_caption_only(self, tmp_path, monkeypatch):
        """vlm_always=True + VLM 不可用 + 有图注 → 只删 marker 保留图注，不产生 [图: 噪声。"""
        from core.rag.parsing import markdown_sections

        # VLM 不可用：预取与单点 fallback 都返回空（不调真实 VLM）
        monkeypatch.setattr(markdown_sections, "_vlm_caption_sync", lambda _b: "")

        def _fake_factory(_cache, _lock):
            async def _fake_caption(*_a, **_kw):
                return ""

            return _fake_caption

        monkeypatch.setattr(
            "core.rag.llamaindex.image_extractor._make_vision_caption_func",
            _fake_factory,
        )

        images = tmp_path / "images"
        images.mkdir()
        (images / "pic.jpg").write_bytes(b"fake-bytes")

        md = "![](images/pic.jpg)\n\n图 1-1 股权架构图\n"
        out = markdown_sections.resolve_image_refs(md, asset_dir=images, vlm_always=True)
        assert "![](images/" not in out
        assert "图 1-1 股权架构图" in out  # 图注保留
        assert "[图:" not in out  # VLM 失败不落地噪声

    def test_prefetch_dedup_by_byte_hash(self, tmp_path, monkeypatch):
        """预取按字节 sha256 去重：同内容两个文件名只触发一次 VLM 调用。"""
        from core.rag.parsing import markdown_sections

        images = tmp_path / "images"
        images.mkdir()
        content = b"identical-image-bytes"
        (images / "a.jpg").write_bytes(content)
        (images / "b.jpg").write_bytes(content)  # 同内容不同名

        calls = {"n": 0}

        def _counting_factory(_cache, _lock):
            async def _caption(prompt, *, image_data=None, **_kw):
                calls["n"] += 1
                return "描述"

            return _caption

        monkeypatch.setattr(
            "core.rag.llamaindex.image_extractor._make_vision_caption_func",
            _counting_factory,
        )

        desc_cache: dict[str, str] = {}
        wrote = markdown_sections._prefetch_descriptions(["a.jpg", "b.jpg"], images, desc_cache)
        assert wrote is True
        assert calls["n"] == 1  # 同字节 hash 只调一次
        assert len(desc_cache) == 1


# ── cache.load_ir 排除 content_list_v2 ────────────────────────────────────────


class TestLoadIrExcludesV2:
    def test_prefers_v1_over_v2(self, tmp_path):
        from core.rag.parsing import cache

        (tmp_path / "full.md").write_text("# T\nbody", encoding="utf-8")
        (tmp_path / "stem_content_list.json").write_text(
            json.dumps([{"type": "text", "text": "v1-flat", "page_idx": 0}]),
            encoding="utf-8",
        )
        # v2 是 list[list[dict]]——误选会让 blocks 形状错误
        (tmp_path / "stem_content_list_v2.json").write_text(
            json.dumps([[{"type": "text", "text": "v2-nested"}]]),
            encoding="utf-8",
        )

        _md, blocks, _asset = cache.load_ir(tmp_path)
        assert blocks == [{"type": "text", "text": "v1-flat", "page_idx": 0}]

    def test_v2_only_yields_no_blocks(self, tmp_path):
        """只有 v2 文件时显式排除 → blocks=None（退化为 markdown 兜底），不误读 v2。"""
        from core.rag.parsing import cache

        (tmp_path / "full.md").write_text("# T\nbody", encoding="utf-8")
        (tmp_path / "stem_content_list_v2.json").write_text(
            json.dumps([[{"type": "text", "text": "v2"}]]),
            encoding="utf-8",
        )

        _md, blocks, _asset = cache.load_ir(tmp_path)
        assert blocks is None
