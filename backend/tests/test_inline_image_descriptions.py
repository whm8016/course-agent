"""回归：图片描述回填 chunk（Phase 4，C 方案）。

验证 ingestion._append_image_desc_chunks：
- enabled 时把 desc_cache 每条描述包成 `【图片描述】\\n{desc}` 独立 chunk 追加，
  sources 配对 `image_desc::img-{i}`，chunks/sources 严格等长。
- 空/纯空白描述跳过，只追加有效项。
- desc_cache 不存在 / 损坏 → 降级返回 0，不抛异常（绝不阻断索引）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 补 backend 根到 sys.path，使 core.rag 可导入
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.rag.ingestion import _append_image_desc_chunks  # noqa: E402


def test_appends_image_desc_chunks_from_cache(tmp_path):
    """desc_cache 有 2 条描述 → 追加 2 个 chunk，sources 配对 image_desc::img-*。"""
    cache = tmp_path / "image_desc_cache.json"
    cache.write_text(json.dumps({
        "hash1": "一张电路图，包含电阻 R1=750Ω",
        "hash2": "实验面板布局图",
    }, ensure_ascii=False), encoding="utf-8")

    chunks = ["正文chunk1", "正文chunk2"]
    sources = ["f::chunk-0", "f::chunk-1"]

    added = _append_image_desc_chunks(chunks, sources, cache)

    assert added == 2
    assert len(chunks) == len(sources) == 4  # 配对等长
    assert chunks[2] == "【图片描述】\n一张电路图，包含电阻 R1=750Ω"
    assert chunks[3] == "【图片描述】\n实验面板布局图"
    assert sources[2] == "image_desc::img-0"
    assert sources[3] == "image_desc::img-1"
    # 原始 chunk 不受影响
    assert chunks[0] == "正文chunk1"


def test_no_cache_degrades_to_zero(tmp_path):
    """desc_cache 不存在 → 返回 0，chunks/sources 不变，不抛异常。"""
    missing = tmp_path / "nope.json"
    chunks = ["a"]
    sources = ["s"]
    added = _append_image_desc_chunks(chunks, sources, missing)
    assert added == 0
    assert chunks == ["a"]
    assert sources == ["s"]


def test_empty_or_blank_descs_skipped(tmp_path):
    """空字符串/纯空白描述跳过，只追加有效描述，added 编号只对有效项递增。"""
    cache = tmp_path / "image_desc_cache.json"
    cache.write_text(json.dumps({
        "h1": "有效描述",
        "h2": "",
        "h3": "   \n  ",
    }, ensure_ascii=False), encoding="utf-8")

    chunks: list[str] = []
    sources: list[str] = []
    added = _append_image_desc_chunks(chunks, sources, cache)

    assert added == 1
    assert len(chunks) == len(sources) == 1
    assert chunks[0] == "【图片描述】\n有效描述"
    assert sources[0] == "image_desc::img-0"


def test_corrupt_cache_degrades(tmp_path):
    """desc_cache 损坏（非法 JSON）→ 降级返回 0，不抛异常。"""
    cache = tmp_path / "image_desc_cache.json"
    cache.write_text("{ not valid json", encoding="utf-8")
    chunks = ["a"]
    sources = ["s"]
    added = _append_image_desc_chunks(chunks, sources, cache)
    assert added == 0
    assert chunks == ["a"]
