"""回归：图片描述位置回填（DOCX 占位符 + 位置清单 + 孤儿 chunk 去重）。

验证 ingestion 的两个函数：
- _resolve_image_placeholders：把 [[IMG:sha16]] 占位符按 image_desc_by_blob 清单回填成
  [图: desc]；清单缺失降级为空串；fill=False 仅清理；返回已回填 sha16 集合供去重。
- _append_image_desc_chunks：读位置清单，把未内联的图片描述作为带【来源:】前缀的独立
  chunk 追加，source 用真实路径；已在 inlined 集合里的图跳过（去重）。

位置清单 image_desc_by_blob.json 与 image_desc_cache.json 同目录，结构：
    {blob_sha256: {"desc": str, "source": str}}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 补 backend 根到 sys.path，使 core.rag 可导入
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.rag.ingestion import (  # noqa: E402
    _append_image_desc_chunks,
    _resolve_image_placeholders,
)

# 两个真实的 64 位 sha（前 16 位即占位符里用的身份）
_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA16_A = _SHA_A[:16]
_SHA16_B = _SHA_B[:16]


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    """写位置清单，返回 img_cache 路径（image_desc_cache.json，作为目录锚点）。"""
    (tmp_path / "image_desc_by_blob.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8",
    )
    # img_cache 只是个路径锚点，_blob_manifest_path 取它的 .parent
    return tmp_path / "image_desc_cache.json"


# ── _resolve_image_placeholders ──────────────────────────────────────────────


def test_resolve_fills_placeholder_from_manifest(tmp_path):
    """清单有描述 → [[IMG:sha16]] 回填成 [图: desc]，sha16 进 inlined 集合。"""
    img_cache = _write_manifest(tmp_path, {
        _SHA_A: {"desc": "数字示波器面板", "source": "/docs/a.docx"},
    })
    chunks = [f"【章节: 实验三】\n电压测量。[[IMG:{_SHA16_A}]] 结束。"]

    resolved, inlined = _resolve_image_placeholders(chunks, img_cache)

    assert resolved == ["【章节: 实验三】\n电压测量。[图: 数字示波器面板] 结束。"]
    assert inlined == {_SHA16_A}


def test_resolve_missing_manifest_cleans_placeholder(tmp_path):
    """清单不存在 → 占位符移除（降级空串），inlined 为空，不抛异常。"""
    img_cache = tmp_path / "image_desc_cache.json"  # 不写 manifest
    chunks = [f"正文[[IMG:{_SHA16_A}]]尾部"]

    resolved, inlined = _resolve_image_placeholders(chunks, img_cache)

    assert resolved == ["正文尾部"]
    assert inlined == set()


def test_resolve_placeholder_not_in_manifest_cleaned(tmp_path):
    """清单存在但不含该图（如公式碎片被过滤）→ 占位符移除，不进 inlined。"""
    img_cache = _write_manifest(tmp_path, {
        _SHA_A: {"desc": "某图", "source": "/docs/a.docx"},
    })
    chunks = [f"[[IMG:{('c' * 64)[:16]}]]"]  # 不在清单里的 sha16

    resolved, inlined = _resolve_image_placeholders(chunks, img_cache)

    assert resolved == [""]
    assert inlined == set()


def test_resolve_fill_false_only_cleans(tmp_path):
    """fill=False（开关关）→ 即便清单有描述也只清理占位符，inlined 为空。"""
    img_cache = _write_manifest(tmp_path, {
        _SHA_A: {"desc": "数字示波器面板", "source": "/docs/a.docx"},
    })
    chunks = [f"[[IMG:{_SHA16_A}]]"]

    resolved, inlined = _resolve_image_placeholders(chunks, img_cache, fill=False)

    assert resolved == [""]
    assert inlined == set()


def test_resolve_word_caption_preserved(tmp_path):
    """占位符后紧跟 Word 原图注 → 回填描述、保留图注文本。"""
    img_cache = _write_manifest(tmp_path, {
        _SHA_A: {"desc": "面板布局", "source": "/docs/a.docx"},
    })
    chunks = [f"[[IMG:{_SHA16_A}]] 图3 示波器前面板"]

    resolved, inlined = _resolve_image_placeholders(chunks, img_cache)

    assert resolved == ["[图: 面板布局] 图3 示波器前面板"]
    assert inlined == {_SHA16_A}


def test_resolve_multiple_placeholders_one_chunk(tmp_path):
    """一个 chunk 多张图 → 全部回填，都进 inlined。"""
    img_cache = _write_manifest(tmp_path, {
        _SHA_A: {"desc": "图A", "source": "/docs/a.docx"},
        _SHA_B: {"desc": "图B", "source": "/docs/a.docx"},
    })
    chunks = [f"[[IMG:{_SHA16_A}]] 和 [[IMG:{_SHA16_B}]]"]

    resolved, inlined = _resolve_image_placeholders(chunks, img_cache)

    assert resolved == ["[图: 图A] 和 [图: 图B]"]
    assert inlined == {_SHA16_A, _SHA16_B}


# ── _append_image_desc_chunks ────────────────────────────────────────────────


def test_appends_orphan_with_source_prefix(tmp_path):
    """两条未内联描述 → 各追加为带【来源:】前缀的 chunk，source 真实路径::image-sha16。"""
    img_cache = _write_manifest(tmp_path, {
        _SHA_A: {"desc": "电路图，含 R1=750Ω", "source": "/data/电路实验.docx"},
        _SHA_B: {"desc": "面板布局图", "source": "/data/电路实验.docx"},
    })
    chunks = ["正文1"]
    sources = ["f::chunk-0"]

    added = _append_image_desc_chunks(chunks, sources, img_cache)

    assert added == 2
    assert len(chunks) == len(sources) == 3
    assert chunks[1] == "【来源: 电路实验.docx】\n【图片描述】\n电路图，含 R1=750Ω"
    assert chunks[2] == "【来源: 电路实验.docx】\n【图片描述】\n面板布局图"
    assert sources[1] == f"/data/电路实验.docx::image-{_SHA16_A}"
    assert sources[2] == f"/data/电路实验.docx::image-{_SHA16_B}"
    # 原始 chunk 不受影响
    assert chunks[0] == "正文1"


def test_skips_inlined_images(tmp_path):
    """已在 inlined 集合里的图（已内联进正文）不重复追加为孤儿 chunk。"""
    img_cache = _write_manifest(tmp_path, {
        _SHA_A: {"desc": "已内联图", "source": "/docs/a.docx"},
        _SHA_B: {"desc": "未内联图", "source": "/docs/a.docx"},
    })
    chunks: list[str] = []
    sources: list[str] = []

    added = _append_image_desc_chunks(chunks, sources, img_cache, inlined={_SHA16_A})

    assert added == 1
    assert len(chunks) == len(sources) == 1
    assert "未内联图" in chunks[0]
    assert sources[0] == f"/docs/a.docx::image-{_SHA16_B}"


def test_no_manifest_degrades_to_zero(tmp_path):
    """位置清单不存在 → 返回 0，chunks/sources 不变，不抛异常。"""
    img_cache = tmp_path / "image_desc_cache.json"  # 不写 manifest
    chunks = ["a"]
    sources = ["s"]
    added = _append_image_desc_chunks(chunks, sources, img_cache)
    assert added == 0
    assert chunks == ["a"]
    assert sources == ["s"]


def test_empty_desc_skipped(tmp_path):
    """空描述条目跳过，只追加有效的。"""
    img_cache = _write_manifest(tmp_path, {
        _SHA_A: {"desc": "有效描述", "source": "/docs/a.docx"},
        _SHA_B: {"desc": "   ", "source": "/docs/a.docx"},
    })
    chunks: list[str] = []
    sources: list[str] = []

    added = _append_image_desc_chunks(chunks, sources, img_cache)

    assert added == 1
    assert chunks == ["【来源: a.docx】\n【图片描述】\n有效描述"]


def test_corrupt_manifest_degrades(tmp_path):
    """位置清单损坏（非法 JSON）→ 降级返回 0，不抛异常。"""
    (tmp_path / "image_desc_by_blob.json").write_text("{ not valid json", encoding="utf-8")
    img_cache = tmp_path / "image_desc_cache.json"
    chunks = ["a"]
    sources = ["s"]
    added = _append_image_desc_chunks(chunks, sources, img_cache)
    assert added == 0
    assert chunks == ["a"]


def test_source_fallback_when_empty(tmp_path):
    """source 为空 → 无【来源:】前缀，source 退回 image_desc::image-sha16。"""
    img_cache = _write_manifest(tmp_path, {
        _SHA_A: {"desc": "无来源图", "source": ""},
    })
    chunks: list[str] = []
    sources: list[str] = []

    added = _append_image_desc_chunks(chunks, sources, img_cache)

    assert added == 1
    assert chunks[0] == "【图片描述】\n无来源图"  # 无前缀
    assert sources[0] == f"image_desc::image-{_SHA16_A}"
