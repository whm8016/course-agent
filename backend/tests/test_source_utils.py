"""source_utils 单测：来源前缀的解析/剥离 + chunk 后缀剥离。

parse_source_prefix 与 ingestion._build_source_prefix 成对（构建↔解析），
strip_source_prefix 是其降级版（只取 body 丢 section）。纯函数，离线直测。
"""
from core.rag.source_utils import (
    parse_source_prefix,
    strip_chunk_suffix,
    strip_source_prefix,
)


# ── parse_source_prefix（与 ingestion._build_source_prefix 成对）──────────────


def test_parse_chapter_with_page():
    ck = "【章节: 实验三 > 串联电路 | 第3页】\n戴维南等效是把复杂电路…"
    assert parse_source_prefix(ck) == ("实验三 > 串联电路", "戴维南等效是把复杂电路…")


def test_parse_chapter_no_page():
    ck = "【章节: 基尔霍夫定律】\n节点电流代数和为零…"
    assert parse_source_prefix(ck) == ("基尔霍夫定律", "节点电流代数和为零…")


def test_parse_degraded_source_filename():
    # PDF/TXT/MD 无章节 → 退化用文件名作 section 分组键（ingestion._build_source_prefix elif 分支）
    ck = "【来源: chapter5.pdf】\n正文内容"
    assert parse_source_prefix(ck) == ("chapter5.pdf", "正文内容")


def test_parse_degraded_source_with_page():
    ck = "【来源: intro.pdf | 第2页】\n正文"
    assert parse_source_prefix(ck) == ("intro.pdf", "正文")


def test_parse_no_prefix_returns_empty_section():
    ck = "没有前缀的纯正文 chunk"
    assert parse_source_prefix(ck) == ("", ck)


def test_parse_body_equals_strip():
    # parse 的 body 必须与 strip_source_prefix 严格一致（同一 _SOURCE_PREFIX_RE）
    ck = "【章节: 叠加定理 | 第7页】\n独立源单独作用…"
    assert parse_source_prefix(ck)[1] == strip_source_prefix(ck)


# ── strip_* 回归（已有行为不动）─────────────────────────────────────────────


def test_strip_source_prefix_noop_without_prefix():
    assert strip_source_prefix("无前缀") == "无前缀"


def test_strip_source_prefix_strips():
    assert strip_source_prefix("【章节: a】\n正文") == "正文"


def test_strip_chunk_suffix():
    assert strip_chunk_suffix("a.pdf::chunk-3") == "a.pdf"
    assert strip_chunk_suffix("a.pdf") == "a.pdf"
