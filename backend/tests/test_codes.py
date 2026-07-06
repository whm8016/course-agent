"""core/codes.py 的纯函数单测（无 DB 依赖）。"""
from __future__ import annotations

import pytest

from core.codes import ALPHABET, DEFAULT_LENGTH, format_code, generate_code, normalize_code


# ── generate_code ──────────────────────────────────────────────────────────
def test_generate_default_length_and_alphabet():
    code = generate_code()
    assert len(code) == DEFAULT_LENGTH
    assert all(ch in ALPHABET for ch in code)


def test_generate_custom_length():
    assert len(generate_code(12)) == 12


def test_generate_rejects_too_short():
    with pytest.raises(ValueError):
        generate_code(3)


def test_generate_excludes_ambiguous_glyphs():
    """生成的码绝不包含 0/O/1/I/L 等易混字形。"""
    for _ in range(500):
        code = generate_code()
        for ch in "01OIL":
            assert ch not in code


def test_generate_uniqueness_sample():
    """大样本下碰撞概率极低（纵深防御层面的 sanity check）。"""
    codes = {generate_code() for _ in range(2000)}
    assert len(codes) > 1990


# ── normalize_code ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "raw, expected",
    [
        # 新码：用户带连字符/空格/小写
        ("ab2k-9m xq", "AB2K9MXQ"),
        # 旧 hex 码（含 0/1），lossless 保留
        ("a1b2 0f3c", "A1B20F3C"),
        # 换行/Tab
        ("AB\n2K\t-9M", "AB2K9M"),
        # 邀请码旧 hex 大写
        ("1a2b3c4d", "1A2B3C4D"),
        # 空值
        ("", ""),
        (None, ""),
        ("   ", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize_code(raw) == expected


def test_normalize_idempotent():
    raw = "ab2k-9mxq"
    once = normalize_code(raw)
    twice = normalize_code(once)
    assert once == twice


# ── format_code ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ab2k9mxq", "AB2K-9MXQ"),
        ("AB2K-9MXQ", "AB2K-9MXQ"),  # 已分组再 format 不重复加连字符
        ("a1b20f3c", "A1B2-0F3C"),  # 旧 hex 含 0，分组展示无碍
        ("ab2k9m", "AB2K9M"),  # 非默认长度：原样大写，不强行切分
        ("", ""),
        (None, ""),
    ],
)
def test_format(raw, expected):
    assert format_code(raw) == expected


# ── 向后兼容回归 ────────────────────────────────────────────────────────────
def test_backward_compat_legacy_hex_join_code():
    """旧 hex 码归一化后仍等于库内原始存储值，DB 查询可命中（零迁移兼容核心）。"""
    legacy_stored = "1a2b3c4d"  # 假设库里存的是这个（小写 hex）
    # 学生各种花式输入都应归一化到同一个值
    assert normalize_code("1A2B-3C4D") == legacy_stored.upper()
    assert normalize_code(" 1a2b 3c4d ") == legacy_stored.upper()
    assert normalize_code("1A2B3C4D") == legacy_stored.upper()
