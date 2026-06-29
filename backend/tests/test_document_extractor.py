"""utils/document_extractor.py 单测。

重点验证 ``_extract_text_like`` 的多编码解码（B1：原实现误调不存在的
``FileTypeRouter.decode_bytes`` 致 NameError，已改为本地多编码 fallback），以及
``extract_documents_from_records`` 清 base64 的 DeepTutor 铁律。
"""
from __future__ import annotations

import base64

from utils.document_extractor import extract_documents_from_records, extract_text_from_bytes


def test_extract_text_utf8():
    assert extract_text_from_bytes("a.txt", "hello 快速排序".encode("utf-8")) == "hello 快速排序"


def test_extract_text_utf8_bom_stripped():
    # UTF-8 BOM 应被 utf-8-sig 解码器吃掉
    data = b"\xef\xbb\xbf" + "标题".encode("utf-8")
    assert extract_text_from_bytes("note.md", data) == "标题"


def test_extract_text_gbk():
    # GBK 编码的源码文件应能解码（不抛 UnicodeDecodeError / NameError）
    assert extract_text_from_bytes("code.py", "print('中文')".encode("gbk")) == "print('中文')"


def test_extract_text_ascii():
    assert extract_text_from_bytes("a.txt", b"plain ascii") == "plain ascii"


def test_extract_documents_from_records_clears_base64():
    b64 = base64.b64encode("文档内容".encode("utf-8")).decode()
    records = [{"type": "file", "filename": "doc.txt", "base64": b64}]
    doc_texts, updated = extract_documents_from_records(records)
    assert len(doc_texts) == 1
    assert doc_texts[0].startswith("[File: doc.txt]")
    assert "文档内容" in doc_texts[0]
    # base64 必须被清空（DeepTutor 铁律：持久化省 DB 空间 + 不泄露原文字节）
    assert updated[0]["base64"] == ""


def test_extract_documents_from_records_image_unchanged():
    # 图片/非文档 record 原样返回（不解析、不清 base64）
    records = [{"type": "image", "filename": "a.png", "base64": "xyz"}]
    doc_texts, updated = extract_documents_from_records(records)
    assert doc_texts == []
    assert updated[0]["base64"] == "xyz"


def test_extract_documents_from_records_no_base64_skipped():
    # 文档 record 但无 base64 → 跳过（不炸），doc_texts 不含它
    records = [{"type": "file", "filename": "doc.txt", "base64": ""}]
    doc_texts, updated = extract_documents_from_records(records)
    assert doc_texts == []
    assert updated[0]["base64"] == ""
