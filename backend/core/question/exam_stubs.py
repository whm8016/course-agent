"""
仿卷试卷解析：未接入 MinerU 时占位。仅 generate_from_topic 可不依赖本模块。
"""

from __future__ import annotations


def parse_pdf_with_mineru(pdf_path: str, output_dir: str) -> bool:
    raise RuntimeError(
        "parse_pdf_with_mineru 未实现：需接入 MinerU 或自研 PDF 解析。"
        "请仅使用 generate_from_topic（按知识点出题）。"
    )


def extract_questions_from_paper(working_dir: str, output_dir: str | None = None) -> bool:
    raise RuntimeError(
        "extract_questions_from_paper 未实现。"
        "请仅使用 generate_from_topic（按知识点出题）。"
    )