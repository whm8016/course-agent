"""
仿卷试卷解析：未接入 MinerU 时占位。仿卷（mimic）路径会调用本模块，
需接入 MinerU 或自研 PDF 解析后方可使用；按知识点出题走出题能力 WS /api/run/quiz。
"""

from __future__ import annotations


def parse_pdf_with_mineru(pdf_path: str, output_dir: str) -> bool:
    raise RuntimeError(
        "parse_pdf_with_mineru 未实现：需接入 MinerU 或自研 PDF 解析。"
        "请改用按知识点出题（出题能力 WS /api/run/quiz）。"
    )


def extract_questions_from_paper(working_dir: str, output_dir: str | None = None) -> bool:
    raise RuntimeError(
        "extract_questions_from_paper 未实现。"
        "请改用按知识点出题（出题能力 WS /api/run/quiz）。"
    )