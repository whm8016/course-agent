"""出题输出目录（替代 path_service.get_question_dir）。"""

from __future__ import annotations

from pathlib import Path

from config import QUESTION_LOG_DIR


def get_question_dir() -> Path:
    p = Path(QUESTION_LOG_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p