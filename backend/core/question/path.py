"""出题输出目录（替代 path_service.get_question_dir）。"""

from __future__ import annotations

from pathlib import Path

from settings import get_settings
QUESTION_LOG_DIR = get_settings().paths.question_log_dir


def get_question_dir() -> Path:
    p = Path(QUESTION_LOG_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p