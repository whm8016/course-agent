"""Path helpers for TutorBot workspace."""

import os
from pathlib import Path


def get_bot_workspace_root() -> Path:
    from config import BASE_DIR
    default = os.path.join(BASE_DIR, "data", "tutorbot")
    return Path(os.getenv("TUTORBOT_WORKSPACE_DIR", default))


def get_media_dir(channel: str = "") -> Path:
    media = get_bot_workspace_root() / "media"
    if channel:
        media = media / channel
    media.mkdir(parents=True, exist_ok=True)
    return media
