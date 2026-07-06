"""Path helpers for TutorBot workspace."""

from pathlib import Path


def get_bot_workspace_root() -> Path:
    from settings import get_settings
    TUTORBOT_WORKSPACE_DIR = get_settings().paths.tutorbot_workspace_dir
    return Path(TUTORBOT_WORKSPACE_DIR)


def get_media_dir(channel: str = "") -> Path:
    media = get_bot_workspace_root() / "media"
    if channel:
        media = media / channel
    media.mkdir(parents=True, exist_ok=True)
    return media
