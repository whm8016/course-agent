"""Configuration schema for TutorBot channels and tools."""

from pydantic import BaseModel, ConfigDict, Field


class ChannelsConfig(BaseModel):
    """Configuration for chat channels. Extra fields hold per-channel configs."""

    model_config = ConfigDict(extra="allow")

    send_progress: bool = True
    send_tool_hints: bool = False
