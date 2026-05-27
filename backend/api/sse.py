from collections.abc import AsyncIterable

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from api.auth import get_current_user

router = APIRouter(prefix="/sse", tags=["sse"])


class Prompt(BaseModel):
    text: str


@router.post("/chat/stream", response_class=EventSourceResponse)
async def stream_chat(
    prompt: Prompt,
    user: dict = Depends(get_current_user),
) -> AsyncIterable[ServerSentEvent]:
    _ = user  # auth gate only
    words = prompt.text.split()
    for word in words:
        yield ServerSentEvent(data=word, event="token")
    yield ServerSentEvent(raw_data="[DONE]", event="done")