"""Feishu/Lark channel implementation using lark-oapi SDK with WebSocket long connection."""

import asyncio
import importlib.util
import json
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.bot.bus.events import OutboundMessage
from core.bot.bus.queue import MessageBus

from .base import BaseChannel

logger = logging.getLogger(__name__)

FEISHU_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None


class FeishuConfig(BaseModel):
    """Feishu/Lark channel configuration."""

    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    verification_token: str = ""
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    react_emoji: str = "THUMBSUP"
    group_policy: Literal["open", "mention"] = "mention"


def _extract_post_text(content_json: dict) -> str:
    """Extract plain text from Feishu post (rich text) message."""
    root = content_json
    if isinstance(root, dict) and isinstance(root.get("post"), dict):
        root = root["post"]
    if not isinstance(root, dict):
        return ""

    def _parse_block(block: dict) -> str:
        if not isinstance(block, dict) or not isinstance(block.get("content"), list):
            return ""
        texts = []
        if title := block.get("title"):
            texts.append(title)
        for row in block["content"]:
            if not isinstance(row, list):
                continue
            for el in row:
                if not isinstance(el, dict):
                    continue
                tag = el.get("tag")
                if tag in ("text", "a"):
                    texts.append(el.get("text", ""))
                elif tag == "at":
                    texts.append(f"@{el.get('user_name', 'user')}")
        return " ".join(texts).strip()

    if "content" in root:
        text = _parse_block(root)
        if text:
            return text

    for key in ("zh_cn", "en_us", "ja_jp"):
        if key in root:
            text = _parse_block(root[key])
            if text:
                return text
    for val in root.values():
        if isinstance(val, dict):
            text = _parse_block(val)
            if text:
                return text
    return ""


class FeishuChannel(BaseChannel):
    """Feishu/Lark channel using WebSocket long connection.

    Uses WebSocket to receive events — no public IP or webhook required.
    """

    name = "feishu"
    display_name = "Feishu"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return FeishuConfig().model_dump()

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = FeishuConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        if not FEISHU_AVAILABLE:
            logger.error("Feishu SDK not installed. Run: pip install lark-oapi")
            return

        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return

        import lark_oapi as lark

        self._running = True
        self._loop = asyncio.get_running_loop()

        self._client = (
            lark.Client.builder()
            .app_id(self.config.app_id)
            .app_secret(self.config.app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        event_handler = (
            lark.EventDispatcherHandler.builder(
                self.config.encrypt_key or "",
                self.config.verification_token or "",
            )
            .register_p2_im_message_receive_v1(self._on_message_sync)
            .build()
        )

        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        def run_ws():
            import time
            import lark_oapi.ws.client as _lark_ws_client

            ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(ws_loop)
            _lark_ws_client.loop = ws_loop
            try:
                while self._running:
                    try:
                        self._ws_client.start()
                    except Exception as e:
                        logger.warning("Feishu WebSocket error: %s", e)
                    if self._running:
                        time.sleep(5)
            finally:
                ws_loop.close()

        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()

        logger.info("Feishu bot started with WebSocket long connection")

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        logger.info("Feishu bot stopped")

    # --- Smart format detection ---
    _COMPLEX_MD_RE = re.compile(
        r"```"
        r"|^\|.+\|.*\n\s*\|[-:\s|]+\|"
        r"|^#{1,6}\s+",
        re.MULTILINE,
    )

    @classmethod
    def _detect_msg_format(cls, content: str) -> str:
        stripped = content.strip()
        if cls._COMPLEX_MD_RE.search(stripped):
            return "interactive"
        if len(stripped) > 2000:
            return "interactive"
        if len(stripped) <= 200:
            return "text"
        return "text"

    def _send_message_sync(
        self, receive_id_type: str, receive_id: str, msg_type: str, content: str
    ) -> bool:
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        try:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type(receive_id_type)
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(receive_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                )
                .build()
            )
            response = self._client.im.v1.message.create(request)
            if not response.success():
                logger.error(
                    "Failed to send Feishu %s message: code=%s, msg=%s",
                    msg_type, response.code, response.msg,
                )
                return False
            return True
        except Exception as e:
            logger.error("Error sending Feishu %s message: %s", msg_type, e)
            return False

    async def send(self, msg: OutboundMessage) -> None:
        if not self._client:
            logger.warning("Feishu client not initialized")
            return

        try:
            receive_id_type = "chat_id" if msg.chat_id.startswith("oc_") else "open_id"
            loop = asyncio.get_running_loop()

            if msg.content and msg.content.strip():
                fmt = self._detect_msg_format(msg.content)

                if fmt == "text":
                    text_body = json.dumps({"text": msg.content.strip()}, ensure_ascii=False)
                    await loop.run_in_executor(
                        None, self._send_message_sync,
                        receive_id_type, msg.chat_id, "text", text_body,
                    )
                else:
                    card = {
                        "config": {"wide_screen_mode": True},
                        "elements": [{"tag": "markdown", "content": msg.content}],
                    }
                    await loop.run_in_executor(
                        None, self._send_message_sync,
                        receive_id_type, msg.chat_id, "interactive",
                        json.dumps(card, ensure_ascii=False),
                    )
        except Exception as e:
            logger.error("Error sending Feishu message: %s", e)

    def _on_message_sync(self, data: Any) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    async def _on_message(self, data: Any) -> None:
        try:
            event = data.event
            message = event.message
            sender = event.sender

            message_id = message.message_id
            if message_id in self._processed_message_ids:
                return
            self._processed_message_ids[message_id] = None

            while len(self._processed_message_ids) > 1000:
                self._processed_message_ids.popitem(last=False)

            if sender.sender_type == "bot":
                return

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_id = message.chat_id
            chat_type = message.chat_type
            msg_type = message.message_type

            if chat_type == "group" and self.config.group_policy == "mention":
                raw_content = message.content or ""
                if "@_all" not in raw_content:
                    has_bot_mention = False
                    for mention in getattr(message, "mentions", None) or []:
                        mid = getattr(mention, "id", None)
                        if not mid:
                            continue
                        if not getattr(mid, "user_id", None) and (
                            getattr(mid, "open_id", None) or ""
                        ).startswith("ou_"):
                            has_bot_mention = True
                            break
                    if not has_bot_mention:
                        return

            try:
                content_json = json.loads(message.content) if message.content else {}
            except json.JSONDecodeError:
                content_json = {}

            content = ""
            if msg_type == "text":
                content = content_json.get("text", "")
            elif msg_type == "post":
                content = _extract_post_text(content_json)
            else:
                content = f"[{msg_type}]"

            if not content:
                return

            reply_to = chat_id if chat_type == "group" else sender_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                metadata={"message_id": message_id, "chat_type": chat_type, "msg_type": msg_type},
            )
        except Exception as e:
            logger.error("Error processing Feishu message: %s", e)
