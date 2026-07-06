"""
Multimodal Message Utilities
=============================

Converts plain-text messages + image attachments into the multimodal
message format expected by vision-capable LLMs.

Supports:
- OpenAI-compatible API (content array with image_url blocks)
- Anthropic API (content array with image source blocks)

对标 DeepTutor services/llm/multimodal.py 的两步式注入：
- Stage-1（本模块，``prepare_multimodal_messages``）：乐观注入图片，**不查
  supports_vision**。优先用已有 ``base64`` 内联 data URL；没有 base64 但有
  ``file_path`` 时读盘兜底（对标 DeepTutor 本地解析，也让 ``from_image_path``
  路径无需调用方预填 base64）；仅外部 http(s) URL 才以 URL 形式发送。
- Stage-2（``core/llm/llm.py::_create_with_image_fallback``）：调用失败时若
  :func:`is_image_input_unsupported` 且 :func:`should_degrade_to_text`，剥图
  用**同一模型**重试纯文本。
"""

from __future__ import annotations

import base64 as _b64
import logging
from pathlib import Path
from typing import Any

from .capabilities import supports_vision

logger = logging.getLogger(__name__)

MIME_FALLBACK = "image/png"


def _guess_mime_type(filename: str | None, fallback: str = MIME_FALLBACK) -> str:
    filename = filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
    }.get(ext, fallback)


def _build_openai_image_part(
    *,
    base64_data: str,
    mime_type: str,
    url: str = "",
) -> dict[str, Any]:
    if url:
        image_url = url
    else:
        image_url = f"data:{mime_type};base64,{base64_data}"
    return {"type": "image_url", "image_url": {"url": image_url}}


def _build_anthropic_image_part(
    *,
    base64_data: str,
    mime_type: str,
) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": base64_data,
        },
    }


def prepare_multimodal_messages(
    messages: list[dict[str, Any]],
    attachments: list[Any] | None,
    binding: str = "openai",
    model: str | None = None,
    fallback_text: str = "",
) -> list[dict[str, Any]]:
    """Inject image attachments into the last user message (Stage 1).

    Images are injected **optimistically for every provider/model** — this
    function does not consult ``supports_vision``. A model that natively
    understands images therefore always receives them, even one we have no
    capability entry for. When a model genuinely cannot handle images the
    request fails and the Stage-2 fallback (:func:`should_degrade_to_text`
    + :func:`strip_image_parts_inplace`, applied at each call site's retry
    seam) strips the images and retries as text-only.

    The last user message ``content`` is converted from a plain string into a
    content-parts array holding the original text plus the image(s). Images
    are only *dropped* here when they have neither base64 nor a readable
    ``file_path`` nor an external http(s) URL (each drop is logged).

    Mutates ``messages`` in place and returns the same list.

    Args:
        messages: The OpenAI-style messages list (mutated in place).
        attachments: ``Attachment`` objects from ``UnifiedContext``.
        binding: Provider binding (``"openai"``, ``"anthropic"``, …).
        model: Model name (reserved; Stage-1 does not gate on it).
        fallback_text: Text used for the text part when the last user message
            content is an empty string (e.g. user sent only an image).
    """
    del model  # reserved for future URL-vs-base64 format selection
    if not attachments:
        return messages

    image_attachments = [a for a in attachments if a.is_image()]
    if not image_attachments:
        return messages

    last_user_idx = _find_last_user_message(messages)
    if last_user_idx is None:
        return messages

    is_anthropic = (binding or "").lower() in ("anthropic", "claude")
    _inject_images(
        messages,
        last_user_idx,
        image_attachments,
        anthropic=is_anthropic,
        fallback_text=fallback_text,
    )
    return messages


def _find_last_user_message(messages: list[dict[str, Any]]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def _inject_images(
    messages: list[dict[str, Any]],
    user_idx: int,
    image_attachments: list[Any],
    *,
    anthropic: bool = False,
    fallback_text: str = "",
) -> None:
    """Inject image parts into the user message at *user_idx* (in place)."""
    msg = messages[user_idx]
    original_content = msg.get("content", "")

    if isinstance(original_content, str):
        text = original_content if original_content != "" else (fallback_text or "")
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    elif isinstance(original_content, list):
        content_parts = list(original_content)
    else:
        content_parts = [{"type": "text", "text": str(original_content)}]

    for att in image_attachments:
        b64 = getattr(att, "base64", "") or ""
        url = getattr(att, "url", "") or ""
        file_path = getattr(att, "file_path", "") or ""

        # 已有 base64 优先；否则从 file_path 读盘兜底（对标 DeepTutor 本地解析），
        # 让 from_image_path 路径无需调用方预填 base64。
        if not b64 and file_path:
            try:
                raw = Path(file_path).read_bytes()
                b64 = _b64.b64encode(raw).decode("ascii")
            except OSError as exc:
                logger.warning("failed to read image file %s: %s", file_path, exc)

        if not b64 and not url:
            continue

        # mime：优先显式 mime_type，否则按 filename（无则 file_path）猜扩展名
        mime = getattr(att, "mime_type", "") or _guess_mime_type(
            getattr(att, "filename", "") or file_path or "image.png"
        )

        if anthropic:
            # Anthropic 只接受 base64 source；没有 base64（且读盘失败 / 仅服务器相对
            # 路径 /api/uploads/...）的图片无法发给外部 provider，丢弃。
            if not b64:
                logger.warning("Anthropic image part requires base64; dropping %r", url)
                continue
            content_parts.append(_build_anthropic_image_part(base64_data=b64, mime_type=mime))
        else:
            if b64:
                # 有 base64 时优先内联 data URL——多数 provider 都接受，也避开把服务器
                # 相对路径 /api/uploads/... 当外部 image URL 发给 LLM 的问题。
                content_parts.append(_build_openai_image_part(base64_data=b64, mime_type=mime))
            elif url.lower().startswith(("http://", "https://")):
                # 仅外部 http(s) URL 才以 URL 形式发送（对标 DeepTutor）。
                content_parts.append(
                    _build_openai_image_part(base64_data="", mime_type=mime, url=url)
                )
            else:
                logger.warning(
                    "Dropping unresolvable image url %r (no base64, not http(s))", url
                )
                continue

    messages[user_idx] = {**msg, "content": content_parts}


_IMAGE_BLOCK_TYPES = frozenset({"image_url", "image"})


def _block_image_placeholder(block: dict[str, Any]) -> str:
    """Human-readable text placeholder for an image block being stripped."""
    meta = block.get("_meta") or {}
    label = ""
    if isinstance(meta, dict):
        label = str(meta.get("path") or meta.get("filename") or "").strip()
    if not label and block.get("type") == "image_url":
        image_url = block.get("image_url") or {}
        if isinstance(image_url, dict):
            url = str(image_url.get("url") or "").strip()
            if url and not url.startswith("data:"):
                label = url
    return f"[image: {label}]" if label else "[image omitted]"


def has_image_parts(messages: list[dict[str, Any]]) -> bool:
    """Return True when any message content contains image blocks."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in _IMAGE_BLOCK_TYPES:
                return True
    return False


def strip_image_parts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a **new** message list with image blocks replaced by text
    placeholders. Use when the caller must preserve the original (e.g. to
    attempt a text-only retry while keeping the image payload for a possible
    second provider)."""
    stripped: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            stripped.append(dict(msg))
            continue
        new_content: list[dict[str, Any]] = [
            {"type": "text", "text": _block_image_placeholder(item)}
            if isinstance(item, dict) and item.get("type") in _IMAGE_BLOCK_TYPES
            else item
            for item in content
        ]
        stripped.append({**msg, "content": new_content})
    return stripped


def strip_image_parts_inplace(messages: list[dict[str, Any]]) -> bool:
    """Replace image blocks with text placeholders **in place**; return True
    if any were replaced.

    Used by call sites that share one message list across retries / loop
    iterations (the chat agentic loop) so the degrade persists and images are
    not re-sent — and re-rejected — on every subsequent call."""
    found = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for idx, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES:
                content[idx] = {"type": "text", "text": _block_image_placeholder(block)}
                found = True
    return found


def should_degrade_to_text(
    binding: str | None,
    model: str | None,
    messages: list[dict[str, Any]],
) -> bool:
    """Stage-2 fallback decision.

    After a request that carried image content fails, return True when we
    should strip the images and retry as text-only — i.e. the payload
    actually had image parts **and** the model is *not* in the known-vision
    allowlist (``supports_vision`` is False). For allowlisted (known
    vision-capable) models we keep the images so a genuine error surfaces
    instead of silently returning a misleading text-only answer.
    """
    if not has_image_parts(messages):
        return False
    return not supports_vision(binding or "openai", model)


def _error_text(exc: BaseException) -> str:
    """从异常对象里抠出可读错误文本（兼容 openai/python SDK 异常的不同字段）。

    对标 DeepTutor ``agent_loop._error_text``。
    """
    response = getattr(exc, "response", None)
    body = (
        getattr(exc, "body", None)
        or getattr(exc, "doc", None)
        or getattr(response, "text", None)
        or getattr(exc, "message", None)
        or str(exc)
    )
    return str(body).lower()


def is_image_input_unsupported(exc: BaseException) -> bool:
    """Stage-2 判据：异常是否表明当前模型/接口不接受图片输入。

    覆盖 OpenAI / Anthropic / 国产网关的常见拒绝措辞，以及部分网关把 content
    数组里的非字符串元素报成 ``must be a string`` / ``expected a string`` 的
    情况。对标 DeepTutor ``agent_loop._is_image_input_unsupported``。
    """
    text = _error_text(exc)
    return any(
        marker in text
        for marker in (
            "image",
            "vision",
            "multimodal",
            "image_url",
            "content type",
            "must be a string",
            "expected a string",
            "expected string",
            "invalid type for 'messages",
        )
    )


__all__ = [
    "has_image_parts",
    "is_image_input_unsupported",
    "prepare_multimodal_messages",
    "should_degrade_to_text",
    "strip_image_parts",
    "strip_image_parts_inplace",
]
