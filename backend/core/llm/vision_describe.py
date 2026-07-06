"""两阶段图片处理：当主回答模型不支持 vision 时，先用 vision 模型描述图片，
再把描述文本喂给主回答模型（避免直接剥图丢失信息）。

设计动机（用户确认）：
- Stage-1 乐观注入 + Stage-2 剥图降级（multimodal.py / llm.py 现有逻辑）会在主模型
  不支持 vision 时丢掉图片，导致「看不到图片」。deepseek 等纯文本模型即此场景。
- 两阶段（本模块）：主模型不支持 vision 时，用专门的 vision 模型把图片转成文字描述，
  把描述拼进 user_message，主模型就能基于文字「看懂」图片。
- 比直接剥图更好——信息不丢失。对标 ingestion image_extractor 的 vision 描述，但
  解耦 RAG 摄入（ImageModalProcessor）语义，独立用于 chat。

调用链：chat_pipeline.run → describe_image_attachments → 主模型用描述文字回答。
"""
from __future__ import annotations

import base64 as _b64
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 让 vision 模型把图片转成可被纯文本回答模型利用的结构化描述。
_VISION_DESC_PROMPT = (
    "请详细、准确地描述这张图片的内容。要求：\n"
    "1. 完整转述图片中的所有文字（含题目、选项、标注、坐标轴标签等）；\n"
    "2. 描述图表/图形/示意图的形状、数据、关系（如有函数曲线请说明趋势与关键点）；\n"
    "3. 若含公式，请用 LaTeX 转写；\n"
    "4. 不要做解答或推理，只做客观描述。"
)


def _resolve_image_data_url(att: Any) -> str | None:
    """把 Attachment 转成 vision API 可识别的 data URL（优先 base64，回退 file_path 读盘）。"""
    b64 = (getattr(att, "base64", None) or "").strip()
    if not b64:
        fp = getattr(att, "file_path", None)
        if fp:
            try:
                raw = Path(fp).read_bytes()
                b64 = _b64.b64encode(raw).decode("ascii")
            except OSError as exc:
                logger.warning("vision_describe: 读图失败 %s: %s", fp, exc)
                return None
    if not b64:
        return None
    mime = (getattr(att, "mime_type", None) or "").strip() or "image/png"
    return f"data:{mime};base64,{b64}"


async def describe_image_attachments(
    vision_client: Any,
    vision_model: str,
    images: list[Any],
    *,
    prompt: str = _VISION_DESC_PROMPT,
    max_tokens: int = 1024,
) -> list[str]:
    """用 vision 模型逐张描述图片，返回描述列表（与 images 等长，失败/无数据项为空串）。

    Args:
        vision_client: 暴露 ``chat.completions.create`` 的 OpenAI 兼容 async client。
        vision_model: 支持 vision 的模型名（如 qwen-vl-plus）。
        images: Attachment 列表（需含 base64 或 file_path）。
        prompt: 描述指令。
        max_tokens: 单张描述上限。

    Returns:
        描述文本列表，顺序与 images 一一对应；描述失败返回空串占位。
    """
    if not vision_client or not vision_model or not images:
        return []

    descriptions: list[str] = []
    for idx, att in enumerate(images):
        data_url = _resolve_image_data_url(att)
        if not data_url:
            descriptions.append("")
            logger.warning("vision_describe: 图片 %d 无可用数据，跳过", idx + 1)
            continue
        try:
            resp = await vision_client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            desc = ((resp.choices[0].message.content or "").strip()) if resp.choices else ""
            descriptions.append(desc)
            if not desc:
                logger.warning("vision_describe: 图片 %d 返回空描述", idx + 1)
        except Exception as exc:  # 单张失败不阻断其余
            logger.warning("vision_describe: 描述图片 %d 失败: %s", idx + 1, exc)
            descriptions.append("")
    return descriptions


def build_image_description_block(descriptions: list[str]) -> str:
    """把非空描述拼成喂给主回答模型的文本块（带编号）。"""
    lines: list[str] = []
    n = 0
    for desc in descriptions:
        if not desc:
            continue
        n += 1
        lines.append(f"【图片{n} 视觉内容】\n{desc}")
    return "\n\n".join(lines)


async def resolve_vision_runtime(user_id: str = "") -> tuple[Any, str | None]:
    """解析两阶段图片描述用的 (vision_client, vision_model)。

    视觉模型走独立供应商（可异于对话供应商：对话 deepseek，视觉 dashscope/qwen-vl）。
    优先级：
    1. 用户自配的视觉独立供应商（vision_binding/key/url/model，且 supports_vision）
    2. 全局 VISION_MODEL（settings.vision）+ VISION_API_KEY/BASE_URL（默认回退 EMBEDDING_* 阿里 dashscope）
    3. 都没有 → (None, None)，调用方回退到原有的剥图降级
    """
    from core.llm.capabilities import supports_vision
    from core.llm.provider_factory import get_llm_client_for_profile
    from settings import get_settings

    settings = get_settings()

    # 1. 用户视觉独立供应商
    if user_id:
        try:
            from core.db.user_llm_provider import get_active_provider_view

            user_prof = await get_active_provider_view(user_id)
            v_model = (user_prof or {}).get("vision_model") or ""
            if user_prof and v_model:
                v_prof = {
                    "binding": user_prof.get("vision_binding") or "",
                    "api_key": user_prof.get("vision_api_key") or "",
                    "base_url": user_prof.get("vision_base_url") or "",
                    "api_version": "",
                }
                if supports_vision(v_prof["binding"], v_model):
                    return get_llm_client_for_profile(v_prof), v_model
        except Exception:
            logger.exception("resolve vision runtime: 用户视觉供应商解析失败")

    # 2. 全局 VISION_MODEL 回退（VISION_API_KEY/BASE_URL，默认回退 EMBEDDING_* 阿里 dashscope）
    v_model = settings.vision.model
    v_key = settings.vision.api_key.get_secret_value()
    v_url = settings.vision.base_url
    if v_model and v_key and v_url:
        try:
            from openai import AsyncOpenAI

            return AsyncOpenAI(api_key=v_key, base_url=v_url), v_model
        except Exception:
            logger.exception("resolve vision runtime: 构造全局 vision client 失败")

    return None, None


async def describe_images_into(
    context: Any,
    base_text: str,
    *,
    user_id: str = "",
    text_model: str | None = None,
    binding: str | None = None,
) -> str:
    """两阶段图片描述的统一入口（chat / deep_solve / deep_research / quiz 共用）。

    若主回答模型不支持 vision 但存在可用 vision 模型：把 ``context.attachments`` 里的
    图片转成文字描述，返回 ``base_text`` 拼上描述块，并从 ``context.attachments``
    移除图片（loop 不再注入）。否则原样返回 ``base_text``、不动附件。

    Args:
        context: ``UnifiedContext``（读写 ``attachments``；附件需带 base64 或 file_path）。
        base_text: 该轮将作为 ``user_message`` 的基础文案（由各 pipeline 传入）。
        user_id: 用于解析用户自配视觉/对话供应商。
        text_model/binding: 主回答模型名与 binding；为 None 时用全局默认
            (``settings.llm.text_model`` / ``settings.llm.binding``)，用于判断
            主模型是否原生支持 vision。

    Returns:
        增强后的 user_message 文案（= base_text + 图片描述块）；无需描述时原样返回。
    """
    from core.llm.capabilities import supports_vision
    from core.observability import log_flow
    from settings import get_settings

    image_attachments = [a for a in (context.attachments or []) if a.is_image()]
    if not image_attachments:
        return base_text

    settings = get_settings()
    eff_model = text_model or settings.llm.text_model
    eff_binding = (binding or settings.llm.binding or "").strip()
    # 主模型原生支持 vision → 不需要两阶段，loop 走 Stage-1 乐观注入即可
    if supports_vision(eff_binding, eff_model):
        return base_text

    vision_client, eff_vision_model = await resolve_vision_runtime(user_id)
    if not vision_client or not eff_vision_model:
        log_flow("image_two_stage_skipped",
                 reason="no_vision_model", main_model=eff_model,
                 image_count=len(image_attachments))
        return base_text

    descriptions = await describe_image_attachments(
        vision_client, eff_vision_model, image_attachments
    )
    block = build_image_description_block(descriptions)
    if not block:
        # 全部描述失败 → 不动附件，让 loop 的 Stage-2 剥图降级兜底
        log_flow("image_two_stage_failed", image_count=len(image_attachments),
                 vision_model=eff_vision_model)
        return base_text

    # 描述拼入文案，并移除图片附件（loop 不再注入图片）
    context.attachments = [a for a in (context.attachments or []) if not a.is_image()]
    described = sum(1 for d in descriptions if d)
    log_flow("image_two_stage_ok",
             image_count=len(image_attachments), described=described,
             vision_model=eff_vision_model, desc_chars=len(block))
    base = (base_text or "").strip()
    return f"{base}\n\n{block}".strip() if base else block


__all__ = [
    "describe_image_attachments",
    "build_image_description_block",
    "resolve_vision_runtime",
    "describe_images_into",
]
