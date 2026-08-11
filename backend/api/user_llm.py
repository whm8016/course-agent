"""用户级 LLM provider REST API。

路由前缀 /llm/me —— 当前用户的 provider 配置（多租户隔离）。
所有端点需认证（get_current_user）。api_key 在 GET 时脱敏（仅返回 is_set bool）。

端点：
- GET    /llm/me         → 当前用户 provider（key 脱敏）
- PUT    /llm/me         → upsert（覆盖或新增）
- DELETE /llm/me         → 删除（回退平台默认）
- POST   /llm/me/test    → 测试连通性（用刚填配置发极简 completion）
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import get_current_user
from core.db.user_llm_provider import (
    UserProviderPayload,
    delete_provider,
    get_provider_admin_view,
    upsert_provider,
)
from core.llm.provider_factory import clear_llm_client_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm/me", tags=["user-llm-provider"])


class UserProviderResponse(BaseModel):
    """用户端响应：api_key 脱敏为 *_set bool（对话 + 视觉两把）。"""

    # 对话供应商
    binding: str = ""
    api_key_set: bool = False  # True=已设置，不回传明文
    base_url: str = ""
    api_version: str = ""
    text_model: str = ""
    # 视觉独立供应商
    vision_binding: str = ""
    vision_api_key_set: bool = False
    vision_base_url: str = ""
    vision_model: str = ""


def _masked_response(view: dict | None) -> UserProviderResponse:
    """admin view -> 用户端脱敏视图（api_key 仅 *_set bool，不回传明文）。

    GET / PUT 共用：PUT 原先回吐 admin view（含 decrypt 出的明文 key），每次保存都把
    明文过一遍响应体/网关日志，抵消 GET 脱敏的意义。现统一走本函数脱敏。
    """
    if not view:
        return UserProviderResponse()
    return UserProviderResponse(
        binding=view.get("binding", ""),
        api_key_set=bool(view.get("api_key")),
        base_url=view.get("base_url", ""),
        api_version=view.get("api_version", ""),
        text_model=view.get("text_model", ""),
        vision_binding=view.get("vision_binding", ""),
        vision_api_key_set=bool(view.get("vision_api_key")),
        vision_base_url=view.get("vision_base_url", ""),
        vision_model=view.get("vision_model", ""),
    )


@router.get("", response_model=UserProviderResponse)
async def get_my_provider(user: dict = Depends(get_current_user)) -> UserProviderResponse:
    """获取当前用户的 LLM provider 配置（api_key 脱敏）。"""
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="未认证")
    return _masked_response(await get_provider_admin_view(user_id))


@router.put("")
async def upsert_my_provider(
    payload: UserProviderPayload, user: dict = Depends(get_current_user)
) -> dict:
    """新增或更新当前用户的 LLM provider 配置。"""
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="未认证")
    saved = await upsert_provider(user_id, payload)
    clear_llm_client_cache()  # key 变更后强制重建 client 缓存
    logger.info("user %s upsert provider binding=%s", user_id, payload.binding)
    # 1.5：返回脱敏视图（api_key_set），不回吐 admin view 的明文 key
    return {"saved": True, "provider": _masked_response(saved).model_dump()}


@router.delete("")
async def delete_my_provider(user: dict = Depends(get_current_user)) -> dict:
    """删除当前用户的 LLM provider 配置（回退平台默认）。"""
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="未认证")
    deleted = await delete_provider(user_id)
    clear_llm_client_cache()
    logger.info("user %s delete provider deleted=%s", user_id, deleted)
    return {"deleted": deleted}


async def _probe_completion(
    *,
    binding: str | None,
    api_key: str,
    base_url: str | None,
    api_version: str | None,
    model: str,
    timeout: int = 20,
) -> dict:
    """发极简 completion 验证对话模型连通（与 api/llm.py _probe_profile 同语义）。

    用 get_llm_client（不缓存），避免测试 client 污染生产 client 缓存。
    """
    if not model:
        return {"ok": False, "model": "", "error": "未配置模型"}
    try:
        from core.llm.provider_factory import get_llm_client

        client = get_llm_client(
            binding=binding,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            model=model,
            timeout=timeout,
        )
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            stream=False,
        )
        return {"ok": True, "model": model, "error": ""}
    except Exception as e:
        return {"ok": False, "model": model, "error": str(e)[:300]}


@router.post("/test")
async def test_my_provider(
    payload: UserProviderPayload, user: dict = Depends(get_current_user)
) -> dict:
    """测试用户提交的 provider 配置连通性（不持久化）。

    对话模型与视觉模型各发一次极简 completion。视觉模型仅当用户填了 vision_model
    时才测——否则生产 resolve_vision_runtime 会走全局 VISION_* 回退（不归个人配置管），
    此处不报错。返回结构向后兼容：顶层 ok/binding/model/error 仍表示对话模型结果，
    另附 text / vision 两栏明细。
    """
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="未认证")

    # 1.6 SSRF：用户可达的探测端点，校验自定义 base_url 指向公网（拒内网/元数据地址）。
    # base_url 留空走供应商默认（如 dashscope 官方），不校验。
    from utils.url_guard import assert_public_http_url
    for _bu in ((payload.base_url or "").strip(), (payload.vision_base_url or "").strip()):
        if _bu:
            try:
                assert_public_http_url(_bu)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"base_url 不安全：{exc}")

    # ── 对话模型 ────────────────────────────────────────────────────────
    text_binding = (payload.binding or "").strip()
    text_res = await _probe_completion(
        binding=text_binding or None,
        api_key=(payload.api_key or "").strip(),
        base_url=(payload.base_url or "").strip() or None,
        api_version=(payload.api_version or "").strip() or None,
        model=(payload.text_model or "").strip(),
    )

    # ── 视觉模型（仅当填了 vision_model）────────────────────────────────
    # 与生产 resolve_vision_runtime 分支1 一致：用 get_llm_client_for_profile 构造，
    # 空 binding/key/base_url 自动回退全局 llm 凭证（settings.llm.*），故用户只填
    # vision_model（视觉与对话同供应商）也能正确探测。
    vision_model = (payload.vision_model or "").strip()
    vision_res: dict | None = None
    if vision_model:
        from core.llm.capabilities import supports_vision
        from core.llm.provider_factory import get_llm_client_for_profile

        v_prof = {
            "binding": (payload.vision_binding or "").strip(),
            "api_key": (payload.vision_api_key or "").strip(),
            "base_url": (payload.vision_base_url or "").strip(),
            "api_version": "",
        }
        # 静态能力提示：模型名按能力表不支持图片输入时给 warning（不阻断连通测试，
        # 因 supports_vision 基于命名约定，可能漏掉新模型）。
        warning: str | None = None
        if not supports_vision(v_prof["binding"] or None, vision_model):
            warning = "按能力表该模型可能不支持图片输入，实际看图或失败（请确认模型名，如 qwen-vl-plus / gpt-4o）"
        try:
            v_client = get_llm_client_for_profile(v_prof, timeout=20)
            await v_client.chat.completions.create(
                model=vision_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                stream=False,
            )
            vision_res = {"ok": True, "model": vision_model, "error": "", "warning": warning}
        except Exception as e:
            vision_res = {"ok": False, "model": vision_model, "error": str(e)[:300], "warning": warning}

    return {
        "ok": text_res["ok"],  # 总体：对话通即可用（向后兼容；视觉未配不应让总体变红）
        "binding": text_binding or None,
        "model": text_res["model"],
        "error": text_res["error"],
        "text": text_res,
        "vision": vision_res,  # None = 未配置视觉模型（走平台默认）
    }
