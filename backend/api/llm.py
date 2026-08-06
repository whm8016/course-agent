"""LLM 供应商 profile 管理 REST API（provider 设置）。

provider 以 profile 池形式部署级共享（管理员预配）；所有用户对话时从 /selectable
下拉选 profile，后端按该 profile 动态构造 client（provider_factory.get_llm_client_for_profile）
注入 run_agent_loop，即时生效、无需重启。

- GET    /api/llm/providers            -> 列出支持的供应商（公开，供前端 binding 下拉）
- GET    /api/llm/profiles/selectable  -> 可选 profile 列表（全员，去 api_key，对话下拉用）
- GET    /api/llm/profiles             -> 完整 profile 列表（admin，含 key 供编辑回填）
- POST   /api/llm/profiles/{id}        -> 新增/更新 profile（admin）
- DELETE /api/llm/profiles/{id}        -> 删除 profile（admin）
- PUT    /api/llm/active               -> 设默认 profile（admin）
- POST   /api/llm/profiles/{id}/test   -> 测试已存 profile 连接（admin）
- POST   /api/llm/probe                -> 测试未保存配置（admin，不持久化）
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import get_current_user, require_admin
from core.llm import catalog as catalog_mod
from core.llm.provider_factory import clear_llm_client_cache
from core.llm.provider_registry import PROVIDERS

router = APIRouter(prefix="/llm")


class ProfilePayload(BaseModel):
    name: str = ""
    binding: str = ""
    api_key: str = ""
    base_url: str = ""
    api_version: str = ""
    text_model: str = ""
    fast_model: str = ""
    vision_model: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    fallback_api_key: str = ""
    fallback_base_url: str = ""
    fallback_model: str = ""


class ActiveRequest(BaseModel):
    profile_id: str


# ── 公开：支持的供应商 ──────────────────────────────────────────────────────

@router.get("/providers")
async def list_providers(user: dict = Depends(get_current_user)):
    """列出支持的供应商（供前端 admin 配置页的 binding 下拉 + 默认 base_url 预填）。"""
    return {
        "providers": [
            {
                "name": p.name,
                "backend": p.backend,
                "default_api_base": p.default_api_base or "",
                "env_key": p.env_key,
            }
            for p in PROVIDERS
        ]
    }


# ── 全员：可选 profile（对话下拉用，去 key）────────────────────────────────

@router.get("/profiles/selectable")
async def list_selectable_profiles(user: dict = Depends(get_current_user)):
    catalog = await catalog_mod.load_catalog_cached()
    active = catalog_mod.active_profile_id(catalog)
    return {
        "profiles": [
            catalog_mod.profile_public_view(p, active)
            for p in catalog_mod.list_profiles(catalog)
        ],
        "active": active,
    }


# ── admin：完整 profile 管理 ─────────────────────────────────────────────────

@router.get("/profiles")
async def list_profiles_admin(_: dict = Depends(require_admin)):
    catalog = await catalog_mod.load_catalog_cached()
    active = catalog_mod.active_profile_id(catalog)
    return {
        "profiles": [
            catalog_mod.profile_admin_view(p, active)
            for p in catalog_mod.list_profiles(catalog)
        ],
        "active": active,
    }


@router.post("/profiles/{profile_id}")
async def upsert_profile_route(
    profile_id: str, payload: ProfilePayload, _: dict = Depends(require_admin)
):
    if not profile_id.strip():
        raise HTTPException(status_code=400, detail="profile id 不能为空")
    # 写文件丢线程池，避免阻塞事件循环；写后失效缓存，全部 worker 下次读自动 miss 重载
    saved = await asyncio.to_thread(
        catalog_mod.upsert_profile, profile_id, payload.model_dump()
    )
    clear_llm_client_cache()  # key 变更后强制重建（正常路径按指纹也会自动正确，双保险）
    await catalog_mod.invalidate_catalog_cache()
    active = await catalog_mod.active_profile_id_cached()
    return {"saved": True, "profile": catalog_mod.profile_admin_view(saved, active)}


@router.delete("/profiles/{profile_id}")
async def delete_profile_route(profile_id: str, _: dict = Depends(require_admin)):
    deleted = await asyncio.to_thread(catalog_mod.delete_profile, profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"profile '{profile_id}' not found")
    clear_llm_client_cache()
    await catalog_mod.invalidate_catalog_cache()
    return {"deleted": True}


@router.put("/active")
async def set_active_route(payload: ActiveRequest, _: dict = Depends(require_admin)):
    ok = await asyncio.to_thread(catalog_mod.set_active, payload.profile_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"profile '{payload.profile_id}' not found")
    clear_llm_client_cache()
    await catalog_mod.invalidate_catalog_cache()
    return {"active": payload.profile_id}


async def _probe_profile(prof: dict, timeout: int = 20) -> dict:
    """用 profile 配置发极简 completion 验证对话 + 视觉两路连通（不持久化、不入缓存）。

    admin profile 的视觉模型与对话模型共用同一 binding/api_key/base_url（不拆独立
    视觉供应商，区别于学生 /llm/me）。返回结构与 /llm/me/test 对齐：顶层=对话结果
    （向后兼容），另附 text/vision 明细；vision 未配则为 None。
    """
    from core.llm.catalog import profile_text_model, profile_vision_model
    from core.llm.provider_factory import get_llm_client

    binding = (prof.get("binding") or "").strip()
    api_key = (prof.get("api_key") or "").strip()
    base_url = (prof.get("base_url") or "").strip() or None
    api_version = (prof.get("api_version") or "").strip() or None

    async def _ping(model: str, warning: str | None) -> dict:
        if not model:
            return {"ok": False, "model": "", "error": "未配置模型", "warning": warning}
        try:
            client = get_llm_client(
                binding=binding or None,
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
            return {"ok": True, "model": model, "error": "", "warning": warning}
        except Exception as e:
            return {"ok": False, "model": model, "error": str(e)[:300], "warning": warning}

    text_model = profile_text_model(prof)
    text_res = await _ping(text_model, None)

    vision_res: dict | None = None
    vision_model = profile_vision_model(prof)
    if vision_model:
        from core.llm.capabilities import supports_vision

        warning = None
        if not supports_vision(binding or None, vision_model):
            warning = "按能力表该模型可能不支持图片输入，实际看图或失败（如 qwen-vl-plus / gpt-4o）"
        vision_res = await _ping(vision_model, warning)

    return {
        "ok": text_res["ok"],
        "binding": binding or None,
        "model": text_model,
        "error": text_res["error"],
        "text": text_res,
        "vision": vision_res,
    }


@router.post("/profiles/{profile_id}/test")
async def test_profile_route(profile_id: str, _: dict = Depends(require_admin)):
    prof = await catalog_mod.get_profile_cached(profile_id)
    if not prof:
        raise HTTPException(status_code=404, detail=f"profile '{profile_id}' not found")
    return await _probe_profile(prof)


@router.post("/probe")
async def probe_route(payload: ProfilePayload, _: dict = Depends(require_admin)):
    prof = {
        "binding": payload.binding,
        "api_key": payload.api_key,
        "base_url": payload.base_url,
        "api_version": payload.api_version,
        "models": {
            "text": {"model": payload.text_model},
            "vision": {"model": payload.vision_model},
        },
    }
    return await _probe_profile(prof)
