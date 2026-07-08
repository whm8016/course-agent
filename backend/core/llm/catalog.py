"""LLM provider catalog — data/model_catalog.json 的实时读写层。

config.py 在启动时把 active profile 扁平化为 module 常量（供 embedding / LightRAG /
bot 等启动期消费方使用，向后兼容）。本模块负责**运行期**管理：按 id 取 profile、
CRUD、写回 JSON——所有读操作实时读文件，保证 API 写入后下一个请求立即可见。

provider 以 **profile 池**形式存在（管理员预配多个 provider+model
组合）。用户对话时可在前端下拉指定 model_profile_id，后端按该 profile 动态构造
client 注入 run_agent_loop（见 provider_factory.get_llm_client_for_profile 与
chat_pipeline 的 profile 解析逻辑）——立即生效、无需重启、不触碰全局 client 单例。

【安全】Profile.api_key 使用 SecretStr，日志/打印自动脱敏。API 返回时：
- 公开视图（profile_public_view）：仅返回 api_key_configured bool
- 管理视图（profile_admin_view）：返回明文（仅 admin endpoint）
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from typing import Any

from pydantic import BaseModel, SecretStr

from settings import BASE_DIR

CATALOG_PATH = os.getenv(
    "MODEL_CATALOG_PATH", os.path.join(BASE_DIR, "data", "model_catalog.json")
)

_write_lock = threading.Lock()


# ── Pydantic 模型（SecretStr 脱敏）────────────────────────────────────────────


class EmbeddingConfig(BaseModel):
    """嵌入模型配置（可独立 api_key/base_url）。"""
    model: str = ""
    api_key: SecretStr = SecretStr("")
    base_url: str = ""


class FallbackConfig(BaseModel):
    """fallback provider（主 provider 失败时切换）。"""
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    model: str = ""


class ModelConfig(BaseModel):
    """单任务模型配置。"""
    model: str = ""


class Profile(BaseModel):
    """LLM provider profile —— catalog JSON 单条记录的 Pydantic 模型。

    api_key 使用 SecretStr，日志/打印自动脱敏。需要明文时调用
    .api_key.get_secret_value()。
    """
    id: str = ""
    name: str = ""
    binding: str = ""
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    api_version: str = ""
    models: dict[str, Any] = {}
    fallback: FallbackConfig = FallbackConfig()

    def get_text_model(self) -> str:
        return str((self.models.get("text") or {}).get("model") or "")

    def get_fast_model(self) -> str:
        return str((self.models.get("fast") or {}).get("model") or "")

    def get_embedding_config(self) -> EmbeddingConfig:
        emb = self.models.get("embedding") or {}
        return EmbeddingConfig(
            model=str(emb.get("model") or ""),
            api_key=SecretStr(str(emb.get("api_key") or "")),
            base_url=str(emb.get("base_url") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """导出为 dict（api_key 明文，用于 provider_factory）。"""
        return {
            "id": self.id,
            "name": self.name,
            "binding": self.binding,
            "api_key": self.api_key.get_secret_value(),
            "base_url": self.base_url,
            "api_version": self.api_version,
            "models": self.models,
            "fallback": {
                "api_key": self.fallback.api_key.get_secret_value(),
                "base_url": self.fallback.base_url,
                "model": self.fallback.model,
            },
        }


def _empty_catalog() -> dict[str, Any]:
    return {"active_profile": "default", "profiles": []}


def load_catalog() -> dict[str, Any]:
    """读取整个 catalog（实时读文件，确保 API 写入后立即可见）。

    文件缺失 / 损坏时回退空结构（active=default, profiles=[]）。
    """
    try:
        with open(CATALOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _empty_catalog()


def save_catalog(data: dict[str, Any]) -> None:
    """写回 catalog（线程安全；保留 _comment / _examples 等额外字段）。

    M-20：用临时文件 + os.replace 原子替换，杜绝 load_catalog 读到「写了一半」的
    半截 JSON（JSONDecodeError 回退空 catalog → 丢失所有 profile）。os.replace 在
    Windows/Linux 均为原子的目标文件替换。
    """
    data.setdefault("profiles", [])
    data.setdefault("active_profile", "default")
    with _write_lock:
        directory = os.path.dirname(CATALOG_PATH) or "."
        os.makedirs(directory, exist_ok=True)
        # 临时文件与目标同目录，保证 os.replace 是同分区原子改名
        fd, tmp_path = tempfile.mkstemp(prefix=".catalog_", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, CATALOG_PATH)
        except Exception:
            # 写失败清理临时文件，不留垃圾
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


@contextlib.contextmanager
def _atomic_update() -> Any:
    """load→modify→save 临界区上下文（M-20：消除 TOCTOU 丢更新）。

    with _atomic_update() as data:
        data["profiles"].append(...)   # 在锁内修改
    # 退出时自动 save_catalog（仍持 _write_lock）

    此前 upsert/delete/set_active 各自 load_catalog → 改 dict → save_catalog，三步之间
    另一线程可能已写入，本线程的 save 会覆盖对方修改（丢更新）。现在整个「读-改-写」
    在 _write_lock 内原子完成。
    """
    with _write_lock:
        data = load_catalog()
        yield data
        # 写回（_write_lock 可重入？threading.Lock 不可重入——save_catalog 内部再
        # acquire 会死锁。故 _save_inplace 直接在已持锁状态下写文件，不再 acquire）。
        _save_inplace(data)


def _save_inplace(data: dict[str, Any]) -> None:
    """save_catalog 的无锁内核（_atomic_update 已持 _write_lock 时调用）。"""
    data.setdefault("profiles", [])
    data.setdefault("active_profile", "default")
    directory = os.path.dirname(CATALOG_PATH) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".catalog_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, CATALOG_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def active_profile_id(catalog: dict[str, Any] | None = None) -> str:
    data = catalog if catalog is not None else load_catalog()
    return str(data.get("active_profile") or "default")


def list_profiles(catalog: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = catalog if catalog is not None else load_catalog()
    return [p for p in (data.get("profiles") or []) if isinstance(p, dict)]


def get_profile(profile_id: str, catalog: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """按 id 取 profile 原始对象；不存在 / 空入参返回 None。"""
    if not profile_id:
        return None
    data = catalog if catalog is not None else load_catalog()
    for p in data.get("profiles") or []:
        if isinstance(p, dict) and p.get("id") == profile_id:
            return p
    return None


# ── profile 视图投影 ──────────────────────────────────────────────────────

def profile_display_name(p: dict[str, Any]) -> str:
    return str(p.get("name") or p.get("id") or "")


def profile_text_model(p: dict[str, Any]) -> str:
    return str((p.get("models") or {}).get("text", {}).get("model") or "")


def profile_fast_model(p: dict[str, Any]) -> str:
    return str((p.get("models") or {}).get("fast", {}).get("model") or "")


def profile_vision_model(p: dict[str, Any]) -> str:
    return str((p.get("models") or {}).get("vision", {}).get("model") or "")


def profile_public_view(p: dict[str, Any], active_id: str) -> dict[str, Any]:
    """公开视图（去 api_key 明文）——供所有用户在对话下拉选用 provider/model。"""
    return {
        "id": p.get("id", ""),
        "name": profile_display_name(p),
        "binding": p.get("binding", ""),
        "text_model": profile_text_model(p),
        "fast_model": profile_fast_model(p),
        "vision_model": profile_vision_model(p),
        "base_url_configured": bool(p.get("base_url")),
        "api_key_configured": bool(p.get("api_key")),
        "active": p.get("id") == active_id,
    }


def profile_admin_view(p: dict[str, Any], active_id: str) -> dict[str, Any]:
    """管理员视图（含 api_key 明文，供编辑回填；仅 admin endpoint 返回）。"""
    models = p.get("models") or {}
    emb = models.get("embedding") or {}
    fb = p.get("fallback") or {}
    return {
        "id": p.get("id", ""),
        "name": profile_display_name(p),
        "binding": p.get("binding", ""),
        "api_key": p.get("api_key", ""),
        "base_url": p.get("base_url", ""),
        "api_version": p.get("api_version", ""),
        "text_model": (models.get("text") or {}).get("model", ""),
        "fast_model": (models.get("fast") or {}).get("model", ""),
        "vision_model": (models.get("vision") or {}).get("model", ""),
        "embedding_model": emb.get("model", ""),
        "embedding_api_key": emb.get("api_key", ""),
        "embedding_base_url": emb.get("base_url", ""),
        "fallback_api_key": (fb.get("api_key", "")),
        "fallback_base_url": (fb.get("base_url", "")),
        "fallback_model": (fb.get("model", "")),
        "active": p.get("id") == active_id,
    }


# ── 写操作 ─────────────────────────────────────────────────────────────────

def normalize_profile_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """把前端扁平 payload 还原成 catalog 的嵌套 profile 结构。"""
    return {
        "id": str(raw.get("id") or "").strip(),
        "name": str(raw.get("name") or "").strip(),
        "binding": str(raw.get("binding") or "").strip(),
        "api_key": str(raw.get("api_key") or ""),
        "base_url": str(raw.get("base_url") or "").strip(),
        "api_version": str(raw.get("api_version") or "").strip(),
        "models": {
            "text": {"model": str(raw.get("text_model") or "").strip()},
            "fast": {"model": str(raw.get("fast_model") or "").strip()},
            "vision": {"model": str(raw.get("vision_model") or "").strip()},
            "embedding": {
                "model": str(raw.get("embedding_model") or "").strip(),
                "api_key": str(raw.get("embedding_api_key") or ""),
                "base_url": str(raw.get("embedding_base_url") or "").strip(),
            },
        },
        "fallback": {
            "api_key": str(raw.get("fallback_api_key") or ""),
            "base_url": str(raw.get("fallback_base_url") or "").strip(),
            "model": str(raw.get("fallback_model") or "").strip(),
        },
    }


def upsert_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """新增 / 更新指定 id 的 profile（写回 JSON）。返回写入后的 profile 原始对象。"""
    profile = normalize_profile_payload({**payload, "id": profile_id})
    with _atomic_update() as data:
        profiles = data.setdefault("profiles", [])
        for i, p in enumerate(profiles):
            if isinstance(p, dict) and p.get("id") == profile_id:
                profiles[i] = profile
                break
        else:
            profiles.append(profile)
    return profile


def delete_profile(profile_id: str) -> bool:
    with _atomic_update() as data:
        profiles = data.get("profiles") or []
        remaining = [p for p in profiles if not (isinstance(p, dict) and p.get("id") == profile_id)]
        if len(remaining) == len(profiles):
            return False
        data["profiles"] = remaining
        if data.get("active_profile") == profile_id:
            data["active_profile"] = remaining[0]["id"] if remaining else "default"
    return True


def set_active(profile_id: str) -> bool:
    with _atomic_update() as data:
        if not any(
            isinstance(p, dict) and p.get("id") == profile_id
            for p in data.get("profiles") or []
        ):
            return False
        data["active_profile"] = profile_id
    return True


__all__ = [
    "CATALOG_PATH",
    "Profile",
    "EmbeddingConfig",
    "FallbackConfig",
    "load_catalog",
    "save_catalog",
    "active_profile_id",
    "list_profiles",
    "get_profile",
    "profile_display_name",
    "profile_text_model",
    "profile_fast_model",
    "profile_vision_model",
    "profile_public_view",
    "profile_admin_view",
    "normalize_profile_payload",
    "upsert_profile",
    "delete_profile",
    "set_active",
]
