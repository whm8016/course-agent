"""模型上下文窗口探测 + 持久化缓存（通用上下文管理重构·窗口探测层）。

对标 DeepTutor ``detect_context_window``（``services/config/context_window_detection.py``）的
「探测优先」架构（调研依据见 plan「通用上下文管理重构」决策三）：探测在请求热路径之外跑
（``main.py`` lifespan 启动预热 / admin 手动重探），结果落 ``data/context_window_cache.json``，
热路径 ``context_window.resolve_effective_window`` 只做同步查表，零网络开销。照搬 DeepTutor
的分工——配置期探测 + 持久化 + 运行时同步读。

探测实现：``GET {base_url}/models``，递归扫描 payload 找 8 个候选字段名
（``context_window`` / ``context_length`` / ``max_context_tokens`` / ``max_input_tokens`` /
``input_token_limit`` / ``max_prompt_tokens`` / ``max_model_len`` / ``max_sequence_length``），
覆盖 OpenAI 系 / vLLM / Gemini 等不同命名；model id 做别名匹配（精确 + ``/`` ``:`` 分隔的
partial，处理 ``provider/model`` 这类前缀）。

**必须承认的局限**：原生 OpenAI 与部分云厂商的 ``/models`` 只返回 ``id``/``created``/``owned_by``，
不暴露窗口，探测会失败——这正是仍需保留 ``context_window._MODEL_WINDOWS`` 表作第 3 级的原因。
但对 vLLM / Ollama / LiteLLM 代理 / OpenRouter 这类自托管与聚合端点，探测能拿到准确值。
admin 端点返回的 ``source`` 区分 ``probe``（探测拿到）/ ``table``（探测失败退回表）/ ``heuristic``
（表也没有）/ ``explicit``（显式配置），让运维能判断当前用的是哪一级。

缓存键 = ``(normalized_base_url, normalized_model)``，值 = ``{"window": int, "detected_at": iso}``，
带 TTL（默认 7 天，模型窗口变更频率低）。

**热路径零开销**：``read_probe_cache`` 用进程内 mtime 缓存——``os.stat`` 便宜，mtime 未变则
复用已解析 dict，不重复读盘+解析 JSON。写用 temp + ``os.replace`` 原子替换，防热路径读到半截 JSON。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import httpx

from core.observability import log_flow
from settings import BASE_DIR, get_settings

logger = logging.getLogger(__name__)

# 8 个候选字段名（对标 DeepTutor _CONTEXT_WINDOW_KEYS，覆盖 OpenAI系/vLLM/Gemini 命名变体）。
_CONTEXT_WINDOW_KEYS = (
    "context_window",
    "context_length",
    "max_context_tokens",
    "max_input_tokens",
    "input_token_limit",
    "max_prompt_tokens",
    "max_model_len",
    "max_sequence_length",
)

_CACHE_FILENAME = "context_window_cache.json"

# ---------------------------------------------------------------------------
# 进程内 mtime 缓存：(mtime, parsed_dict)。mtime 未变则复用，避免每轮重新读+解析 JSON。
# 用模块级 tuple + global 声明，与 context_builder._encoding 同款风格。
# ---------------------------------------------------------------------------
_mem_cache: tuple[float, dict[str, Any]] = (0.0, {})


def _cache_path() -> str:
    """探测缓存文件路径（与 model_catalog.json 同目录，gitignore 的运行时数据）。"""
    return os.path.join(BASE_DIR, "data", _CACHE_FILENAME)


def _reset_mem_cache() -> None:
    """清进程内缓存（测试用：强制下次重新读盘）。"""
    global _mem_cache
    _mem_cache = (0.0, {})


def _normalize_key(base_url: str, model: str) -> tuple[str, str]:
    """归一化缓存键：base_url 去尾斜杠小写、model 小写。"""
    return ((base_url or "").strip().rstrip("/").lower(), (model or "").strip().lower())


def _cache_entry_key(base_url: str, model: str) -> str:
    """拼成 JSON 顶层 key 字符串（``base_url|model``，两段均已归一化）。"""
    b, m = _normalize_key(base_url, model)
    return f"{b}|{m}"


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _load_cache() -> dict[str, Any]:
    """读缓存文件 -> dict。stat mtime 未变则复用进程内缓存（热路径零开销）。

    文件缺失 -> 空 dict（不抛，绝不影响热路径）。解析失败 -> 记 warning + 空 dict，并缓存
    当前 mtime，避免每轮重复读盘解析坏文件（坏文件 mtime 不变，复用空 dict 直到文件被改）。
    """
    global _mem_cache
    path = _cache_path()
    try:
        mtime = os.stat(path).st_mtime
    except FileNotFoundError:
        _mem_cache = (0.0, {})
        return {}
    except OSError:
        # stat 异常（权限等）：退回上次缓存内容，不阻断热路径。
        return _mem_cache[1]
    if mtime == _mem_cache[0] and _mem_cache[1] is not None:
        # mtime 未变则复用已解析 dict（含空 dict：文件被清空也复用，不重复读盘解析）。
        return _mem_cache[1]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError):
        log_flow("context.window_probe_cache_read_failed", logger=logger,
                 level=logging.WARNING, path=path)
        data = {}
    _mem_cache = (mtime, data)
    return data


def read_probe_cache(base_url: str, model: str) -> int | None:
    """同步读探测缓存（热路径用，零网络）。

    返回未过期窗口值；无条目 / 过期 / 时间戳损坏 / 文件缺失 -> None（调用方退回
    ``_MODEL_WINDOWS`` 表或 heuristic）。TTL 取 ``settings.context_budget.probe_cache_ttl_s``。
    """
    cfg = get_settings()
    key = _cache_entry_key(base_url, model)
    b, m = _normalize_key(base_url, model)
    if not b or not m:
        return None
    entry = _load_cache().get(key)
    if not isinstance(entry, dict):
        return None
    window = entry.get("window")
    if not isinstance(window, int) or window <= 0:
        return None
    detected_at = _parse_iso(entry.get("detected_at"))
    if detected_at is None:
        return None  # 时间戳损坏 -> 视为过期，触发重探
    age = (datetime.now(timezone.utc) - detected_at).total_seconds()
    if age > cfg.context_budget.probe_cache_ttl_s:
        return None  # 过期
    return window


def write_probe_cache(base_url: str, model: str, window: int) -> None:
    """写探测结果到缓存文件（原子写：temp + os.replace，防热路径读到半截 JSON）。

    多 worker 并发写同一 key：``os.replace`` 原子，last-writer-wins——各 worker 对同一
    ``(base_url, model)`` 探测结果相同，无不一致。写后更新进程内 mtime 缓存。
    """
    if not isinstance(window, int) or window <= 0:
        return
    b, m = _normalize_key(base_url, model)
    if not b or not m:
        return
    key = f"{b}|{m}"
    data = _load_cache()
    data[key] = {
        "window": window,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except OSError:
        log_flow("context.window_probe_cache_write_failed", logger=logger,
                 level=logging.WARNING, path=path)
        return
    # 更新进程内缓存，下次读复用（其他 worker 各自下次 stat 见新 mtime 自动重载）。
    try:
        _mem_cache = (os.stat(path).st_mtime, data)
    except OSError:
        _reset_mem_cache()


# ---------------------------------------------------------------------------
# payload 解析（对标 DeepTutor _recursive_context_window / _extract_context_window_from_payload）
# ---------------------------------------------------------------------------
def _coerce_positive_int(value: Any) -> int | None:
    """把任意类型转成 >0 的 int；bool/负数/空串 -> None。"""
    if isinstance(value, bool):
        return None  # bool 是 int 子类，须先排除
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str):
        try:
            iv = int(value.strip())
        except ValueError:
            return None
        return iv if iv > 0 else None
    return None


def _model_aliases(model: str) -> set[str]:
    """model id 别名集合：原值 + ``/`` 与 ``:`` 分隔后段（处理 ``provider/model``、``model:v2``）。"""
    value = (model or "").strip().lower()
    if not value:
        return set()
    aliases = {value}
    if "/" in value:
        aliases.add(value.split("/", 1)[1])
    if ":" in value:
        aliases.add(value.split(":", 1)[1])
    return {a for a in aliases if a}


def _record_identities(item: Mapping[str, Any]) -> set[str]:
    """单条 model 记录的别名集合（扫 id / model / name 三个常见字段）。"""
    ids: set[str] = set()
    for k in ("id", "model", "name"):
        ids.update(_model_aliases(str(item.get(k, "") or "")))
    return ids


def _recursive_context_window(value: Any) -> int | None:
    """递归扫 value（dict/list）找 8 个候选字段名，命中即返回首个 >0 int。"""
    if isinstance(value, Mapping):
        for k in _CONTEXT_WINDOW_KEYS:
            parsed = _coerce_positive_int(value.get(k))
            if parsed is not None:
                return parsed
        for nested in value.values():
            parsed = _recursive_context_window(nested)
            if parsed is not None:
                return parsed
    elif isinstance(value, list):
        for nested in value:
            parsed = _recursive_context_window(nested)
            if parsed is not None:
                return parsed
    return None


def _iter_model_records(payload: Any) -> Iterable[Mapping[str, Any]]:
    """从 payload 提取 model 记录列表：裸 list，或 dict 里 data/models/result/items 之一。"""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item
        return
    if not isinstance(payload, Mapping):
        return
    for k in ("data", "models", "result", "items"):
        items = payload.get(k)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, Mapping):
                    yield item


def _extract_context_window_from_payload(payload: Any, model: str) -> int | None:
    """从 /models payload 提取目标 model 的窗口：精确别名匹配优先，其次 ``/`` 分隔 partial。"""
    target = _model_aliases(model)
    if not target:
        return None
    exact: list[Mapping[str, Any]] = []
    partial: list[Mapping[str, Any]] = []
    for item in _iter_model_records(payload):
        identities = _record_identities(item)
        if not identities:
            continue
        if identities & target:
            exact.append(item)
            continue
        # partial：处理 provider/model 这类前缀（OpenRouter 等）
        if any(ie.endswith(f"/{a}") or a.endswith(f"/{ie}")
               for ie in identities for a in target):
            partial.append(item)
    for item in [*exact, *partial]:
        parsed = _recursive_context_window(item)
        if parsed is not None:
            return parsed
    return None


async def detect_context_window(model: str, base_url: str, api_key: str) -> int | None:
    """异步 ``GET {base_url}/models`` 探测真实窗口（对标 DeepTutor ``_detect_from_models_endpoint``）。

    递归扫 8 候选字段 + model id 别名匹配。HTTP 失败 / 非 200 / 无匹配 -> None（调用方退回
    ``_MODEL_WINDOWS`` 表）。超时取 ``settings.context_budget.probe_timeout_s``（默认 12s）。
    """
    base = (base_url or "").strip()
    m = (model or "").strip()
    if not base or not m:
        return None
    url = f"{base.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timeout = get_settings().context_budget.probe_timeout_s
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            log_flow("context.window_probe_http_non200", logger=logger,
                     level=logging.DEBUG, url=url, status=resp.status_code)
            return None
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - 探测 best-effort，任何异常都降级为 None
        log_flow("context.window_probe_http_failed", logger=logger,
                 level=logging.DEBUG, url=url, error=str(exc))
        return None
    return _extract_context_window_from_payload(payload, m)


async def warmup_probe(
    models: list[str] | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """启动预热 / admin 手动重探：探测给定模型（默认 active profile 的 text+fast）。

    - ``force=False``（启动预热）：缓存已有未过期条目则跳过该模型（避免每重启都打 /models）。
    - ``force=True``（admin 重探）：无视缓存 TTL，强制重探并覆写。
    best-effort：单模型失败不影响其余。返回每模型探测结果供调用方日志/展示：

        ``{"model", "base_url", "detected": int|None, "skipped_cached": bool}``
    """
    cfg = get_settings()
    if not cfg.context_budget.probe_enabled:
        return []
    burl = base_url if base_url is not None else cfg.llm.base_url
    akey = api_key if api_key is not None else cfg.llm.api_key.get_secret_value()
    if models is None:
        models = [cfg.llm.text_model, cfg.llm.fast_model]
    # 去重保序
    seen: set[str] = set()
    uniq = [m for m in models if m and m not in seen and not seen.add(m)]
    results: list[dict[str, Any]] = []
    for m in uniq:
        if not force:
            cached = read_probe_cache(burl, m)
            if cached:
                results.append({"model": m, "base_url": burl,
                                "detected": cached, "skipped_cached": True})
                continue
        detected = await detect_context_window(m, burl, akey)
        if detected:
            write_probe_cache(burl, m, detected)
        results.append({"model": m, "base_url": burl,
                        "detected": detected, "skipped_cached": False})
    return results


__all__ = [
    "detect_context_window",
    "read_probe_cache",
    "write_probe_cache",
    "warmup_probe",
]
