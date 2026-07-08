"""IM 账号绑定码（短期一次性，打通 web User.id ↔ IM openid）。

绑定流程：
  1. web 用户调 POST /api/bot/bind/code 生成 6 位码（关联其 user_id，10 分钟有效）
  2. 用户在 QQ/飞书私聊 bot 发「绑定 <码>」
  3. bot loop 拦截该指令 → consume_bind_code 校验 → 写 UserSocialBinding
  4. 之后该 IM openid 的消息经 _resolve_user_id 命中同一 user_id
     → v3 read_l3_concat 注入 + turn 结束更新 v3 memory/graph → 长期记忆跨渠道

存储（M-34）：绑定码存 Redis（``im:bindcode:<CODE>`` → user_id，TTL 10 分钟）。
多 worker 部署下任意 worker 生成的码、任意 worker（含 follower bot loop）都能校验，
跨进程共享。Redis 不可用时降级回进程内存（单 worker 仍可用，多 worker 降级期间
绑定可能失败需重试——安全侧，不损坏数据）。
"""
from __future__ import annotations

import logging
import re
import secrets
import time

logger = logging.getLogger(__name__)

_CODE_TTL = 600  # 10 分钟
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 去掉易混字符（0/O/1/I/L）
# 「绑定 <4-8位码>」（允许中间空格/全角，码字母数字）
_BIND_RE = re.compile(r"^\s*绑\s*定\s*([A-Za-z0-9]{4,8})\s*$")

_BINDKEY_PREFIX = "im:bindcode:"

# Redis 不可用时的进程内存降级（单 worker 兜底；多 worker 下仅 best-effort）
_fallback_codes: dict[str, tuple[str, float]] = {}


def _key(code: str) -> str:
    return f"{_BINDKEY_PREFIX}{code.upper()}"


def _gc_fallback() -> None:
    now = time.time()
    for c, (_, exp) in list(_fallback_codes.items()):
        if now > exp:
            _fallback_codes.pop(c, None)


def add_bind_code(user_id: str) -> str:
    """为 user_id 生成一个 6 位绑定码；同一 user 旧码失效（一个 user 同时只一个有效码）。

    跨进程（M-34）：码写 Redis（TTL 10 分钟），并记录 ``im:binduser:<user_id>`` 指向
    当前有效码，生成新码时让旧码立即失效（覆盖式）。Redis 不可用 → 降级进程内存。
    """
    code = "".join(secrets.choice(_ALPHABET) for _ in range(6))
    upper = code.upper()

    try:
        from core.db.cache import _get_pool

        r = _get_pool()
        # 让该 user 的旧码失效：读出旧码并删除（同一 user 同时只一个有效码）
        old = r.get(f"im:binduser:{user_id}")
        if old:
            r.delete(_key(str(old)))
        # 写新码 + 更新 user→code 映射（pipeline 批量）
        pipe = r.pipeline()
        pipe.set(_key(upper), user_id, ex=_CODE_TTL)
        pipe.set(f"im:binduser:{user_id}", upper, ex=_CODE_TTL)
        pipe.execute()
        return code
    except Exception:
        logger.warning("bind_code: redis unavailable, falling back to in-memory")
        _fallback_codes[upper] = (user_id, time.time() + _CODE_TTL)
        for c, (uid, _) in list(_fallback_codes.items()):
            if uid == user_id and c != upper:
                _fallback_codes.pop(c, None)
        _gc_fallback()
        return code


async def consume_bind_code(code: str) -> str | None:
    """校验并消费绑定码（一次性 GETDEL）；有效返回 user_id，否则 None。

    异步接口（bot loop 在 async 上下文调用）。Redis 不可用 → 降级进程内存。
    """
    if not code:
        return None
    upper = code.upper()

    try:
        from core.db.cache import _get_pool

        r = _get_pool()
        # GETDEL 原子取出并删除（一次性消费，防并发重复绑定）
        user_id = await r.getdel(_key(upper))
        if user_id:
            # 顺手清 user→code 映射（若一致）
            await r.delete(f"im:binduser:{user_id}")
            return str(user_id)
        return None
    except Exception:
        logger.warning("bind_code: redis unavailable, falling back to in-memory")
        entry = _fallback_codes.pop(upper, None)
        if not entry:
            return None
        uid, expire = entry
        if time.time() > expire:
            return None
        return uid


def parse_bind_command(content: str) -> str | None:
    """匹配「绑定 <码>」→ 返回大写码；不匹配返回 None。"""
    if not content:
        return None
    m = _BIND_RE.match(content)
    return m.group(1).upper() if m else None


__all__ = ["add_bind_code", "consume_bind_code", "parse_bind_command"]
