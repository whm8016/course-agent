"""IM 账号绑定码（短期一次性，打通 web User.id ↔ IM openid）。

绑定流程：
  1. web 用户调 POST /api/bot/bind/code 生成 6 位码（关联其 user_id，10 分钟有效）
  2. 用户在 QQ/飞书私聊 bot 发「绑定 <码>」
  3. bot loop 拦截该指令 → consume_bind_code 校验 → 写 UserSocialBinding
  4. 之后该 IM openid 的消息经 _resolve_user_id 命中同一 user_id
     → v3 read_l3_concat 注入 + turn 结束更新 v3 memory/graph → 长期记忆跨渠道

注意：绑定码存进程内存（带过期 + GC），多 worker 部署下不跨进程共享——单 bot 单
worker 场景够用；多 worker 需迁 Redis（REDIS_URL 已配置，预留升级点）。
"""
from __future__ import annotations

import re
import secrets
import time

_CODE_TTL = 600  # 10 分钟
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 去掉易混字符（0/O/1/I/L）
# 「绑定 <4-8位码>」（允许中间空格/全角，码字母数字）
_BIND_RE = re.compile(r"^\s*绑\s*定\s*([A-Za-z0-9]{4,8})\s*$")

# code(大写) -> (user_id, expire_ts)
_codes: dict[str, tuple[str, float]] = {}


def _gc() -> None:
    now = time.time()
    for c, (_, exp) in list(_codes.items()):
        if now > exp:
            _codes.pop(c, None)


def add_bind_code(user_id: str) -> str:
    """为 user_id 生成一个 6 位绑定码；同一 user 旧码失效（一个 user 同时只一个有效码）。"""
    code = "".join(secrets.choice(_ALPHABET) for _ in range(6))
    _codes[code] = (user_id, time.time() + _CODE_TTL)
    for c, (uid, _) in list(_codes.items()):
        if uid == user_id and c != code:
            _codes.pop(c, None)
    _gc()
    return code


def consume_bind_code(code: str) -> str | None:
    """校验并消费绑定码（一次性）；有效返回 user_id，否则 None。"""
    if not code:
        return None
    entry = _codes.pop(code.upper(), None)
    if not entry:
        return None
    user_id, expire = entry
    if time.time() > expire:
        return None
    return user_id


def parse_bind_command(content: str) -> str | None:
    """匹配「绑定 <码>」→ 返回大写码；不匹配返回 None。"""
    if not content:
        return None
    m = _BIND_RE.match(content)
    return m.group(1).upper() if m else None


__all__ = ["add_bind_code", "consume_bind_code", "parse_bind_command"]
