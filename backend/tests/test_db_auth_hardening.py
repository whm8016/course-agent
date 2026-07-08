"""DB/鉴权模块加固回归测试（H-17 / H-18 / M-42 / M-45 / M-46）。

每个测试钉死一个修复点，防止后续重构把它改回漏洞形态：
- H-17/M-46：连接池接入 settings 并按 worker 缩放（总连接数 ≤ Postgres 上限）。
- H-18：注册名为 admin 的用户不再自动获得 admin 角色（攻击者视角）。
- M-42：session_scope 短生命周期（进/出即开闭连接，异常回滚）。
- M-45：API key 加密失败显式报错，拒绝静默落库空 key。

纯单元测试，不依赖 conftest 的 app 起停（除 H-18 走真实 register 端点）。
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# H-17 / M-46：连接池接入 settings + 按 worker 缩放
# ---------------------------------------------------------------------------

def test_engine_kwargs_postgres_scales_by_workers():
    """Postgres URL 下 pool_size/max_overflow 取自 settings 并按 backend_workers 整除。

    旧实现裸读 os.getenv，4 worker × (10+15)=100 连接易打爆 Postgres max_connections。
    修复后每 worker 自缩：pool=max(2,10//4)=2, overflow=max(1,15//4)=3 → 单 worker 5。
    """
    from core.db import database

    # 强制非 sqlite 分支 + 固定 settings（patch DATABASE_URL 与 get_settings 返回值）
    fake_db = type("S", (), {"pool_size": 10, "max_overflow": 15})()
    fake_settings = type("S", (), {"backend_workers": 4, "db": fake_db})()
    with patch.object(database, "DATABASE_URL", "postgresql+asyncpg://u:p@h/db"), \
         patch.object(database, "get_settings", return_value=fake_settings):
        kw = database._engine_kwargs()
    assert kw["pool_size"] == 2          # 10 // 4 = 2
    assert kw["max_overflow"] == 3       # 15 // 4 = 3
    assert kw["pool_pre_ping"] is True
    assert kw["pool_recycle"] == 1800
    # 总连接上限 = workers × (pool + overflow) = 4 × 5 = 20（远低于 PG 默认 100）
    total = 4 * (kw["pool_size"] + kw["max_overflow"])
    assert total == 20


def test_engine_kwargs_pool_floor_is_2():
    """pool_size 缩放下限为 2：单 worker 也不会被压到 0/1 导致饿死。"""
    from core.db import database

    fake_db = type("S", (), {"pool_size": 5, "max_overflow": 3})()
    fake_settings = type("S", (), {"backend_workers": 8, "db": fake_db})()
    with patch.object(database, "DATABASE_URL", "postgresql+asyncpg://u:p@h/db"), \
         patch.object(database, "get_settings", return_value=fake_settings):
        kw = database._engine_kwargs()
    # 5//8 = 0 → floor 2；3//8 = 0 → overflow floor 1
    assert kw["pool_size"] == 2
    assert kw["max_overflow"] == 1


def test_engine_kwargs_sqlite_unchanged():
    """SQLite 不走池（保持 check_same_thread=False），不受缩放影响。"""
    from core.db import database

    with patch.object(database, "DATABASE_URL", "sqlite+aiosqlite:///:memory:"):
        kw = database._engine_kwargs()
    assert kw == {"connect_args": {"check_same_thread": False}}


# ---------------------------------------------------------------------------
# H-18：注册名为 admin 的用户不再自动提权（攻击者视角）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h18_registering_admin_username_is_not_admin(client: AsyncClient):
    """攻击者注册 username='admin'：仅得到 student 角色，is_admin=False。

    旧实现 create_user 里 `if username == ADMIN_USERNAME: role = "admin"` → 撞名即提权。
    修复后该逻辑删除，admin 仅由安全通道（DBA bootstrap / 现有 admin 显式授权）授予。
    """
    r = await client.post(
        "/api/auth/register",
        json={"username": "admin", "password": "attacker-pass-123", "display_name": "Attacker"},
    )
    # 409（已被 conftest 注册过）或 200，两种都要证明非 admin
    assert r.status_code in (200, 409), r.text
    if r.status_code == 200:
        user = r.json()["user"]
        assert user["role"] != "admin", "H-18 回归：撞名 admin 自动提权！"
        assert user["is_admin"] is False


@pytest.mark.asyncio
async def test_h18_arbitrary_admin_like_username_not_promoted(client: AsyncClient):
    """任何用户名（含 admin/Admin/ADMIN 变体）都不得自动提权。"""
    for name in ("Admin", "ADMIN", "admin"):
        uname = f"{name}_{__import__('os').urandom(2).hex()}"
        r = await client.post(
            "/api/auth/register",
            json={"username": uname, "password": "pass12345", "display_name": "X"},
        )
        assert r.status_code == 200, r.text
        user = r.json()["user"]
        assert user["role"] == "student"
        assert user["is_admin"] is False


# ---------------------------------------------------------------------------
# M-42：session_scope 短生命周期
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m42_session_scope_commits_on_clean_exit():
    """session_scope 正常退出 commit：写入在退出后可读。"""
    from sqlalchemy import text

    from core.db.database import AsyncSessionLocal, session_scope

    # 用一个临时表验证 commit 生效（SQLite 内存 + StaticPool 共享 schema）
    async with session_scope() as db:
        await db.execute(text("CREATE TABLE IF NOT EXISTS _m42_probe (v INTEGER)"))
        await db.execute(text("INSERT INTO _m42_probe (v) VALUES (1)"))

    async with AsyncSessionLocal() as db2:
        row = (await db2.execute(text("SELECT v FROM _m42_probe"))).first()
    assert row is not None and row[0] == 1


@pytest.mark.asyncio
async def test_m42_session_scope_rolls_back_on_exception():
    """session_scope 异常退出回滚：写入不残留。"""
    from sqlalchemy import text

    from core.db.database import AsyncSessionLocal, session_scope

    marker = "m42_rollback_probe"
    with pytest.raises(RuntimeError):
        async with session_scope() as db:
            await db.execute(text(f"CREATE TABLE IF NOT EXISTS {marker} (v INTEGER)"))
            await db.execute(text(f"INSERT INTO {marker} (v) VALUES (99)"))
            raise RuntimeError("boom")  # 触发回滚

    # CREATE TABLE 是 DDL（SQLite 自动提交，不受事务回滚影响），但 INSERT 应被回滚
    async with AsyncSessionLocal() as db2:
        rows = (await db2.execute(text(f"SELECT v FROM {marker}"))).fetchall()
    assert rows == [], "M-42 回归：session_scope 异常未回滚，写入残留"


# ---------------------------------------------------------------------------
# M-45：API key 加密失败显式报错（不再静默存空 key）
# ---------------------------------------------------------------------------

def test_m45_resolve_key_raises_when_encryption_returns_empty():
    """非空 raw 却加密出空串 → ValueError，拒绝落库空 key。

    旧实现 encrypt_secret 异常返回 ""，_resolve_key 直接落库空串 → 用户以为存了 key，
    运行时 LLM 调用 401。修复后对「非空 raw + 空密文」显式报错。
    """
    from core.db import user_llm_provider

    with patch.object(user_llm_provider, "encrypt_secret", return_value=""):
        with pytest.raises(ValueError, match="加密失败"):
            user_llm_provider._resolve_key("sk-real-key", existing_encrypted="")


def test_m45_resolve_key_empty_raw_preserves_existing():
    """raw 留空 → 保留原 key（不报错，向后兼容「只改模型不重输 key」）。"""
    from core.db import user_llm_provider

    with patch.object(user_llm_provider, "encrypt_secret", return_value="should-not-be-called"):
        result = user_llm_provider._resolve_key("", existing_encrypted="old-cipher")
    assert result == "old-cipher"


def test_m45_resolve_key_empty_raw_and_no_existing_yields_empty():
    """raw 空 + 无既有值 → 空串（首配即留空，正常业务，不报错）。"""
    from core.db import user_llm_provider

    with patch.object(user_llm_provider, "encrypt_secret", return_value=""):
        result = user_llm_provider._resolve_key("", existing_encrypted="")
    assert result == ""
