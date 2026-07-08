"""IM 绑定码单测：指令解析 + 生成/消费（一次性 + 同 user 覆盖旧码）。

M-34：consume_bind_code 已改为 async + Redis 存储；测试环境 Redis 不可用（DB__REDIS_URL
=memory:// 不是合法 redis URL）→ 自动降级进程内存，测试覆盖降级路径下的语义。
"""

from core.bot.binding import add_bind_code, consume_bind_code, parse_bind_command


def test_parse_bind_command():
    assert parse_bind_command("绑定 ABC123") == "ABC123"
    assert parse_bind_command("  绑定  abc123  ") == "ABC123"
    assert parse_bind_command("绑定AB12") == "AB12"
    assert parse_bind_command("绑定 1234") == "1234"  # 4 位（下限）
    assert parse_bind_command("绑定 123") is None  # 3 位太短
    assert parse_bind_command("你好") is None
    assert parse_bind_command("") is None
    assert parse_bind_command("绑定码 ABC123") is None  # 非「绑定」开头词


async def test_add_consume_bind_code():
    code = add_bind_code("user-1")
    assert len(code) == 6
    assert await consume_bind_code(code) == "user-1"
    assert await consume_bind_code(code) is None  # 一次性
    assert await consume_bind_code("NOTEXST") is None  # 无效


async def test_consume_case_insensitive():
    code = add_bind_code("user-ci")
    assert await consume_bind_code(code.lower()) == "user-ci"  # 大小写不敏感


async def test_add_bind_code_replaces_old():
    c1 = add_bind_code("user-2")
    c2 = add_bind_code("user-2")
    assert c1 != c2
    assert await consume_bind_code(c1) is None  # 旧码失效
    assert await consume_bind_code(c2) == "user-2"
