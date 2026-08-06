"""C-2 回归测试：014 补建 7 张缺失表 + 013 greenfield 守卫。

验证：
1. 014.upgrade() 在仅含 users 的空库上建出 7 张缺失表（含索引/唯一约束）。
2. 014 幂等：重复执行不报错（_table_exists 守卫，对齐 001-013 风格）。
3. 013.upgrade() 在 knowledge_bases 不存在时不抛 'no such table'（greenfield 守卫）。

跑迁移用 alembic MigrationContext + Operations.context 绑定 op（绕过 env.py 的
settings/async 依赖），直接 sync SQLite 库验证建表 DDL。StaticPool 保证 :memory:
库跨连接共享 schema（否则每条连接是独立空库）。
"""
import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.pool import StaticPool

VERSIONS = Path(__file__).resolve().parent.parent / "alembic" / "versions"

EXPECTED_TABLES = {
    "knowledge_bases",
    "kb_files",
    "notebook_categories",
    "notebook_entries",
    "notebook_entry_categories",
    "bot_notifications",
    "user_mcp_enrollments",
}


def _load_migration(filename: str):
    spec = importlib.util.spec_from_file_location(f"mig_{filename}", VERSIONS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engine_with_users():
    """空 SQLite 内存库 + users 基表（014/013 的 FK/查询依赖基表存在）。"""
    engine = sa.create_engine("sqlite:///:memory:", poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "CREATE TABLE users (id VARCHAR(32) PRIMARY KEY, "
                "role VARCHAR(16), is_admin BOOLEAN)"
            )
        )
    return engine


def test_014_creates_missing_tables():
    mod = _load_migration("014_create_missing_tables.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        insp = sa.inspect(conn)

        tables = set(insp.get_table_names())
        assert EXPECTED_TABLES.issubset(tables), EXPECTED_TABLES - tables

        # course_id 是 unique index（模型 unique=True, index=True）
        kb_indexes = {i["name"] for i in insp.get_indexes("knowledge_bases")}
        assert "ix_knowledge_bases_course_id" in kb_indexes
        assert "ix_knowledge_bases_join_code" in kb_indexes
        # notebook_categories 用显式 UniqueConstraint
        nc_uniques = {u["name"] for u in insp.get_unique_constraints("notebook_categories")}
        assert "uq_notebook_category_user_name" in nc_uniques
        # kb_files 复合索引
        kf_indexes = {i["name"] for i in insp.get_indexes("kb_files")}
        assert "idx_kb_files_kb" in kf_indexes


def test_014_idempotent():
    """重复跑 014 不报错（已存在的表被 _table_exists 跳过）。"""
    mod = _load_migration("014_create_missing_tables.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
            mod.upgrade()  # 第二次：幂等，不应抛"表已存在"
        tables = set(sa.inspect(conn).get_table_names())
    assert EXPECTED_TABLES.issubset(tables)


def test_015_creates_academic_tables():
    """015 在仅含 users 的空库上建出 course_schedules + grades（含索引/唯一约束）。"""
    mod = _load_migration("015_create_academic_tables.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        insp = sa.inspect(conn)

        tables = set(insp.get_table_names())
        assert {"course_schedules", "grades"}.issubset(tables)

        # course_schedules：单列索引 + 复合索引
        cs_indexes = {i["name"] for i in insp.get_indexes("course_schedules")}
        assert "ix_course_schedules_course_id" in cs_indexes
        assert "idx_course_schedule_course_weekday" in cs_indexes

        # grades：复合唯一约束 + 单列/复合索引
        g_uniques = {u["name"] for u in insp.get_unique_constraints("grades")}
        assert "uq_grade_student_course_item" in g_uniques
        g_indexes = {i["name"] for i in insp.get_indexes("grades")}
        assert "ix_grades_student_id" in g_indexes
        assert "idx_grade_student_course" in g_indexes


def test_015_idempotent():
    """重复跑 015 不报错（已存在的表被 _table_exists 跳过）。"""
    mod = _load_migration("015_create_academic_tables.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
            mod.upgrade()  # 第二次：幂等
        tables = set(sa.inspect(conn).get_table_names())
    assert {"course_schedules", "grades"}.issubset(tables)



def test_013_greenfield_skips_when_kb_table_absent():
    """013 在 knowledge_bases 不存在时安全跳过（C-2：堵 greenfield 顺序陷阱）。

    回归：旧 013 无条件 SELECT FROM knowledge_bases，greenfield 跑到此处时表由
    014 创建（014 尚未执行）→ 抛 'no such table'。修复后守卫命中即 return。
    """
    mod = _load_migration("013_backfill_kb_join_codes.py")
    engine = _engine_with_users()  # 无 knowledge_bases
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()  # 守卫命中 → return，不触发 SELECT
    # 走到这里即通过


def test_013_runs_when_kb_table_present():
    """knowledge_bases 已存在（含 join_code 列）时 013 正常跑 backfill，不误跳。"""
    mod = _load_migration("013_backfill_kb_join_codes.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        # 造一张空的 knowledge_bases（带 join_code 列），backfill 应查到 0 行
        conn.execute(
            sa.text(
                "CREATE TABLE knowledge_bases (id VARCHAR(32) PRIMARY KEY, "
                "join_code VARCHAR(16))"
            )
        )
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()  # 表存在 → 守卫不命中 → 正常执行 backfill（0 行）


def test_020_creates_memory_episodes():
    """020 在仅含 users 的空库上建出 memory_episodes（含唯一约束 + 索引）。"""
    mod = _load_migration("020_memory_episodes.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        insp = sa.inspect(conn)

        tables = set(insp.get_table_names())
        assert "memory_episodes" in tables

        # (session_id, turn_id) 唯一约束——幂等基础
        uq = {u["name"] for u in insp.get_unique_constraints("memory_episodes")}
        assert "uq_episodes_session_turn" in uq
        # outbox 扫描索引 + user 维度索引
        idx = {i["name"] for i in insp.get_indexes("memory_episodes")}
        assert "idx_episodes_outbox" in idx
        assert "idx_episodes_user" in idx


def test_021_creates_knowledge_mastery():
    """021 在仅含 users 的空库上建出 knowledge_mastery（含唯一约束 + 索引）。"""
    mod = _load_migration("021_knowledge_mastery.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        insp = sa.inspect(conn)

        tables = set(insp.get_table_names())
        assert "knowledge_mastery" in tables

        # (user_id, course_id, kp_id) 唯一约束——追加观测靠 UPDATE 不新增行
        uq = {u["name"] for u in insp.get_unique_constraints("knowledge_mastery")}
        assert "uq_mastery_user_course_kp" in uq
        idx = {i["name"] for i in insp.get_indexes("knowledge_mastery")}
        assert "idx_mastery_user_course" in idx


def test_migration_chain_head_is_021():
    """021 被识别为唯一 head，且 down_revision 链回 base（链不断、无分叉）。

    防止 revision/down_revision 写错导致 alembic 不识别或多 head。ScriptDirectory
    只解析 versions 目录，不连 DB、不 exec env.py，故无需 settings/env 配置。
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(VERSIONS.parent.parent / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    assert script.get_heads() == ["021"]

    revisions = [r.revision for r in script.walk_revisions()]
    # head 能倒走回 base，证明链完整（021→020→019→018→017→016→015→…→001）
    assert "021" in revisions and "020" in revisions and "019" in revisions
    assert "018" in revisions and "017" in revisions and "016" in revisions and "001" in revisions
