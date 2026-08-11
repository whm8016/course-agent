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

import pytest

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


def test_023_creates_learning_events():
    """023 在仅含 users 的空库上建出 learning_events（含两个 cron 访问索引）。"""
    mod = _load_migration("023_learning_events.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        insp = sa.inspect(conn)

        tables = set(insp.get_table_names())
        assert "learning_events" in tables

        idx = {i["name"] for i in insp.get_indexes("learning_events")}
        # FAQ 聚类 cron（course,verb,time）+ 学生 rollup cron（actor,course,time）
        assert "idx_events_course_verb_time" in idx
        assert "idx_events_actor_course_time" in idx


def test_024_creates_llm_usage_tables():
    """024 在仅含 users 的空库上建出 llm_usage_records + llm_usage_daily（含索引/唯一约束）。"""
    mod = _load_migration("024_llm_usage.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        insp = sa.inspect(conn)

        tables = set(insp.get_table_names())
        assert "llm_usage_records" in tables
        assert "llm_usage_daily" in tables

        # 明细表 cron 访问索引（按人/按课/全局时间窗口）
        rec_idx = {i["name"] for i in insp.get_indexes("llm_usage_records")}
        assert "idx_usage_user_time" in rec_idx
        assert "idx_usage_course_time" in rec_idx
        assert "idx_usage_created" in rec_idx

        # 日汇总：幂等唯一约束 + 按天索引
        daily_idx = {i["name"] for i in insp.get_indexes("llm_usage_daily")}
        assert "uq_usage_daily" in daily_idx
        assert "idx_usage_daily_day" in daily_idx


def test_025_creates_learning_rollups():
    """025 建出 course_daily_rollup + student_course_rollup（含唯一约束 + 索引）。"""
    mod = _load_migration("025_learning_rollups.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        insp = sa.inspect(conn)

        tables = set(insp.get_table_names())
        assert {"course_daily_rollup", "student_course_rollup"}.issubset(tables)

        cd_uq = {u["name"] for u in insp.get_unique_constraints("course_daily_rollup")}
        assert "uq_course_daily_rollup" in cd_uq
        sc_uq = {u["name"] for u in insp.get_unique_constraints("student_course_rollup")}
        assert "uq_student_course_rollup" in sc_uq
        sc_idx = {i["name"] for i in insp.get_indexes("student_course_rollup")}
        assert "idx_student_course_rollup_course" in sc_idx


def test_027_creates_course_faq():
    """027 建出 course_faq（高频问题语义聚类读模型，含 course_id 索引）。"""
    mod = _load_migration("027_course_faq.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        insp = sa.inspect(conn)
        assert "course_faq" in set(insp.get_table_names())
        idx = {i["name"] for i in insp.get_indexes("course_faq")}
        assert "ix_course_faq_course_id" in idx


def test_migration_chain_head_is_031():
    """031 被识别为唯一 head，且 down_revision 链回 base（链不断、无分叉）。

    防止 revision/down_revision 写错导致 alembic 不识别或多 head。ScriptDirectory
    只解析 versions 目录，不连 DB、不 exec env.py，故无需 settings/env 配置。
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(VERSIONS.parent.parent / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    assert script.get_heads() == ["031"]

    revisions = [r.revision for r in script.walk_revisions()]
    # head 能倒走回 base，证明链完整（025→024→023→022→021→020→019→018→017→016→015→…→001）
    assert {"030", "029", "028", "022", "021", "020", "019", "018",
            "017", "016", "015", "001"}.issubset(revisions)


def test_029_drops_users_dead_memory_cols():
    """029 删除 users 的 4 个死记忆列（summary/profile/scope/preferences_memory，P3）。"""
    mod = _load_migration("029_drop_users_dead_memory_cols.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        # users 基表只有 id/role/is_admin；先补 4 列模拟 029 前状态
        for col, typ, dflt in [
            ("summary_memory", "TEXT", "''"),
            ("profile_memory", "TEXT", "'{}'"),
            ("scope_memory", "TEXT", "''"),
            ("preferences_memory", "TEXT", "''"),
        ]:
            conn.execute(sa.text(
                f"ALTER TABLE users ADD COLUMN {col} {typ} NOT NULL DEFAULT {dflt}"
            ))
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        cols = {c["name"] for c in sa.inspect(conn).get_columns("users")}
    assert {"summary_memory", "profile_memory", "scope_memory",
            "preferences_memory"}.isdisjoint(cols)


@pytest.mark.xfail(
    reason="SQLite 不支持 ALTER constraint（002 create_unique_constraint 等），全链需 batch mode；"
           "create_all 是 SQLite 测试基座（实证 P4.1 kill create_all 不可行的根因）",
    strict=False,
)
def test_full_upgrade_head_produces_all_tables():
    """全链 alembic upgrade head（SQLite）产出全部表，防 migrations<->model 漂移（P4.1 安全网）。

    对标 create_all 测试基座：确保生产走 Alembic 也能建出完整 schema。逐 revision 顺序跑
    upgrade()，绕过 env.py 的 settings/async 依赖。create_all 仍是 function-scoped 测试
    fixture 的快速基座（每测跑 29 条迁移不可行），本测试作为「迁移链产出完整 schema」的守卫。
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(VERSIONS.parent.parent / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    revisions = list(reversed(list(script.walk_revisions())))  # base->head

    engine = sa.create_engine("sqlite:///:memory:", poolclass=StaticPool)
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            for rev in revisions:
                mod = _load_migration(Path(rev.path).name)
                mod.upgrade()
        tables = set(sa.inspect(conn).get_table_names())

    for t in ["users", "sessions", "messages", "knowledge_bases", "kb_builds",
              "kb_files", "notebook_entries", "notebook_categories", "enrollments",
              "course_schedules", "grades", "memory_episodes", "knowledge_mastery",
              "learning_events", "course_daily_rollup", "student_course_rollup",
              "llm_usage_records", "llm_usage_daily", "research_checkpoints",
              "bot_notifications", "teacher_applications", "teacher_invites"]:
        assert t in tables, f"full upgrade head missing table: {t}"


def test_031_adds_users_graph_version():
    """031 给 users 加 graph_version（OCC 并发保护，P3 修 bug）。"""
    mod = _load_migration("031_users_graph_version.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        cols = {c["name"] for c in sa.inspect(conn).get_columns("users")}
    assert "graph_version" in cols


def test_028_drops_kb_dead_status_cols():
    """028 删除 knowledge_bases 的 7 个死状态列（status/progress/progress_msg/
    chunks_done/chunks_total/token_estimate/error_msg），保留 file_count（P2）。"""
    mod = _load_migration("028_drop_kb_dead_status_cols.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        # 造一张带死列的 knowledge_bases（模拟 014 建表后、028 前）
        conn.execute(sa.text(
            "CREATE TABLE knowledge_bases (id VARCHAR(32) PRIMARY KEY, "
            "course_id VARCHAR(64), name VARCHAR(256), file_count INTEGER, "
            "status VARCHAR(32), progress INTEGER, progress_msg TEXT, "
            "chunks_done INTEGER, chunks_total INTEGER, token_estimate INTEGER, error_msg TEXT)"
        ))
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        cols = {c["name"] for c in sa.inspect(conn).get_columns("knowledge_bases")}
    # 7 死列已删
    assert {"status", "progress", "progress_msg", "chunks_done",
            "chunks_total", "token_estimate", "error_msg"}.isdisjoint(cols)
    # file_count 保留
    assert "file_count" in cols


def test_026_adds_course_id_to_messages_and_notebook():
    """026 给 messages + notebook_entries 加 course_id（NOT NULL 默认''）+ 索引，
    并从 sessions 关联子查询回填存量行（P1 租主硬化）。"""
    mod = _load_migration("026_messages_notebook_course_id.py")
    engine = _engine_with_users()
    with engine.begin() as conn:
        # 026 依赖 sessions + messages + notebook_entries 三张表
        conn.execute(sa.text(
            "CREATE TABLE sessions (id VARCHAR(32) PRIMARY KEY, course_id VARCHAR(64))"
        ))
        conn.execute(sa.text(
            "CREATE TABLE messages (id VARCHAR(32) PRIMARY KEY, session_id VARCHAR(32), "
            "role VARCHAR(16), content TEXT, msg_type VARCHAR(16), metadata TEXT, created_at FLOAT)"
        ))
        conn.execute(sa.text(
            "CREATE TABLE notebook_entries (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id VARCHAR(32), session_id VARCHAR(32), question_id VARCHAR(64))"
        ))
        conn.execute(sa.text("INSERT INTO sessions (id, course_id) VALUES ('s1','c1'),('s2','c2')"))
        conn.execute(sa.text("INSERT INTO messages (id, session_id) VALUES ('m1','s1')"))
        conn.execute(sa.text(
            "INSERT INTO notebook_entries (user_id, session_id, question_id) VALUES ('u1','s2','q1')"
        ))

        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            mod.upgrade()
        insp = sa.inspect(conn)

        # 列 + 索引已加
        assert "course_id" in {c["name"] for c in insp.get_columns("messages")}
        assert "course_id" in {c["name"] for c in insp.get_columns("notebook_entries")}
        assert "idx_messages_course" in {i["name"] for i in insp.get_indexes("messages")}
        assert "idx_notebook_entries_course" in {i["name"] for i in insp.get_indexes("notebook_entries")}
        # 回填：从 sessions 反查 course_id
        assert conn.execute(sa.text("SELECT course_id FROM messages WHERE id='m1'")).scalar() == "c1"
        assert conn.execute(
            sa.text("SELECT course_id FROM notebook_entries WHERE session_id='s2'")
        ).scalar() == "c2"
