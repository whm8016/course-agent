"""kb_builds 双后端构建状态回归（Phase 1）。

验证数据模型：一门课可同时持有 LightRAG + pgvector 两套索引，各自 status 独立；
_kb_to_dict / _kb_to_course 从 kb_builds 聚合（不再读 KB 行旧 status 列）；
get_or_create_build 幂等、不同 backend 各一行。

不依赖 main/client fixture（本地 venv 缺 pythonjsonlogger 跑不动 main）——聚合/序列化用
内存模型，get_or_create 用 init_db 直连 sqlite。
"""
from __future__ import annotations

from core.db.database import KnowledgeBase, KbBuild, aggregate_build_status


def _b(status: str, backend: str = "lightrag", **kw) -> KbBuild:
    # 默认 chunks_total=1：aggregate_build_status 的 ready 要求 chunks>0（防空库误判就绪）。
    # 需要空库场景（如 0-chunk ready）显式传 chunks_total=0。
    kw.setdefault("chunks_total", 1)
    return KbBuild(backend=backend, status=status, **kw)


def test_aggregate_build_status_priority():
    """聚合优先级：indexing > error > paused > ready > pending；空 → pending。"""
    assert aggregate_build_status([]) == "pending"
    assert aggregate_build_status([_b("ready")]) == "ready"
    assert aggregate_build_status([_b("ready"), _b("indexing")]) == "indexing"
    assert aggregate_build_status([_b("ready"), _b("error")]) == "error"
    assert aggregate_build_status([_b("paused"), _b("ready")]) == "paused"
    assert aggregate_build_status([_b("pending"), _b("pending")]) == "pending"
    # 两后端都 ready（双索引就绪）
    assert aggregate_build_status(
        [_b("ready", "lightrag"), _b("ready", "llamaindex_pg")]
    ) == "ready"
    # ready 但 chunks_total=0（空索引）→ 不算就绪，落 pending（防空库绿徽章骗用户）
    assert aggregate_build_status([_b("ready", chunks_total=0)]) == "pending"
    # 一后端 0-chunk ready、另一后端有 chunks ready → 仍有可用后端 → ready
    assert aggregate_build_status(
        [_b("ready", "lightrag", chunks_total=0), _b("ready", "llamaindex_pg", chunks_total=80)]
    ) == "ready"


def test_kb_to_dict_dual_builds():
    """_kb_to_dict：聚合状态 + builds 数组 + 顶层代表 build 取 indexing 那个。"""
    from api.admin import _kb_to_dict

    kb = KnowledgeBase(id="k1", course_id="c1", name="t", index_backend="lightrag", file_count=2)
    kb.builds = [
        _b("ready", "lightrag", kb_id="k1", chunks_total=120, progress=100),
        _b("indexing", "llamaindex_pg", kb_id="k1", progress=42,
           progress_msg="embedding…", chunks_total=100, chunks_done=42),
    ]
    d = _kb_to_dict(kb)
    # 有在建 → 聚合 indexing
    assert d["status"] == "indexing"
    # 两后端都在 builds 数组，label 正确
    by_backend = {b["backend"]: b for b in d["builds"]}
    assert set(by_backend) == {"lightrag", "llamaindex_pg"}
    assert by_backend["llamaindex_pg"]["label"] == "pgvector"
    assert by_backend["lightrag"]["label"] == "LightRAG"
    # 顶层代表 build = indexing（pg），故顶层 progress/chunks 来自 pg
    assert d["progress"] == 42
    assert d["progress_msg"] == "embedding…"
    assert d["chunks_total"] == 100
    # index_backend 保留（默认后端，兼容旧前端）
    assert d["index_backend"] == "lightrag"


def test_kb_to_course_index_backends():
    """_kb_to_course：index_backends 只含 ready+有 chunks 的后端（学生端选择器可见性依据）。"""
    from api.courses import _kb_to_course

    # 只有 lightrag ready，pg 还 pending
    kb = KnowledgeBase(id="k2", course_id="c2", name="t")
    kb.builds = [
        _b("ready", "lightrag", kb_id="k2", chunks_total=120),
        _b("pending", "llamaindex_pg", kb_id="k2", chunks_total=0),
    ]
    c = _kb_to_course(kb)
    assert c["kb_status"] == "ready"
    assert c["rag_enabled"] is True
    assert c["index_backends"] == ["lightrag"]  # pg 未就绪不入选

    # 两后端都 ready → 都入选（排序稳定）
    kb.builds = [
        _b("ready", "llamaindex_pg", kb_id="k2", chunks_total=80),
        _b("ready", "lightrag", kb_id="k2", chunks_total=120),
    ]
    c2 = _kb_to_course(kb)
    assert c2["index_backends"] == ["lightrag", "llamaindex_pg"]
    assert c2["rag_enabled"] is True

    # 都没就绪 → rag_enabled=False，index_backends 空
    kb.builds = [_b("pending", "lightrag", kb_id="k2")]
    c3 = _kb_to_course(kb)
    assert c3["rag_enabled"] is False
    assert c3["index_backends"] == []


async def test_get_or_create_build_idempotent():
    """get_or_create_build：同 backend 幂等返回同一行；不同 backend 各一行。"""
    from core.db.database import AsyncSessionLocal, init_db, close_db

    await init_db()
    try:
        from api.kb_indexing import get_or_create_build

        async with AsyncSessionLocal() as db:
            async with db.begin():
                kb = KnowledgeBase(course_id=f"c_{__import__('os').urandom(3).hex()}", name="t")
                db.add(kb)
                await db.flush()
                kb_id = kb.id

                b1 = await get_or_create_build(db, kb_id, "lightrag")
                b1_id = b1.id
                # 再次 get-or-create 同 backend → 复用同一行（不新建）
                b2 = await get_or_create_build(db, kb_id, "lightrag")
                assert b2.id == b1_id
                # 不同 backend → 新行
                b3 = await get_or_create_build(db, kb_id, "llamaindex_pg")
                assert b3.id != b1_id
                assert b3.backend == "llamaindex_pg"
                assert b1.backend == "lightrag"
    finally:
        await close_db()


async def test_run_indexing_pg_empty_chunks_marks_error(monkeypatch):
    """0-chunk（解析全失败）的 pg 索引 → build 落 error 且 error_msg 含真实原因，不误判 ready。

    回归根因：MinerU 拒 200+ 页 PDF → parse_files 返 0 chunk → indexer 返 skipped →
    旧逻辑把 skipped 当 ready（绿徽章骗用户、按钮却禁用）。现 skipped/0-chunk 判 error。
    """
    from api.admin import _run_indexing_llamaindex_pg
    from api.kb_indexing import get_build, get_or_create_build
    from core.db.database import AsyncSessionLocal, KnowledgeBase, close_db, init_db
    from core.rag.types import IndexResult

    cid = f"c_empty_{__import__('os').urandom(3).hex()}"
    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                kb = KnowledgeBase(course_id=cid, name="t")
                db.add(kb)
                await db.flush()
                kb_id = kb.id
                build = await get_or_create_build(db, kb_id, "llamaindex_pg")
                build.status = "indexing"  # _apply_final 只在 indexing 时回写终态

        # mock get_indexer：index() 返 skipped + 0 chunk + 透传真实原因（来自 parse_errors）
        class _FakeIndexer:
            async def index(self, *args, **kwargs):
                return IndexResult(
                    course_id=cid,
                    files_indexed=1,
                    chunks_created=0,
                    status="skipped",
                    error="MinerU 解析失败: number of pages exceeds limit (200 pages)",
                )

            async def delete(self, course_id):
                return True

        monkeypatch.setattr("core.rag.get_indexer", lambda name: _FakeIndexer())

        await _run_indexing_llamaindex_pg(kb_id, cid, ["/data/x.pdf"], 0, "llamaindex_pg")

        async with AsyncSessionLocal() as db:
            b = await get_build(db, kb_id, "llamaindex_pg")
            assert b.status == "error"
            assert "exceeds limit" in b.error_msg  # 真实原因写进 error_msg
            assert (b.chunks_total or 0) == 0
    finally:
        await close_db()
