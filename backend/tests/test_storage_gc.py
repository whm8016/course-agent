"""磁盘派生数据 GC 回归测试（core.storage.gc）。

覆盖（plan §五 + 安全护栏硬要求）：
- sweep_by_age：删过期、保留新、dry_run 不真删。
- sweep_by_size_lru：按 mtime 升序删到上限（最旧先淘汰，Bazel 口径）。
- 护栏：1h mtime 宽限期跳过热数据、越界路径（is_relative_to）拒删、parse_cache 只收 ready 目录。
- collect_orphans：假 DB 判无主 lightrag_store 目录 / uploads 文件。
- run_gc：编排多 target + dry_run 报告；kb_store/raw 永不被删。

隔离：sweep 原语直接吃 tmp_path 根（不读 settings）；run_gc/collect_orphans 用
monkeypatch 把 settings.paths.* 指到 tmp_path。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from core.storage.gc import (
    _collect_files,
    _collect_parse_cache_sig_dirs,
    _MTIME_GRACE_SEC,
    _safe_to_delete,
    sweep_by_age,
    sweep_by_size_lru,
)

_DAY = 86400


def _set_mtime(path, days_ago: float) -> None:
    t = time.time() - days_ago * _DAY
    os.utime(path, (t, t))


# ---------------------------------------------------------------------------
# sweep_by_age
# ---------------------------------------------------------------------------

def test_sweep_by_age_deletes_old_keeps_new(tmp_path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    old.write_text("x" * 100)
    new.write_text("y" * 100)
    _set_mtime(old, 30)  # 30 天前
    _set_mtime(new, 0.01)  # 刚写

    stats = sweep_by_age(tmp_path, 7 * _DAY, dry_run=False, collect=_collect_files)
    assert stats.deleted == 1
    assert not old.exists()
    assert new.exists()
    assert stats.freed_bytes == 100


def test_sweep_by_age_disabled_when_zero(tmp_path):
    """max_age_sec <= 0 → 短路（对应 uploads_max_age_days=0 关闭按年龄清）。"""
    f = tmp_path / "a.json"
    f.write_text("x")
    _set_mtime(f, 30)
    stats = sweep_by_age(tmp_path, 0, dry_run=False, collect=_collect_files)
    assert stats.deleted == 0
    assert f.exists()


def test_sweep_by_age_dry_run_no_delete(tmp_path):
    old = tmp_path / "old.json"
    old.write_text("x" * 50)
    _set_mtime(old, 30)
    stats = sweep_by_age(tmp_path, 7 * _DAY, dry_run=True, collect=_collect_files)
    assert stats.deleted == 1
    assert stats.freed_bytes == 50
    assert old.exists()  # dry_run 不真删
    assert str(old) in stats.deleted_paths


# ---------------------------------------------------------------------------
# sweep_by_size_lru（mtime 升序，最旧先淘汰）
# ---------------------------------------------------------------------------

def test_sweep_by_size_lru_oldest_first(tmp_path):
    """3 个 60B 文件，cap 100 → 删最旧的 2 个，留最新的。"""
    files = []
    for i, name in enumerate(["a", "b", "c"]):  # a 最旧，c 最新
        f = tmp_path / f"{name}.json"
        f.write_text("x" * 60)
        _set_mtime(f, 10 - i)  # a=10天前, b=9, c=8
        files.append(f)
    stats = sweep_by_size_lru(tmp_path, 100, dry_run=False, collect=_collect_files)
    # total 180 → 删 a(→120>100) → 删 b(→60<=100 停)
    assert stats.deleted == 2
    assert not files[0].exists() and not files[1].exists()
    assert files[2].exists()


def test_sweep_by_size_lru_under_cap_no_delete(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("x" * 10)
    _set_mtime(f, 10)
    stats = sweep_by_size_lru(tmp_path, 1000, dry_run=False, collect=_collect_files)
    assert stats.deleted == 0
    assert f.exists()


def test_sweep_by_size_lru_disabled_when_zero(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("x" * 10)
    _set_mtime(f, 10)
    stats = sweep_by_size_lru(tmp_path, 0, dry_run=False, collect=_collect_files)
    assert stats.deleted == 0


# ---------------------------------------------------------------------------
# 护栏：mtime 宽限期 + 越界拒删
# ---------------------------------------------------------------------------

def test_grace_period_protects_recent(tmp_path):
    """超 cap 但 mtime 在 1h 宽限期内 → 跳过不删（防删到正在写的热文件）。"""
    recent = tmp_path / "recent.json"
    recent.write_text("x" * 200)
    # mtime = now（宽限期内）
    stats = sweep_by_size_lru(tmp_path, 50, dry_run=False, collect=_collect_files)
    assert stats.deleted == 0
    assert stats.skipped == 1
    assert recent.exists()


def test_safe_to_delete_rejects_out_of_root(tmp_path):
    """路径逃出 root（is_relative_to 失败）→ 拒删。"""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"  # 在 root 之外
    outside.write_text("x")
    _set_mtime(outside, 2)  # 超出宽限期
    now = time.time()
    assert _safe_to_delete(outside, root.resolve(), now=now, grace_sec=_MTIME_GRACE_SEC) is False
    assert outside.exists()


def test_safe_to_delete_grace_blocks_recent(tmp_path):
    root = tmp_path
    f = root / "hot.txt"
    f.write_text("x")
    # mtime = now（宽限期内）
    assert _safe_to_delete(f, root.resolve(), now=time.time(), grace_sec=_MTIME_GRACE_SEC) is False


# ---------------------------------------------------------------------------
# parse_cache 收集器：只收 ready（有 manifest）的签名目录
# ---------------------------------------------------------------------------

def test_collect_parse_cache_only_ready(tmp_path):
    from core.rag.parsing.cache import MANIFEST_FILENAME, signature_dir

    root = tmp_path / "parse_cache"
    ready = signature_dir(root, "abc1234567890123", "sig1")
    ready.mkdir(parents=True)
    (ready / MANIFEST_FILENAME).write_text("{}")
    (ready / "doc.md").write_text("hello" * 20)

    half = signature_dir(root, "def2345678901234", "sig2")
    half.mkdir(parents=True)
    (half / "doc.md").write_text("partial")  # 无 manifest = 半写

    entries = list(_collect_parse_cache_sig_dirs(root))
    paths = {e[0] for e in entries}
    assert ready in paths
    assert half not in paths
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# uploads 归属解析
# ---------------------------------------------------------------------------

def test_upload_owner_parsing():
    from core.storage.naming import parse_upload_owner_id

    assert parse_upload_owner_id("u1_abc.png") == "u1"
    assert parse_upload_owner_id("no-underscore") is None


# ---------------------------------------------------------------------------
# collect_orphans（假 DB）
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeDb:
    """按调用顺序返回 course_ids / user_ids（collect_orphans 先查 course 再查 user）。"""

    def __init__(self, course_rows, user_rows):
        self._results = [course_rows, user_rows]
        self._n = 0

    async def execute(self, stmt):
        rows = self._results[self._n]
        self._n += 1
        return _FakeResult(rows)


@pytest.mark.asyncio
async def test_collect_orphans(tmp_path, monkeypatch):
    import core.storage.gc as gc
    from settings import get_settings

    s = get_settings()
    lr = tmp_path / "lightrag_store"
    lr.mkdir()
    up = tmp_path / "uploads"
    up.mkdir()
    (lr / "course_live").mkdir()       # 仍归属
    (lr / "course_dead").mkdir()       # 孤儿（DB 无此 course）
    (lr / "random_dir").mkdir()        # 无 course_ 前缀，忽略
    (up / "liveuser_abc.png").write_text("x")   # 有归属（user_id 无下划线，对齐 _short_uuid）
    (up / "deaduser_def.pdf").write_text("yy")  # 孤儿
    monkeypatch.setattr(s.paths, "lightrag_workdir", str(lr))
    monkeypatch.setattr(s.paths, "upload_dir", str(up))
    # 隔离 Redis：本测试只验孤儿判定逻辑，dlock 行为由专门用例覆盖。
    async def _no_locked():
        return set()
    monkeypatch.setattr(gc, "_locked_course_ids", _no_locked)

    db = _FakeDb([("live",)], [("liveuser",)])  # course_id="live" → 目录 course_live
    orphans = await gc.collect_orphans(db)
    assert orphans["lightrag_store"] == [lr / "course_dead"]
    assert orphans["uploads"] == [up / "deaduser_def.pdf"]


@pytest.mark.asyncio
async def test_collect_orphans_skips_indexing_course(tmp_path, monkeypatch):
    """持 indexing:dlock 的孤儿课程目录 → 跳过（不回收）。"""
    import core.storage.gc as gc
    from settings import get_settings

    s = get_settings()
    lr = tmp_path / "lightrag_store"
    lr.mkdir()
    (lr / "course_dead").mkdir()
    monkeypatch.setattr(s.paths, "lightrag_workdir", str(lr))
    monkeypatch.setattr(s.paths, "upload_dir", str(tmp_path / "uploads"))

    async def _locked_dead():
        return {"dead"}  # 假装 course_dead 正在索引

    monkeypatch.setattr(gc, "_locked_course_ids", _locked_dead)
    db = _FakeDb([], [])  # DB 无任何 course → 全是孤儿候选
    orphans = await gc.collect_orphans(db)
    assert orphans["lightrag_store"] == []  # 全在「索引中」，跳过


# ---------------------------------------------------------------------------
# run_gc：编排 + dry_run + kb_store 永不删
# ---------------------------------------------------------------------------

def _point_paths_at(tmp_path, monkeypatch):
    from settings import get_settings

    s = get_settings()
    for attr in ("parse_cache_dir", "lightrag_workdir", "upload_dir", "kb_store_dir", "ingest_chunks_dir"):
        d = tmp_path / attr
        d.mkdir(exist_ok=True)
        monkeypatch.setattr(s.paths, attr, str(d))
    return s


@pytest.mark.asyncio
async def test_run_gc_dry_run_deletes_nothing(tmp_path, monkeypatch):
    import core.storage.gc as gc
    from settings import get_settings

    _point_paths_at(tmp_path, monkeypatch)
    # 种一个过期的 ingest_chunks JSON（7 天阈值，给 30 天）。目录必须落在 patched 的
    # lightrag_workdir 下（不是字面量 lightrag_store）。
    course = Path(get_settings().paths.lightrag_workdir) / "course_x"
    ingest = course / "ingest_chunks"
    ingest.mkdir(parents=True)
    old = ingest / "latest.json"
    old.write_text("x" * 100)
    _set_mtime(old, 30)

    report = await gc.run_gc(dry_run=True)
    assert report["dry_run"] is True
    assert old.exists()  # dry_run 不删
    assert "parse_cache" in report["targets"]
    assert "total_freed_gib" in report


@pytest.mark.asyncio
async def test_run_gc_actually_deletes_old_ingest_chunks(tmp_path, monkeypatch):
    import core.storage.gc as gc
    from settings import get_settings

    _point_paths_at(tmp_path, monkeypatch)
    # 隔离 Redis + DB：本测试只验 ingest_chunks age 清理编排，dlock/孤儿由专门用例覆盖。
    async def _no_locked():
        return set()
    monkeypatch.setattr(gc, "_locked_course_ids", _no_locked)

    course = Path(get_settings().paths.lightrag_workdir) / "course_x"
    ingest = course / "ingest_chunks"
    ingest.mkdir(parents=True)
    old = ingest / "latest.json"
    old.write_text("x" * 100)
    fresh = ingest / "chunks_1.json"
    fresh.write_text("y" * 50)
    _set_mtime(old, 30)
    _set_mtime(fresh, 0.01)

    report = await gc.run_gc(dry_run=False)
    assert not old.exists()       # 过期被删
    assert fresh.exists()         # 新文件保留
    assert report["targets"]["lightrag_store"]["ingest_chunks_age_deleted"] >= 1


@pytest.mark.asyncio
async def test_run_gc_deletes_old_pg_ingest_chunks(tmp_path, monkeypatch):
    """pg 审计 JSON（data/ingest_chunks 独立根）：超期清、新文件留、报告含 pg_ingest_chunks 段。"""
    import core.storage.gc as gc
    from settings import get_settings

    _point_paths_at(tmp_path, monkeypatch)
    # 隔离 Redis + DB：本测试只验 pg ingest_chunks age 清理编排。
    async def _no_locked():
        return set()
    monkeypatch.setattr(gc, "_locked_course_ids", _no_locked)

    # pg 布局：审计 JSON 直接落在 course_dir 下（无 ingest_chunks 子目录）
    course = Path(get_settings().paths.ingest_chunks_dir) / "course_x"
    course.mkdir(parents=True)
    old = course / "latest.json"
    old.write_text("x" * 100)
    fresh = course / "snapshot.json"
    fresh.write_text("y" * 50)
    _set_mtime(old, 30)
    _set_mtime(fresh, 0.01)

    report = await gc.run_gc(dry_run=False)
    assert not old.exists()
    assert fresh.exists()
    assert "pg_ingest_chunks" in report["targets"]
    assert report["targets"]["pg_ingest_chunks"]["deleted"] >= 1


@pytest.mark.asyncio
async def test_run_gc_never_touches_kb_store_raw(tmp_path, monkeypatch):
    """kb_store/raw 永不被 GC 删除（用户决定只监控不清理）。"""
    import core.storage.gc as gc

    _point_paths_at(tmp_path, monkeypatch)
    raw = tmp_path / "kb_store" / "course_x" / "raw"
    raw.mkdir(parents=True)
    teacher_pdf = raw / "teacher.pdf"
    teacher_pdf.write_text("x" * 5000)
    _set_mtime(teacher_pdf, 999)  # 极旧也不该被删

    report = await gc.run_gc(dry_run=False)
    assert teacher_pdf.exists()
    assert report["targets"]["kb_store"]["cleaned"] is False


def test_dir_size(tmp_path):
    from core.storage.gc import dir_size as _dir_size

    assert _dir_size(tmp_path / "missing") == 0
    (tmp_path / "a.bin").write_text("x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_text("y" * 50)
    assert _dir_size(tmp_path) == 150
