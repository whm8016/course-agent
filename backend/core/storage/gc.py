"""磁盘派生数据 GC（offline cron + admin 触发）。

对标 Bazel 磁盘缓存 GC（https://bazel.build/remote/caching#disk-cache）：Bazel 在
issue #5139 论证了「写时强制不超限」跨平台 + 多进程共享 + 不损性能三者难兼得，改为
服务空闲期后台 GC + max_size/max_age 双口径 + mtime 当 recency + 手动触发工具。
本仓库多 worker 共享同一卷、不能让索引变慢，约束同构，照搬同一形态。

纯函数 + 一个编排入口 ``run_gc``，不依赖 ARQ，便于单测。安全护栏（GC 会删文件，硬要求）：

- 所有待删路径 ``resolve()`` 后必须 ``is_relative_to`` 配置根，否则跳过并告警（防越界）。
- 永不进入 ``kb_store/*/raw``（教师原始文件，删掉无法重索引——用户明确决定只监控不清理）。
- parse_cache 只删 ``is_ready()``（有 manifest）的签名目录整体；半写目录交回现有 cleanup_failed。
- 1 小时 mtime 宽限：跳过最近写入的条目；删除前**再查一次 mtime**（Bazel 53839d3 同款防竞态）。
- 跳过持有 ``indexing:dlock:{course_id}:*`` Redis 锁的课程目录，避免和进行中的索引抢文件。
- admin 端点与 cron 均 best-effort：单条删除失败只记日志，不阻断其余清理。
"""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from settings import get_settings

logger = logging.getLogger(__name__)

# mtime 宽限期：最近 1 小时写入/触碰的条目不删（防与正在进行的解析/索引抢文件）。
_MTIME_GRACE_SEC = 3600
_GIB = 1024 ** 3  # 配置里的 *_max_gb 按 GiB 解释（du/df 常显示 GiB）

# sweep 单元：(路径, 字节数, mtime)。collect 决定粒度（文件级 or 目录级）。
Entry = tuple[Path, int, float]


@dataclass
class GcStats:
    """单次 sweep 的统计。deleted_paths 记相对/绝对路径字符串供 dry_run 报告核对。"""

    scanned: int = 0
    deleted: int = 0
    freed_bytes: int = 0
    skipped: int = 0
    deleted_paths: list[str] = field(default_factory=list)

    def merge(self, other: "GcStats") -> "GcStats":
        self.scanned += other.scanned
        self.deleted += other.deleted
        self.freed_bytes += other.freed_bytes
        self.skipped += other.skipped
        self.deleted_paths.extend(other.deleted_paths)
        return self

    @property
    def freed_gib(self) -> float:
        return self.freed_bytes / _GIB


# ---------------------------------------------------------------------------
# 体积
# ---------------------------------------------------------------------------

def dir_size(root: Path) -> int:
    """root 下所有常规文件字节数（递归）。不存在返回 0。"""
    if not root.exists():
        return 0
    total = 0
    try:
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


# ---------------------------------------------------------------------------
# entry 收集器（决定 sweep 粒度）
# ---------------------------------------------------------------------------

def _collect_files(root: Path) -> Iterable[Entry]:
    """文件级 entry：递归枚举 root 下每个常规文件 (path, size, mtime)。"""
    if not root.is_dir():
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        yield p, st.st_size, st.st_mtime


def _collect_parse_cache_sig_dirs(root: Path) -> Iterable[Entry]:
    """parse_cache 专用：签名目录级 entry（整体删，不破坏半写目录）。

    布局 ``<shard>/<source_hash>/<signature>/``（见 core.rag.parsing.cache.signature_dir）。
    仅收 ``is_ready()``（有 manifest）的目录；size=目录内文件总字节，mtime=目录 mtime
    （cache.lookup 命中时 os.utime 刷新它，故 mtime 即「最近访问」，作 LRU recency）。
    """
    from core.rag.parsing.cache import is_ready

    if not root.is_dir():
        return
    for shard in root.iterdir():  # <hash[:2]>
        if not shard.is_dir():
            continue
        for src in shard.iterdir():  # <source_hash>
            if not src.is_dir():
                continue
            for sig in src.iterdir():  # <signature>
                if not sig.is_dir() or not is_ready(sig):
                    continue
                try:
                    mt = sig.stat().st_mtime
                except OSError:
                    continue
                size = 0
                for f in sig.rglob("*"):
                    if f.is_file():
                        try:
                            size += f.stat().st_size
                        except OSError:
                            pass
                yield sig, size, mt


# ---------------------------------------------------------------------------
# 删除原语 + 安全护栏
# ---------------------------------------------------------------------------

def _safe_to_delete(path: Path, root_resolved: Path, *, now: float, grace_sec: float) -> bool:
    """删前复查：resolve 后仍在 root 内 + mtime 超出宽限期（防删到正在写的热文件）。

    ``root_resolved`` 由调用方在 sweep 入口 resolve 一次传入（root 在整个 sweep 内不变，
    避免每个候选条目重复 resolve 的 syscall）。返回 False 的两种情况：路径逃出 root
    （越界，拒删）、mtime 在宽限期内（可能正在用）。stat 失败（文件刚被别人删）也返回 False。
    """
    try:
        if not path.resolve().is_relative_to(root_resolved):
            logger.warning("GC 越界拒删（不在根下）：%s (root=%s)", path, root_resolved)
            return False
        if (now - path.stat().st_mtime) < grace_sec:
            return False
    except OSError:
        return False
    return True


def _delete(path: Path, size: int, stats: GcStats, *, dry_run: bool) -> None:
    """删除一个文件或目录，更新 stats。dry_run 只记账不删。失败只 skipped。"""
    if dry_run:
        stats.deleted += 1
        stats.freed_bytes += max(0, size)
        stats.deleted_paths.append(str(path))
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink()
        stats.deleted += 1
        stats.freed_bytes += max(0, size)
        stats.deleted_paths.append(str(path))
    except OSError as exc:
        logger.warning("GC 删除失败 %s: %s", path, exc)
        stats.skipped += 1


def _try_delete(
    path: Path, size: int, root_resolved: Path, stats: GcStats, *, now: float, dry_run: bool,
) -> bool:
    """护栏 + 删除一体（``_safe_to_delete`` + ``_delete``），统一各 sweep 的调用口径。

    通过护栏 → 删除（返回 True，size 可从预算里扣除）；越界 / 宽限期内 → skipped+1（False）。
    ``_delete`` 自身的 OSError 失败也已内部 skipped+1。所有 sweep/孤儿删除复用本函数。
    """
    if not _safe_to_delete(path, root_resolved, now=now, grace_sec=_MTIME_GRACE_SEC):
        stats.skipped += 1
        return False
    _delete(path, size, stats, dry_run=dry_run)
    return True


def sweep_by_age(
    root: Path,
    max_age_sec: int,
    *,
    dry_run: bool,
    collect: Callable[[Path], Iterable[Entry]] = _collect_files,
    entries: Iterable[Entry] | None = None,
) -> GcStats:
    """删除 mtime 早于 (now - max_age_sec) 的 entry。

    ``max_age_sec <= 0`` → 短路返回空 stats（对应配置里「0=关闭按年龄清」）。
    ``entries`` 非空时直接用（省一次 collect，供 run_gc 把同一批 entries 喂给 age+size 两扫）。
    """
    stats = GcStats()
    if max_age_sec <= 0:
        return stats
    now = time.time()
    root_resolved = root.resolve()
    cutoff = now - max_age_sec
    source = entries if entries is not None else collect(root)
    for path, size, mtime in source:
        stats.scanned += 1
        if mtime > cutoff:
            continue
        _try_delete(path, size, root_resolved, stats, now=now, dry_run=dry_run)
    return stats


def sweep_by_size_lru(
    root: Path,
    max_bytes: int,
    *,
    dry_run: bool,
    collect: Callable[[Path], Iterable[Entry]] = _collect_files,
    entries: Iterable[Entry] | None = None,
) -> GcStats:
    """按 mtime 升序删 entry 直到累计体积 <= max_bytes（Bazel 口径：最旧先淘汰）。

    ``max_bytes <= 0`` → 短路。先 sum 总量，**仅在超限时才排序**（common case 低于上限免 O(n log n)）。
    宽限期内（最近写入）的 entry 跳过，删其余最旧的；全部在宽限期则一个都不删。
    """
    stats = GcStats()
    if max_bytes <= 0:
        return stats
    now = time.time()
    root_resolved = root.resolve()
    items = list(entries if entries is not None else collect(root))
    stats.scanned = len(items)
    total = sum(e[1] for e in items)
    if total <= max_bytes:
        return stats
    items.sort(key=lambda e: e[2])  # mtime 升序（仅超限时才排）
    for path, size, _mtime in items:
        if total <= max_bytes:
            break
        if _try_delete(path, size, root_resolved, stats, now=now, dry_run=dry_run):
            total -= size
    return stats


# ---------------------------------------------------------------------------
# 索引锁守卫（避免与进行中的索引抢文件）
# ---------------------------------------------------------------------------

async def _locked_course_ids() -> set[str]:
    """当前持有 ``indexing:dlock`` 的 course_id 集合（一次 SCAN，避免每课程一次 SCAN）。

    key 形如 ``indexing:dlock:{course_id}:{backend}``，前缀取 ``instance_pool._INDEX_DLOCK_PREFIX``
    （单一事实源——锁所有者定义键格式，GC 不再硬编码）。Redis 不可用 → 空集（按未锁定
    处理；lightrag_store 侧另有 1h mtime 宽限兜底，误删刚写的 ingest_chunks 风险可控）。
    """
    ids: set[str] = set()
    try:
        from core.db.cache import _get_pool
        from core.rag.lightrag.instance_pool import _INDEX_DLOCK_PREFIX

        redis = _get_pool()
        prefix = _INDEX_DLOCK_PREFIX  # "indexing:dlock:"
        async for key in redis.scan_iter(f"{prefix}*"):
            rest = key[len(prefix):] if key.startswith(prefix) else key
            cid = rest.split(":", 1)[0]  # {course_id}:{backend} → course_id
            if cid:
                ids.add(cid)
    except Exception:
        logger.debug("dlock 集合查询失败，按空集处理", exc_info=True)
    return ids


# ---------------------------------------------------------------------------
# 孤儿回收（无 DB 归属）
# ---------------------------------------------------------------------------

async def collect_orphans(db) -> dict[str, list[Path]]:
    """返回无 DB 归属、可安全回收的路径（不删除，由 run_gc 统一删）。

    - ``lightrag_store/course_{cid}/``：cid 不在 KnowledgeBase.course_id 集合 → 孤儿；
      持有 indexing:dlock 的跳过（索引进行中）。
    - ``uploads/{user_id}_{uuid}.ext``：user_id 不在 User.id 集合 → 孤儿文件。

    kb_store 不在此列（用户决定只监控不清理 raw）。
    """
    from sqlalchemy import select

    from core.db.database import KnowledgeBase, User
    from core.storage.naming import parse_upload_owner_id

    course_ids = {row[0] for row in (await db.execute(select(KnowledgeBase.course_id))).all()}
    user_ids = {row[0] for row in (await db.execute(select(User.id))).all()}
    locked_ids = await _locked_course_ids()  # 一次 SCAN，下面按集合判定

    s = get_settings()
    lr_root = Path(s.paths.lightrag_workdir)
    lr_orphans: list[Path] = []
    if lr_root.is_dir():
        for d in lr_root.iterdir():
            if not d.is_dir() or not d.name.startswith("course_"):
                continue
            cid = d.name[len("course_"):]
            if cid in course_ids:
                continue  # 仍归属某 KB
            if cid in locked_ids:
                logger.info("孤儿回收跳过（索引进行中）course=%s", cid)
                continue
            lr_orphans.append(d)

    up_root = Path(s.paths.upload_dir)
    up_orphans: list[Path] = []
    if up_root.is_dir():
        for f in up_root.iterdir():
            if not f.is_file():
                continue
            owner = parse_upload_owner_id(f.name)
            if owner is None or owner in user_ids:
                continue  # 有归属，或无法判定（保守不动）
            up_orphans.append(f)

    return {"lightrag_store": lr_orphans, "uploads": up_orphans}


# ---------------------------------------------------------------------------
# lightrag_store 专项：只清 ingest_chunks（审计 JSON），graphml 永不删
# ---------------------------------------------------------------------------

async def _sweep_ingest_chunks(
    root: Path, *, subdir: str | None, max_age_sec: int, max_bytes: int, dry_run: bool,
) -> tuple[GcStats, int]:
    """清扫所有课程的 ingest_chunks JSON（age + size LRU），返回 (stats, locked_count)。

    subdir 非 None 时收 ``course_dir / subdir``（LightRAG 布局：graphml 与 ingest_chunks
    同级，只清子目录，故 graphml 永不被删）；subdir=None 时直接收 ``course_dir`` 下文件
    （pg 布局：``data/ingest_chunks/course_{id}/`` 整目录即审计 JSON）。
    持有 indexing:dlock 的课程整目录跳过（一次 SCAN 取 locked 集合，按集合判定）。
    size 口径作用于本批 entry 全体（非整卷）——整卷 graphml 超额无法安全裁剪，由 run_gc 打 warning。

    subdir 为必填 keyword（不给默认值）：强制调用方显式传，避免把 lightrag 误按扁平布局清扫
    （那会连 graphml 一起收进 candidates）。pg 调用方传 None，lightrag 调用方传子目录名。
    """
    stats = GcStats()
    if not root.is_dir():
        return stats, 0
    now = time.time()
    root_resolved = root.resolve()
    locked_ids = await _locked_course_ids()

    entries: list[Entry] = []
    locked = 0
    for course_dir in sorted(root.iterdir()):
        if not course_dir.is_dir() or not course_dir.name.startswith("course_"):
            continue
        cid = course_dir.name[len("course_"):]
        ingest_dir = course_dir / subdir if subdir else course_dir
        if not ingest_dir.is_dir():
            continue
        if cid in locked_ids:
            locked += 1
            continue
        entries.extend(_collect_files(ingest_dir))  # 复用通用文件收集器
    stats.scanned = len(entries)

    # 1) age：删过期 JSON
    deleted_ids: set[int] = set()
    if max_age_sec > 0:
        cutoff = now - max_age_sec
        for i, (path, size, mtime) in enumerate(entries):
            if mtime > cutoff:
                continue
            if _try_delete(path, size, root_resolved, stats, now=now, dry_run=dry_run):
                deleted_ids.add(i)

    # 2) size LRU：幸存者按 mtime 升序删到上限
    if max_bytes > 0:
        survivors = sorted(
            ((i, e) for i, e in enumerate(entries) if i not in deleted_ids),
            key=lambda ie: ie[1][2],
        )
        total = sum(e[1] for _, e in survivors)
        for _i, (path, size, _mtime) in survivors:
            if total <= max_bytes:
                break
            if _try_delete(path, size, root_resolved, stats, now=now, dry_run=dry_run):
                total -= size
    return stats, locked


# ---------------------------------------------------------------------------
# 存储用量（admin GET 端点）
# ---------------------------------------------------------------------------

def storage_usage() -> dict:
    """逐项目录体积 + 整卷水位（供 GET /api/admin/storage/usage）。"""
    s = get_settings()
    targets = {
        "parse_cache": dir_size(Path(s.paths.parse_cache_dir)),
        "lightrag_store": dir_size(Path(s.paths.lightrag_workdir)),
        "ingest_chunks": dir_size(Path(s.paths.ingest_chunks_dir)),
        "uploads": dir_size(Path(s.paths.upload_dir)),
        "kb_store": dir_size(Path(s.paths.kb_store_dir)),
    }
    try:
        du = shutil.disk_usage(str(Path(s.paths.lightrag_workdir).resolve()))
        disk = {"total": du.total, "used": du.used, "free": du.free}
    except OSError as exc:
        logger.warning("disk_usage 采样失败：%s", exc)
        disk = {"total": 0, "used": 0, "free": 0}
    used_pct = round(100 * disk["used"] / disk["total"], 1) if disk["total"] else 0.0
    return {
        "targets_bytes": targets,
        "disk": {**disk, "used_pct": used_pct},
        "disk_warn_pct": s.storage_gc.disk_warn_pct,
    }


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------

async def run_gc(*, dry_run: bool = True) -> dict:
    """编排全部 target 的 GC，返回逐项报告。

    顺序：parse_cache(age+size) → lightrag_store ingest_chunks(age+size) + 孤儿课程目录
    → pg ingest_chunks(data/ingest_chunks, age+size) → uploads(孤儿 + age + size)
    → kb_store(只统计)。每步独立 best-effort。

    默认 ``dry_run=True``（只报不删）；admin 端点放开前先核对报告。
    """
    settings = get_settings()
    cfg = settings.storage_gc
    report: dict = {"dry_run": dry_run, "enabled": cfg.enabled, "targets": {}}
    if not cfg.enabled:
        report["skipped"] = "storage_gc disabled (STORAGE_GC__ENABLED=false)"
        return report

    now = time.time()
    total_freed = 0

    def _gb_to_bytes(gb: float) -> int:
        return int(gb * _GIB)

    # ── parse_cache（内容寻址，非课程维度，无 dlock）──
    pc_root = Path(settings.paths.parse_cache_dir)
    pc_size_before = dir_size(pc_root)
    # 收一次喂给 age+size 两扫，避免对签名目录树遍历两遍（每 sig 还要 rglob 求和）。
    pc_entries = list(_collect_parse_cache_sig_dirs(pc_root))
    pc_age = sweep_by_age(
        pc_root, cfg.parse_cache_max_age_days * 86400,
        dry_run=dry_run, entries=pc_entries,
    )
    pc_size = sweep_by_size_lru(
        pc_root, _gb_to_bytes(cfg.parse_cache_max_gb),
        dry_run=dry_run, entries=pc_entries,
    )
    total_freed += pc_age.freed_bytes + pc_size.freed_bytes
    report["targets"]["parse_cache"] = {
        "size_bytes_before": pc_size_before,
        "size_bytes_after": dir_size(pc_root),
        "max_gb": cfg.parse_cache_max_gb, "max_age_days": cfg.parse_cache_max_age_days,
        "age_deleted": pc_age.deleted, "size_deleted": pc_size.deleted,
        "freed_bytes": pc_age.freed_bytes + pc_size.freed_bytes,
        "skipped": pc_age.skipped + pc_size.skipped,
    }

    # ── lightrag_store ingest_chunks + 孤儿课程目录 ──
    lr_root = Path(settings.paths.lightrag_workdir)
    lr_root_resolved = lr_root.resolve()
    lr_size_before = dir_size(lr_root)
    ingest_stats, locked = await _sweep_ingest_chunks(
        lr_root,
        subdir=settings.lightrag.ingest_chunks_subdir,
        max_age_sec=cfg.ingest_chunks_max_age_days * 86400,
        max_bytes=_gb_to_bytes(cfg.lightrag_store_max_gb),
        dry_run=dry_run,
    )
    total_freed += ingest_stats.freed_bytes

    # 孤儿课程目录（课程已从 DB 删除）。collect_orphans 需 DB。
    orphan_stats = GcStats()
    try:
        from core.db.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            orphans = await collect_orphans(db) if cfg.orphan_sweep_enabled else {
                "lightrag_store": [], "uploads": [],
            }
    except Exception:
        logger.warning("collect_orphans 失败，跳过孤儿回收", exc_info=True)
        orphans = {"lightrag_store": [], "uploads": []}

    for d in orphans["lightrag_store"]:
        size = dir_size(d)
        _try_delete(d, size, lr_root_resolved, orphan_stats, now=now, dry_run=dry_run)
    total_freed += orphan_stats.freed_bytes
    report["targets"]["lightrag_store"] = {
        "size_bytes_before": lr_size_before,
        "size_bytes_after": dir_size(lr_root),
        "max_gb": cfg.lightrag_store_max_gb,
        "ingest_chunks_age_deleted": ingest_stats.deleted,
        "ingest_chunks_freed_bytes": ingest_stats.freed_bytes,
        "orphan_dirs_deleted": orphan_stats.deleted,
        "orphan_freed_bytes": orphan_stats.freed_bytes,
        "indexing_locked_courses": locked,
        "graphml_note": "graphml 永不删；若 size_bytes_after 远超 max_gb，需排查大图课程",
    }

    # ── pg ingest_chunks 审计 JSON（独立根 data/ingest_chunks，无 graphml 故无孤儿回收）──
    # 与 lightrag_store/ingest_chunks 同源审计 JSON，仅布局不同（扁平 course_dir vs 子目录），
    # 故复用 _sweep_ingest_chunks（subdir=None）。课程删除后此目录最多残留 0 字节空目录，
    # 不像 lightrag_store 里有永不按 age 删的 graphml 需孤儿回收。
    pg_ing_root = Path(settings.paths.ingest_chunks_dir)
    pg_size_before = dir_size(pg_ing_root)
    pg_stats, pg_locked = await _sweep_ingest_chunks(
        pg_ing_root,
        subdir=None,
        max_age_sec=cfg.ingest_chunks_max_age_days * 86400,
        max_bytes=_gb_to_bytes(cfg.pg_ingest_chunks_max_gb),
        dry_run=dry_run,
    )
    total_freed += pg_stats.freed_bytes
    report["targets"]["pg_ingest_chunks"] = {
        "size_bytes_before": pg_size_before,
        "size_bytes_after": dir_size(pg_ing_root),
        "max_gb": cfg.pg_ingest_chunks_max_gb,
        "max_age_days": cfg.ingest_chunks_max_age_days,
        "deleted": pg_stats.deleted,
        "freed_bytes": pg_stats.freed_bytes,
        "skipped": pg_stats.skipped,
        "indexing_locked_courses": pg_locked,
    }

    # ── uploads（孤儿 + age + size）──
    up_root = Path(settings.paths.upload_dir)
    up_size_before = dir_size(up_root)
    up_orphan_stats = GcStats()
    up_root_resolved = up_root.resolve()
    for f in orphans["uploads"]:
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        _try_delete(f, size, up_root_resolved, up_orphan_stats, now=now, dry_run=dry_run)
    up_age = sweep_by_age(up_root, cfg.uploads_max_age_days * 86400, dry_run=dry_run)
    up_size = sweep_by_size_lru(up_root, _gb_to_bytes(cfg.uploads_max_gb), dry_run=dry_run)
    total_freed += up_orphan_stats.freed_bytes + up_age.freed_bytes + up_size.freed_bytes
    report["targets"]["uploads"] = {
        "size_bytes_before": up_size_before,
        "size_bytes_after": dir_size(up_root),
        "max_gb": cfg.uploads_max_gb, "max_age_days": cfg.uploads_max_age_days,
        "orphan_deleted": up_orphan_stats.deleted,
        "age_deleted": up_age.deleted, "size_deleted": up_size.deleted,
        "freed_bytes": up_orphan_stats.freed_bytes + up_age.freed_bytes + up_size.freed_bytes,
    }

    # ── kb_store（只统计不清理）──
    kb_root = Path(settings.paths.kb_store_dir)
    report["targets"]["kb_store"] = {
        "size_bytes": dir_size(kb_root),
        "cleaned": False,
        "note": "raw 只监控不清理（删掉无法重索引）",
    }

    report["total_freed_bytes"] = total_freed
    report["total_freed_gib"] = round(total_freed / _GIB, 3)

    # 整卷水位告警：inline shutil.disk_usage（廉价 statvfs），不调 storage_usage()——
    # 后者会再把 4 棵派生数据树 rglob 一遍只为读个百分比，GC 刚扫完是纯浪费。
    try:
        du = shutil.disk_usage(str(Path(settings.paths.lightrag_workdir).resolve()))
        used_pct = round(100 * du.used / du.total, 1) if du.total else 0.0
    except OSError:
        used_pct = 0.0
    report["disk_used_pct"] = used_pct
    if used_pct >= cfg.disk_warn_pct:
        logger.warning(
            "磁盘水位告警：整卷使用率 %.1f%% >= %d%%（阈值 STORAGE_GC__DISK_WARN_PCT）",
            used_pct, cfg.disk_warn_pct,
        )

    # 指标（best-effort）
    try:
        from core.observability.metrics import STORAGE_GC_DELETED_BYTES

        STORAGE_GC_DELETED_BYTES.inc(total_freed)
    except Exception:
        logger.debug("STORAGE_GC_DELETED_BYTES 上报失败", exc_info=True)

    logger.info(
        "storage GC 完成 dry_run=%s freed_gib=%.3f disk_used_pct=%.1f",
        dry_run, report["total_freed_gib"], used_pct,
    )
    return report


__all__ = [
    "GcStats",
    "dir_size",
    "sweep_by_age",
    "sweep_by_size_lru",
    "collect_orphans",
    "storage_usage",
    "run_gc",
]
