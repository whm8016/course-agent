"""LightRAG 实例池管理（LRU 缓存）。

从 lightrag_engine.py 提取的实例管理逻辑，负责：
- LRU 实例缓存（容量控制、淘汰）
- 实例初始化与销毁
- 索引锁管理（防止并发索引冲突）
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

from settings import get_settings
LIGHTRAG_LRU_CAPACITY = get_settings().lightrag_lru_capacity_scaled
LIGHTRAG_MAX_ASYNC = get_settings().lightrag.max_async
LIGHTRAG_WORKDIR = get_settings().paths.lightrag_workdir

logger = logging.getLogger(__name__)

# ── 全局状态（模块级，与原 lightrag_engine.py 保持一致）──────────────────────────────

_instances: OrderedDict[str, Any] = OrderedDict()
_instances_lock: asyncio.Lock | None = None  # 懒初始化，避免 import 时无 event loop
_index_locks: dict[str, asyncio.Lock] = {}

# 引用计数（H-10 use-after-evict 防护）：
# _get_instance 拿到实例时 +1，调用方用完调 _release_instance 递减。
# evict_oldest 跳过 in_use>0 的实例——并发多课程请求下，一个课程正在被某次
# 检索/索引使用时，绝不能因容量超限把它 evict 掉（否则拿到的是已 finalize 的死实例）。
_in_use: dict[str, int] = {}

# 索引签名缓存（用于判断是否需要重新索引）
_index_signatures: dict[str, tuple[str, ...]] = {}
_last_auto_index_at: dict[str, float] = {}
_AUTO_INDEX_STATE_DIR = Path(LIGHTRAG_WORKDIR) / ".auto_index_state"
_AUTO_INDEX_LOCK_DIR = Path(LIGHTRAG_WORKDIR) / ".auto_index_locks"


def _get_instances_lock() -> asyncio.Lock:
    """获取全局实例锁（懒初始化）。"""
    global _instances_lock
    if _instances_lock is None:
        _instances_lock = asyncio.Lock()
    return _instances_lock


def _workspace_name(course_id: str) -> str:
    """生成 LightRAG workspace 名称。"""
    return f"course_{course_id}"


def _ensure_lightrag_pg_env() -> None:
    """把 settings.db.url 桥接成 LightRAG PG 后端认的 POSTGRES_* 环境变量。

    LightRAG 的 PG 后端只读自己的 POSTGRES_* env（postgres_impl.ClientManager.get_config），
    不认项目的 SQLAlchemy DB__URL。必须在首次构造 LightRAG 前调用——ClientManager 的连接池
    是进程级单例，首次创建时读一次 env 后不再重读。setdefault 保证幂等，多次调用无害。

    两个硬约束：
    - 绝不设 POSTGRES_WORKSPACE：postgres_impl 的 workspace 优先级是
      PostgreSQLDB.workspace > self.workspace > "default"，一旦设了它会覆盖每门课的
      workspace（course_{id}），所有课程数据混进同一命名空间，多租户隔离直接失效。
    - POSTGRES_MAX_CONNECTIONS 显式压到 5：项目 SQLAlchemy 池已按 worker 严格缩放（4 worker≈20
      连接，见 database.py），但 LightRAG 另开独立 asyncpg 池不受该缩放；5/进程 × 4 worker=20，
      合计 40，远低于 max_connections(100)，留足余量。
    """
    from urllib.parse import urlparse

    url = get_settings().db.url.get_secret_value()
    # 剥掉 SQLAlchemy 的 +asyncpg driver 后缀，LightRAG PG 后端要裸 postgresql:// scheme
    p = urlparse(url.replace("postgresql+asyncpg://", "postgresql://", 1))
    os.environ.setdefault("POSTGRES_HOST", p.hostname or "localhost")
    os.environ.setdefault("POSTGRES_PORT", str(p.port or 5432))
    os.environ.setdefault("POSTGRES_USER", p.username or "postgres")
    os.environ.setdefault("POSTGRES_PASSWORD", p.password or "")
    os.environ.setdefault("POSTGRES_DATABASE", (p.path or "").lstrip("/"))
    os.environ.setdefault("POSTGRES_MAX_CONNECTIONS", "5")


# 部署形态（进程级单值，非每请求派生）。导入时算一次，避免 _get_instance（LRU miss 热路径）
# 与 purge 每次重复 get_settings + urlparse。SQLite 部署 _IS_POSTGRES=False，env 桥接不触发。
_IS_POSTGRES = get_settings().db.url.get_secret_value().startswith("postgres")
if _IS_POSTGRES:
    _ensure_lightrag_pg_env()


def get_instance_count() -> int:
    """返回当前实例数量。"""
    return len(_instances)


def get_instances() -> OrderedDict[str, Any]:
    """返回实例字典（仅供内部使用）。"""
    return _instances


async def evict_oldest() -> str | None:
    """淘汰最旧且当前未被引用的实例（H-10：跳过 in_use>0，避免 use-after-evict）。

    多课程并发检索/索引时，某个实例可能正被使用（_in_use>0）；若直接 popitem(last=False)
    淘汰它，调用方拿到的是已 finalize 的死实例。故遍历 OrderedDict 找第一个 in_use==0
    （或不在 _in_use）的实例淘汰；全部在用则返回 None——调用方据此临时超容，等下次再淘汰。

    Returns:
        被淘汰的 course_id；无实例可淘汰或全部在用时返回 None
    """
    if not _instances:
        return None

    evicted_id: str | None = None
    evicted_rag: Any = None
    # 在 OrderedDict 上找第一个未被引用的 key（保持 LRU 顺序）。不能边遍历边删，
    # 故先定位 key 再 pop。OrderedDict 的迭代顺序 = LRU 顺序（最旧在前）。
    for cid in _instances:
        if _in_use.get(cid, 0) <= 0:
            evicted_id = cid
            evicted_rag = _instances.pop(cid)
            break

    if evicted_id is None:
        # 全部在用：不淘汰任何实例，调用方临时超容
        logger.info(
            "LightRAG LRU evict 跳过：全部实例在用 slots=%d/%d",
            len(_instances), LIGHTRAG_LRU_CAPACITY,
        )
        return None

    # finalize 可能因存储后端抖动失败；用具体捕获 + 日志，不再 bare except 静默吞掉
    # （M-22：原 except Exception: pass 会掩盖真实错误，调试困难）。
    if hasattr(evicted_rag, "finalize_storages"):
        try:
            await evicted_rag.finalize_storages()
        except (OSError, RuntimeError, asyncio.CancelledError) as exc:
            logger.warning(
                "LightRAG finalize_storages 失败 course=%s: %s",
                evicted_id, exc, exc_info=True,
            )
        except Exception as exc:  # 兜底：第三方存储后端的非预期异常
            logger.warning(
                "LightRAG finalize_storages 未预期异常 course=%s: %s",
                evicted_id, exc, exc_info=True,
            )

    # 淘汰后清理引用计数残留（理论上 in_use==0 时 key 已被 _release_instance pop，
    # 这里防御性兜底，避免脏数据残留）。
    _in_use.pop(evicted_id, None)

    logger.info(
        "LightRAG LRU evict course=%s slots=%d/%d",
        evicted_id, len(_instances), LIGHTRAG_LRU_CAPACITY,
    )
    return evicted_id


async def _release_instance(course_id: str) -> None:
    """释放一个实例的引用（_get_instance 获取后的配对调用，H-10）。

    in_use 递减；归零则从 _in_use 移除 key（让该实例重新可被 evict）。不持有
    _instances_lock——_in_use 是普通 dict 的整型读写，单事件循环内原子性足够；
    多 worker 下每个 worker 各有独立 _in_use，不跨进程共享，无需加锁。
    """
    count = _in_use.get(course_id)
    if count is None:
        # 防御：没有获取记录却被释放（双释放/逻辑错误），记日志但不清空实例
        logger.warning("_release_instance 无对应获取记录 course=%s", course_id)
        return
    if count <= 1:
        _in_use.pop(course_id, None)
    else:
        _in_use[course_id] = count - 1


@contextlib.asynccontextmanager
async def lease_instance(course_id: str):
    """租借一个 LightRAG 实例（H-10 引用计数的安全使用方式）。

    _get_instance 获取时 in_use+1，离开 with 块（含异常）时 _release_instance 递减。
    所有使用 LightRAG 实例的调用方都应通过本上下文管理器，避免裸 _get_instance 后
    忘记释放导致引用计数泄漏（泄漏则实例永不被 evict，最终全部超容 OOM）。

    用法：
        async with lease_instance(course_id) as rag:
            await rag.aquery(...)
    """
    rag = await _get_instance(course_id)
    try:
        yield rag
    finally:
        await _release_instance(course_id)


async def _get_instance(course_id: str) -> Any:
    """获取 LightRAG 实例（LRU 缓存）。

    这是从原 lightrag_engine.py 提取的核心逻辑，内部自动引用
    llm_adapter.py 的 llm_func / embedding_func。

    Args:
        course_id: 课程 ID

    Returns:
        LightRAG 实例

    Raises:
        RuntimeError: LightRAG 不可用
    """
    # 延迟导入避免循环依赖
    from core.rag.lightrag.llm_adapter import (
        is_lightrag_available,
        _llm_model_func,
        _embedding_func,
        build_role_llm_configs,
    )

    # 导入 LightRAG（延迟导入避免循环依赖）
    try:
        from lightrag import LightRAG
    except Exception as exc:
        raise RuntimeError(f"LightRAG 依赖不可用: {exc}")

    ok, reason = is_lightrag_available()
    if not ok:
        raise RuntimeError(reason)

    lock = _get_instances_lock()
    async with lock:
        if course_id in _instances:
            _instances.move_to_end(course_id)  # LRU hit
            _in_use[course_id] = _in_use.get(course_id, 0) + 1  # H-10 引用计数 +1
            return _instances[course_id]

        # 淘汰最旧实例直到容量满足。evict_oldest 在全部实例在用时返回 None——
        # 此时不能死循环等（会卡住所有并发请求），只能临时超容（多挂一个实例），
        # 等某个实例被释放后再由后续淘汰回收。
        while len(_instances) >= LIGHTRAG_LRU_CAPACITY:
            evicted = await evict_oldest()
            if evicted is None:
                break  # 全部在用，临时超容

        Path(LIGHTRAG_WORKDIR).mkdir(parents=True, exist_ok=True)

        _extra_kwargs: dict[str, Any] = {}
        import inspect
        _sig = inspect.signature(LightRAG.__init__)
        if "llm_model_max_async" in _sig.parameters:
            _extra_kwargs["llm_model_max_async"] = LIGHTRAG_MAX_ASYNC

        # Rerank：DashScope gte-rerank-v2；有 EMBEDDING__API_KEY（或等价凭证）时挂载，否则跳过
        if "rerank_model_func" in _sig.parameters:
            from core.rag.lightrag.rerank_adapter import build_rerank_func
            _rerank_func = build_rerank_func()
            if _rerank_func is not None:
                _extra_kwargs["rerank_model_func"] = _rerank_func
                # 相关性阈值过滤（LightRAG 1.5.4 原生 min_rerank_score）：仅在 rerank 已挂载时
                # 才有意义——低于阈值的 chunk 在 rerank 后被丢弃。默认 0.0 不传（行为不变）；
                # 口径是 gte-rerank-v2 的 relevance_score，不可照搬裸余弦 0.5。
                if "min_rerank_score" in _sig.parameters:
                    _min_score = get_settings().lightrag.min_rerank_score
                    if _min_score > 0:
                        _extra_kwargs["min_rerank_score"] = _min_score

        # 分角色模型（role_llm_configs，LightRAG 1.5.4+）：extract/keyword 走便宜模型省成本。
        # build_role_llm_configs 在两者皆空时返回 None → 不传 → 全角色回退 base（行为不变）。
        _role_cfgs = None
        if "role_llm_configs" in _sig.parameters:
            _role_cfgs = build_role_llm_configs()
            if _role_cfgs is not None:
                _extra_kwargs["role_llm_configs"] = _role_cfgs

        # 存储后端：Postgres 部署把 KV/Vector/DocStatus 搬出进程内存（每门课常驻 RSS 从数百 MB
        # 降到数十 MB），图谱始终 NetworkX（文件，体积小）。env 桥接 + _IS_POSTGRES 判定已在
        # 模块导入时完成一次；SQLite 部署 _IS_POSTGRES=False 保持默认内存后端。
        if _IS_POSTGRES:
            _extra_kwargs["kv_storage"] = "PGKVStorage"
            _extra_kwargs["vector_storage"] = "PGVectorStorage"
            _extra_kwargs["doc_status_storage"] = "PGDocStatusStorage"
            # 显式写出 NetworkX：有意保留文件图谱后端，不切 PGGraphStorage（Apache AGE 建图
            # 官方 benchmark 434s、p95 1072ms，实测灾难；NetworkX 0.1s）。
            _extra_kwargs["graph_storage"] = "NetworkXStorage"

        rag = LightRAG(
            working_dir=LIGHTRAG_WORKDIR,
            workspace=_workspace_name(course_id),
            llm_model_func=_llm_model_func,
            embedding_func=_embedding_func,
            **_extra_kwargs,
        )
        await rag.initialize_storages()
        if _role_cfgs:
            logger.info(
                "LightRAG role_llm_configs 启用 course=%s roles=%s（其余角色回退 base）",
                course_id, list(_role_cfgs.keys()),
            )
        _instances[course_id] = rag
        _in_use[course_id] = 1  # H-10：新建实例即被本次获取引用

        logger.info(
            "LightRAG LRU load course=%s slots=%d/%d workspace=%s",
            course_id, len(_instances), LIGHTRAG_LRU_CAPACITY, _workspace_name(course_id),
        )
        return rag


# ── 索引锁管理─────────────────────────────────────────────────────


def get_index_lock(course_id: str) -> asyncio.Lock:
    """获取课程索引锁（防止并发索引冲突）。"""
    if course_id not in _index_locks:
        _index_locks[course_id] = asyncio.Lock()
    return _index_locks[course_id]


def clear_index_lock(course_id: str) -> None:
    """清除索引锁。"""
    if course_id in _index_locks:
        del _index_locks[course_id]


# ── 签名缓存（用于判断是否需要重新索引）───────────────────────────────────────


def _build_signature(file_paths: list[str]) -> tuple[str, ...]:
    """构建文件签名（用于判断是否需要重新索引）。"""
    signature: list[str] = []
    for file_path in sorted(file_paths):
        path = Path(file_path)
        stat = path.stat()
        signature.append(f"{file_path}|{stat.st_mtime_ns}|{stat.st_size}")
    return tuple(signature)


def get_cached_signature(course_id: str) -> tuple[str, ...] | None:
    """获取缓存的签名（内存层；持久层用 hydrate_signature 预先载入）。"""
    return _index_signatures.get(course_id)


def set_cached_signature(course_id: str, signature: tuple[str, ...]) -> None:
    """设置缓存的签名（内存层；持久化用 persist_signature 异步写 Redis）。"""
    _index_signatures[course_id] = signature


# M-33：签名缓存持久化（Redis）。原 _index_signatures 是模块级内存态，进程重启/多
# worker 间不共享，导致"文件未变则跳过重索引"的优化失效（每次重启都全量重索引）。
# P0 阶段删掉的 _SIG_PREFIX 即为此预留的半成品名，此处正经实现：内存层 + Redis 持久层
# 双写。get/set 保持同步读内存（零阻塞）；hydrate（启动/缺失时载入）、persist（写后
# 持久化）、invalidate（删索引时清理）为 async，由 indexer 显式调用。
#
# 并发一致性：Redis 单 key 读写各自原子；多 worker 同时判定"未变"都跳过是无害的
# （幂等），同时判定"变了"都重索引由 acquire_index_dlock 分布式锁互斥。签名本身的
# TOCTOU（读旧值→文件已变→误跳过）窗口极小且最坏后果是少跑一次索引（下次会补），可接受。
_SIG_PREFIX = "indexing:sig:"
_SIG_TTL = 30 * 24 * 3600  # 30 天：覆盖 KB 长期不动；过期则视为缓存 miss（重索引一次）


async def hydrate_signature(course_id: str) -> tuple[str, ...] | None:
    """从 Redis 载入签名到内存（缓存 miss 时调用）。Redis 不可用则降级返回 None。"""
    try:
        from core.db.cache import _get_pool
        redis = _get_pool()
    except Exception:
        logger.debug("signature hydrate: Redis 不可用，跳过持久化 course=%s", course_id)
        return None
    try:
        raw = await redis.get(f"{_SIG_PREFIX}{course_id}")
    except Exception:
        logger.debug("signature hydrate 读取失败 course=%s", course_id, exc_info=True)
        return None
    if not raw:
        return None
    try:
        import json
        data = json.loads(raw)
        sig = tuple(data) if isinstance(data, list) else None
        if sig is not None:
            _index_signatures[course_id] = sig
        return sig
    except (ValueError, TypeError):
        logger.warning("signature hydrate 反序列化失败 course=%s", course_id, exc_info=True)
        return None


async def persist_signature(course_id: str) -> None:
    """把内存中的签名持久化到 Redis（set_cached_signature 之后调用）。"""
    sig = _index_signatures.get(course_id)
    if sig is None:
        return
    try:
        from core.db.cache import _get_pool
        redis = _get_pool()
        import json
        await redis.set(
            f"{_SIG_PREFIX}{course_id}",
            json.dumps(list(sig)),
            ex=_SIG_TTL,
        )
    except Exception:
        # 持久化失败不阻断索引流程（内存层仍有效，仅本次重启后丢失）
        logger.debug("signature persist 失败 course=%s", course_id, exc_info=True)


async def invalidate_signature(course_id: str) -> None:
    """删除 Redis 与内存中的签名（删索引/清空 workspace 时调用）。"""
    _index_signatures.pop(course_id, None)
    try:
        from core.db.cache import _get_pool
        redis = _get_pool()
        await redis.delete(f"{_SIG_PREFIX}{course_id}")
    except Exception:
        logger.debug("signature invalidate 失败 course=%s", course_id, exc_info=True)


# ── 重新索引前清场 + 跨进程分布式锁 ──────────────────────────────────────────

# PG 搬迁的 11 个 storage 属性名（与 LightRAG.initialize_storages / finalize_storages 同款
# 列表对齐，仅去掉 graph：图谱始终 NetworkX 文件后端，由 rmtree 清 graphml）。LightRAG 1.5.4
# 不暴露可迭代注册表（上游也是方法内联硬编码），故镜像其列表；测试也 import 本常量保持一致。
_PG_STORAGE_ATTRS = (
    "full_docs", "text_chunks", "full_entities", "full_relations",
    "entity_chunks", "relation_chunks", "entities_vdb", "relationships_vdb",
    "chunks_vdb", "llm_response_cache", "doc_status",
)


async def _drop_workspace_storages(rag: Any, course_id: str) -> None:
    """对 rag 的 PG storage 逐个调 drop()，精准清本课程在 Postgres 的行。

    drop() 发的是 ``DELETE FROM <表> WHERE workspace=$1``——按 workspace 列隔离删行，
    **不 DROP 共享表**（PGVectorStorage.drop 文档明确），故只清本课程，不殃及其他课程。

    drop() 失败不阻断：返回 status!=success 或抛异常都只记日志，因为 purge 本就是清场，
    单个 storage 清不掉不该卡住后续 rmtree + 重索引。
    """
    for attr in _PG_STORAGE_ATTRS:
        storage = getattr(rag, attr, None)
        if storage is None:
            continue
        ns = getattr(storage, "namespace", attr)
        try:
            result = await storage.drop()
        except Exception:
            logger.warning(
                "purge: storage drop 异常 course=%s ns=%s", course_id, ns, exc_info=True,
            )
            continue
        if isinstance(result, dict) and result.get("status") != "success":
            logger.warning(
                "purge: storage drop 未成功 course=%s ns=%s result=%s", course_id, ns, result,
            )


async def purge_course_workspace(course_id: str) -> None:
    """清空某课程的 LightRAG 工作区（实例池缓存 + Postgres 数据 + 磁盘 graphml）。

    重新索引前调用，避免旧文档残留导致 ainsert 把整批判为 "Duplicate document"、
    最终堆在 failed entries。仅用于全新索引（resume_from_chunk==0）；续传绝不能
    调用，否则抹掉已索引进度。

    PG 搬迁后 KV/Vector/DocStatus 数据在 Postgres（按 workspace 列隔离），删它们必须通过
    一个活实例的 storage.drop()。实例无论是否在池里，PG 数据都在；池里没有就现拉一个临时
    实例专门来 drop，否则漏清会让重索引照样判重（purge 的全部意义就是治这个）。
    """
    ws_dir = Path(LIGHTRAG_WORKDIR) / _workspace_name(course_id)

    # 直接 pop 而非走 _get_instance：绕过 is_lightrag_available 检查与 _in_use 计数（purge
    # 不该被可用性瞬时失败挡住），也确保数据已清的旧实例不被复用。drop()/finalize() 是实例
    # 自身方法，不依赖是否在池中。
    async with _get_instances_lock():
        rag = _instances.pop(course_id, None)

    # 冷缓存：实例不在池里也得清 PG 数据——否则旧 chunk 残留，重索引判重。现拉一个临时实例。
    if rag is None:
        try:
            rag = await _get_instance(course_id)
        except Exception:
            logger.warning(
                "purge: 无法加载 LightRAG 实例，PG 数据可能残留 course=%s",
                course_id, exc_info=True,
            )
            rag = None

    if rag is not None:
        # 仅 PG 部署需要 drop 行；SQLite 数据全在文件（JSON/graphml），rmtree 即可清掉。
        if _IS_POSTGRES:
            await _drop_workspace_storages(rag, course_id)
        try:
            await rag.finalize_storages()
        except Exception:
            logger.warning("finalize_storages 失败 course=%s", course_id, exc_info=True)
        # 彻底移除并清引用计数：实例无论来自池（已 pop）还是临时拉起，purge 后都不应留存。
        async with _get_instances_lock():
            _instances.pop(course_id, None)
        _in_use.pop(course_id, None)

    if ws_dir.exists():
        shutil.rmtree(ws_dir, ignore_errors=True)
        logger.info("已清空 LightRAG 工作区 course=%s dir=%s", course_id, ws_dir)


# 分布式索引锁：跨 worker 进程互斥（asyncio.Lock 只在单进程内有效，多容器/多进程
# 部署下护不住）。TTL + 续约守护：索引可达数小时，靠 renew_task 反复 extend 防止
# 锁过期被别的 worker 抢走；持有者崩溃则 TTL 到期自动释放，不留死锁。
_INDEX_DLOCK_TTL = 3600  # 单次续约周期 1 小时
_INDEX_DLOCK_PREFIX = "indexing:dlock:"


async def acquire_index_dlock(course_id: str, backend: str = "lightrag"):
    """获取 (course, backend) 级 Redis 分布式锁。返回 (lock, renew_task)；被占返回 (None, None)。

    key 含 backend：同一课程 lightrag 与 pgvector 两套索引可并行构建（互不阻塞），但同一
    后端同一课程仍互斥（防 ainsert/delete 并发写同一份 store 产生重复/损坏）。
    """
    from core.db.cache import _get_pool
    redis = _get_pool()
    lock = redis.lock(
        f"{_INDEX_DLOCK_PREFIX}{course_id}:{backend}", timeout=_INDEX_DLOCK_TTL
    )
    if not await lock.acquire(blocking=False):
        return None, None

    async def _renew() -> None:
        try:
            while True:
                await asyncio.sleep(_INDEX_DLOCK_TTL / 3)
                try:
                    await lock.extend(_INDEX_DLOCK_TTL)
                except Exception:
                    logger.warning(
                        "索引锁续约失败 course=%s backend=%s", course_id, backend, exc_info=True
                    )
        except asyncio.CancelledError:
            pass

    renew = asyncio.create_task(_renew())
    return lock, renew


async def release_index_dlock(lock, renew) -> None:
    """释放分布式锁并取消续约守护。"""
    if renew is not None:
        renew.cancel()
        try:
            await renew
        except (asyncio.CancelledError, Exception):
            pass
    if lock is not None:
        try:
            await lock.release()
        except Exception:
            # 锁可能已过 TTL 被别人取走，release 找不到自己的 token，属正常
            logger.debug("索引锁释放失败（可能已过期）", exc_info=True)


__all__ = [
    "_get_instance",
    "_release_instance",
    "lease_instance",
    "_get_instances_lock",
    "_workspace_name",
    "get_instance_count",
    "get_instances",
    "evict_oldest",
    "get_index_lock",
    "clear_index_lock",
    "_build_signature",
    "get_cached_signature",
    "set_cached_signature",
    "hydrate_signature",
    "persist_signature",
    "invalidate_signature",
    "_instances",
    "_in_use",
    "_index_locks",
    "_index_signatures",
    "purge_course_workspace",
    "acquire_index_dlock",
    "release_index_dlock",
]