"""后端无关的 Cross-Encoder 精排工厂：httpx 直连 DashScope qwen3-rerank。

PG 检索链路（llamaindex_pg）的精排注入点——不依赖 lightrag 包（PG 后端要能在未装
LightRAG 时独立工作），契约对齐 ``hybrid_retriever.retrieve`` 的 ``rerank_fn``：
    async (query, docs: list[dict], top_n) -> list[dict]

DashScope aliyun rerank 格式（同 lightrag.rerank.ali_rerank，但不直接复用该函数以免
硬依赖 lightrag 包）：请求 ``input.documents`` + ``parameters.top_n``；响应
``output.results``，每项 ``{index, relevance_score}``。

无 ``EMBEDDING__API_KEY`` 或开关关闭（``force=False``）时返回 None，调用方据此跳过精排
（与 lightrag.rerank_adapter.build_rerank_func 同款惰性语义）。
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

import httpx

from settings import get_settings

logger = logging.getLogger(__name__)

# DashScope 文本精排服务端点（aliyun 格式，非 OpenAI 兼容 /v1）。
_DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
# 粗估 token：中文约 1.5 字符/token（与 settings.RerankConfig 的成本估算口径一致）。
_CHARS_PER_TOKEN = 1.5


def _estimate_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN)


async def _call_rerank_api(
    query: str,
    documents: list[str],
    *,
    model: str,
    api_key: str,
    top_n: int,
    timeout_s: float,
) -> list[dict]:
    """POST DashScope rerank，返回 ``[{index, relevance_score}, ...]``。

    ``top_n=len(documents)`` 时返回全部候选（按 relevance 降序）。非 200 / 解析失败抛异常，
    由调用方（hybrid_retriever）降级为融合结果（未精排）。
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": top_n, "return_documents": False},
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(_DASHSCOPE_RERANK_URL, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"DashScope rerank HTTP {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()

    results = (data or {}).get("output", {}).get("results", [])
    if not isinstance(results, list):
        logger.warning("rerank 响应 output.results 非 list: %r", type(results))
        return []
    out: list[dict] = []
    for r in results:
        try:
            out.append(
                {"index": int(r["index"]), "relevance_score": float(r["relevance_score"])}
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def build_rerank_fn(*, force: bool = False) -> Callable[..., Awaitable[list[dict]]] | None:
    """构建精排函数，或 None（无 API key / 开关关闭）。

    Args:
        force: True 时无视 ``settings.rerank.enabled`` 开关，只看 ``embedding.api_key``
            是否存在——消融实验用：生产默认关时仍要能跑出 ``hybrid_rrf+rerank`` 与
            ``hybrid_no_rerank`` 的对比。生产热路径（llamaindex_pg）用默认 ``force=False``
            尊重开关。该开关由 hybrid_retriever 的 ``config.rerank_enabled`` 逐配置决定是否
            真正调用，force 只控制「rerank_fn 本身是否被构造出来」。
    """
    settings = get_settings()
    api_key = settings.embedding.api_key.get_secret_value()
    if not api_key:
        return None
    if not force and not settings.rerank.enabled:
        return None

    cfg = settings.rerank
    model = cfg.model
    candidate_top_n = cfg.candidate_top_n
    max_request_tokens = cfg.max_request_tokens
    timeout_s = cfg.timeout_s

    async def rerank_fn(query: str, docs: list[dict], top_n: int = 5) -> list[dict]:
        """精排候选池：候选池截断 → token 预算保护 → 调 API → index 映回 + score 覆写。

        Args:
            query: 查询文本。
            docs: 融合后的候选 dict 列表（含 content/score），按 RRF/融合序。
            top_n: 调用方期望的精排条数（契约参数；实际请求全量打分返回整池，调用方自行切片）。

        Returns:
            重排后的 dict 列表：[按 relevance 排序的候选池] + [未打分候选] + [尾部未送候选]。
            用 relevance_score 覆写 score（统一量纲，RRF 分数与 relevance_score 不可比）。
            API 失败时向上抛异常，由 hybrid_retriever 降级返回融合结果。
        """
        if not docs:
            return list(docs)

        # 1. 候选池截断：只送前 candidate_top_n 条给 API，尾部候选原样保留（不丢结果，
        #    只是不参与重排序）。
        pool = docs[:candidate_top_n]
        tail = docs[candidate_top_n:]

        # 2. token 预算保护：query + 每个 doc 累加，超 max_request_tokens 则丢尾部候选避 400。
        #    默认规模（20×~1200 token）只用到约 25,000，不触发；是 candidate_top_n 被调大时的兜底。
        used = _estimate_tokens(query or "")
        budget_pool: list[dict] = []
        for d in pool:
            t = _estimate_tokens(d.get("content") or "")
            if used + t > max_request_tokens:
                break
            used += t
            budget_pool.append(d)
        if not budget_pool:
            # 极端：第一条就超预算——退回原序，不调 API（精排本就是兜底增强，不能拖垮主链路）。
            return list(docs)
        pool = budget_pool

        # 3. 调 API：请求全量打分（top_n=len(pool)），拿到整池按 relevance 的排序。
        contents = [d.get("content") or "" for d in pool]
        scored = await _call_rerank_api(
            query,
            contents,
            model=model,
            api_key=api_key,
            top_n=len(pool),
            timeout_s=timeout_s,
        )

        # 4. index 映回原 dict + score 覆写（量纲统一）。
        reranked: list[dict] = []
        seen: set[int] = set()
        for r in scored:
            idx = r["index"]
            seen.add(idx)
            if 0 <= idx < len(pool):
                doc = dict(pool[idx])
                # 保留精排前分数（fused_score 优先，回退 dense/sparse 原始 score），否则覆写后
                # 无法回答「精排把这条从第几名挪到第几名、原来分数多少」。score 覆写行为不变——
                # llamaindex_pg.retrieve 读 score 填 RetrievalResult.score，改它即改检索行为。
                doc["pre_rerank_score"] = doc.get("fused_score", doc.get("score", 0.0))
                doc["rerank_score"] = r["relevance_score"]
                doc["score"] = r["relevance_score"]
                reranked.append(doc)

        # 5. 未被打分的候选（API 过滤/截断）+ 尾部候选原序补后，不丢任何结果。
        leftover = [pool[i] for i in range(len(pool)) if i not in seen]
        return reranked + leftover + tail

    logger.info(
        "build_rerank_fn: reranker ready model=%s candidate_top_n=%d (force=%s)",
        model,
        candidate_top_n,
        force,
    )
    return rerank_fn


__all__ = ["build_rerank_fn"]
