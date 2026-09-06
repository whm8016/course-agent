"""主动发现机制：discover_tools 语义检索元工具（demo 的"新臂"）。

结构刻意对齐 core/mcp/deferred_tools.py 而非照搬 chapter4 的文本 ReAct 协议——
本项目走 OpenAI 原生 tool_calls：

- ``LocalEmbedder``：本地哈希词袋嵌入（中文字 unigram+bigram、英文按词），移植自
  chapter4 active-tool-discovery/offline_backend.py 的 ``_tokens``/``embed``，零 API
  key 依赖，demo 可直接跑。
- ``ToolIndex``：对工具池的 ``name: description`` 建向量索引，余弦最近邻 top-k。
- ``DiscoverToolLoader``：per-turn 句柄，形状照抄 ``DeferredToolLoader``
  （bind_live_schemas 绑定本轮可变 tool_schemas list、命中后 .append() 末尾——同一
  list 引用下一轮即对模型可见，loop.py 的 ``schemas = None if final else tool_schemas``
  机制原样复用），只是查找方式从"精确 name in pool"换成"embedding top-k"。
- ``DISCOVER_TOOLS_SCHEMA`` + ``execute_discover_tools``：参数是自然语言 ``need``；
  执行器经**独立** contextvar 拿当前 turn 的 loader（同 dynamic_tools._CURRENT_LOADER
  的注入模式，但不复用生产的那个）。
"""
from __future__ import annotations

import contextvars
import hashlib
import math
import re
from typing import Any

from core.agent.tool_protocol import ToolResult
from core.mcp.adapter import MCPToolAdapter

# ---------------------------------------------------------------------------
# 本地嵌入（移植 chapter4 offline_backend.py，保持算法逐行等价）
# ---------------------------------------------------------------------------
_DIM = 512


def _tokens(text: str) -> list[str]:
    """中英文混合文本切词袋：英文按词，中文按字 unigram + bigram。"""
    text = text.lower()
    toks: list[str] = []
    for w in re.findall(r"[a-z0-9]+", text):
        toks.append(w)
    han = re.findall(r"[一-鿿]", text)
    toks += han
    toks += [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return toks


class LocalEmbedder:
    """哈希词袋嵌入：确定性、无需联网。相似度由中英文关键词重叠驱动。"""

    name = f"local-hash-{_DIM}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * _DIM
            for tok in _tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % _DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 工具向量索引
# ---------------------------------------------------------------------------
class ToolIndex:
    """对 ``name: description`` 建向量索引；search(need) 返回余弦最近邻 top-k。"""

    def __init__(self, tools: list[MCPToolAdapter] | dict[str, MCPToolAdapter], embedder: LocalEmbedder | None = None) -> None:
        if isinstance(tools, dict):
            tools = list(tools.values())
        self._tools: list[MCPToolAdapter] = list(tools)
        self._embedder = embedder or LocalEmbedder()
        self._texts = [f"{a.wrapped_name}: {a.description}" for a in self._tools]
        self._vecs = self._embedder.embed(self._texts) if self._tools else []

    def search(self, need: str, top_k: int = 4) -> list[tuple[MCPToolAdapter, float]]:
        """need（自然语言能力描述）→ [(adapter, score)] 余弦降序 top-k。"""
        if not self._tools or not need.strip():
            return []
        q = self._embedder.embed([need])[0]
        scored = [(_cosine(q, v), a) for a, v in zip(self._tools, self._vecs)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(a, s) for s, a in scored[:top_k]]


# ---------------------------------------------------------------------------
# per-turn loader（形状对齐 DeferredToolLoader）
# ---------------------------------------------------------------------------
class DiscoverToolLoader:
    """per-turn 句柄：把语义检索命中的工具 schema append 进 live list。

    与 DeferredToolLoader 同一机制骨架（bind_live_schemas / initial_schemas /
    追加后下一轮立即可见），差异仅在查找方式：load(names) 精确名 → discover(need)
    embedding top-k。无 session 持久化（demo 单 turn 用完即弃）。
    """

    def __init__(self, *, pool: list[MCPToolAdapter] | dict[str, MCPToolAdapter], top_k: int = 4) -> None:
        if isinstance(pool, dict):
            self._pool: dict[str, MCPToolAdapter] = dict(pool)
        else:
            self._pool = {a.wrapped_name: a for a in pool}
        self._top_k = top_k
        self._index = ToolIndex(self._pool, LocalEmbedder())
        self._loaded: set[str] = set()
        self._live_schemas: list[dict[str, Any]] | None = None
        self.discover_queries: list[str] = []  # 供指标采集：模型实际发出的 need 列表

    @property
    def loaded_names(self) -> set[str]:
        return set(self._loaded)

    def bind_live_schemas(self, schemas: list[dict[str, Any]]) -> None:
        """绑定本 turn 的 live tool_schemas list（原地 mutate，同 DeferredToolLoader）。"""
        self._live_schemas = schemas

    def initial_schemas(self) -> list[dict[str, Any]]:
        """初始为空（demo 每 task 全新 loader；生产语义下此处返回已加载集合）。"""
        return []

    def discover(self, need: str, top_k: int | None = None) -> dict[str, list[dict[str, str]]]:
        """按 need 语义检索 top-k 并加载进 live list；返回命中明细。

        返回 {"loaded": [{"name","description"}...], "already_loaded": [...]}——
        description 一并返回，供 tool result 文本告知模型"现在多了什么能力"。
        """
        self.discover_queries.append(str(need))
        k = top_k or self._top_k
        loaded: list[dict[str, str]] = []
        already: list[dict[str, str]] = []
        for adapter, _score in self._index.search(need, k):
            if adapter.wrapped_name in self._loaded:
                already.append({"name": adapter.wrapped_name, "description": adapter.description})
                continue
            if self._live_schemas is not None:
                self._live_schemas.append(adapter.to_openai_schema())
            self._loaded.add(adapter.wrapped_name)
            loaded.append({"name": adapter.wrapped_name, "description": adapter.description})
        return {"loaded": loaded, "already_loaded": already}


# ---------------------------------------------------------------------------
# discover_tools 元工具（schema + contextvar + executor）
# ---------------------------------------------------------------------------
DISCOVER_TOOLS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "discover_tools",
        "description": (
            "按能力需求语义检索并加载工具。系统没有提供任何工具清单：当你判断任务需要"
            "某种能力（如『查询外汇汇率』『检索学术论文』）而当前可用工具不足时，用一句"
            "自然语言描述需要的能力，匹配的工具会立即加载进本会话并可直接调用。"
            "可以多次调用，每次描述一种能力需求。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "need": {
                    "type": "string",
                    "description": "需要的能力描述，如『查询两个国家货币之间的汇率』",
                },
            },
            "required": ["need"],
        },
    },
}

# 当前 turn 的 DiscoverToolLoader（独立于生产 _CURRENT_LOADER，避免互相干扰）
_CURRENT_DISCOVER_LOADER: contextvars.ContextVar[DiscoverToolLoader | None] = contextvars.ContextVar(
    "discover_loader", default=None
)


def current_discover_loader() -> DiscoverToolLoader | None:
    return _CURRENT_DISCOVER_LOADER.get()


def reset_discover_loader(token: contextvars.Token | None) -> None:
    """turn 结束时清理 loader（token 来自 set 返回值，同 reset_deferred_loader）。"""
    if token is not None:
        _CURRENT_DISCOVER_LOADER.reset(token)


async def execute_discover_tools(need: str = "", **kwargs: Any) -> ToolResult:
    """discover_tools 执行器：经 contextvar 取本 turn loader，语义检索并加载。"""
    loader = current_discover_loader()
    if loader is None:
        return ToolResult(content="discover_tools 在当前会话不可用（未配置工具发现）。", success=False)
    need = str(need or "").strip()
    if not need:
        return ToolResult(content="discover_tools 需要 need 参数（一句话描述需要的能力）。", success=False)
    outcome = loader.discover(need)
    parts: list[str] = []
    if outcome["loaded"]:
        parts.append("已发现并加载以下工具（现在可调用）：")
        parts += [f"- {t['name']}：{t['description']}" for t in outcome["loaded"]]
    if outcome["already_loaded"]:
        parts.append("以下匹配工具此前已加载：" + "、".join(t["name"] for t in outcome["already_loaded"]))
    if not outcome["loaded"] and not outcome["already_loaded"]:
        parts.append("（无匹配工具，请换一种能力描述再试）")
    return ToolResult(content="\n".join(parts), success=bool(outcome["loaded"] or outcome["already_loaded"]))


def register_discover_tool(registry: Any) -> None:
    """把 discover_tools 注册进 ToolRegistry（执行路由复用 execute_tool）。"""
    from core.agent.registry import ToolEntry

    registry.register(ToolEntry(
        name="discover_tools",
        schema=DISCOVER_TOOLS_SCHEMA,
        executor=execute_discover_tools,
    ))


__all__ = [
    "LocalEmbedder",
    "ToolIndex",
    "DiscoverToolLoader",
    "DISCOVER_TOOLS_SCHEMA",
    "current_discover_loader",
    "reset_discover_loader",
    "execute_discover_tools",
    "register_discover_tool",
]
