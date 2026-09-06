"""单次 (task, arm) 运行的接线层：注册工具、组装两臂差异、调 run_agent_loop、收指标。

两臂共用生产调度内核，只在"工具怎么进入模型视野"这一层分叉（见包 docstring）：

- deferred_exact（现状臂）：render_deferred_tools_manifest 全量索引进 system prompt，
  挂 load_tools schema，loader 走**生产** contextvar（core.agentic.dynamic_tools.
  _CURRENT_LOADER）——_execute_load_tools 原样可用，不改生产代码。
- active_discovery（新臂）：system prompt 零工具索引，只挂 discover_tools schema，
  DiscoverToolLoader 走 discovery.py 自己的 contextvar。

执行路由两臂同一条：mock 工具一次性注册进 get_tool_registry()，loop 的
dispatch_tool_calls → execute_tool → registry 查表，复用生产分发。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from . import config  # noqa: F401  # bootstrap 副作用（sys.path/.env）必须最先执行

logger = logging.getLogger(__name__)

ARMS = ("deferred_exact", "active_discovery")

_BASE_PROMPT = (
    "你是一个务实的任务执行助手。请使用可用的工具完成用户的任务，"
    "拿到所需信息后，用中文给出简明的结果总结。不要编造工具没有返回的数据。"
)

# discover 臂的引导块：替代全量 manifest，告诉模型"没有清单，按需发现"
_DISCOVER_PROMPT_BLOCK = (
    "## 工具发现\n"
    "本会话没有提供任何工具清单。任务需要外部能力（查询、检索、画图、文件操作等）时，"
    "先调用 `discover_tools`，用一句自然语言描述你需要的能力（如「查询两个国家货币之间"
    "的汇率」），匹配的工具会立即加载进本会话并可调用；一次描述一种能力，可以多次调用。"
    "拿到足够信息后再作答。"
)


def ensure_setup() -> Callable[[], None]:
    """process 级初始化：session 重定向 + 关生产侧写 + 合成工具注册；返回还原闭包。

    突变与还原清单同源在此（测试 fixture 只需 ``restore = ensure_setup(); ...; restore()``，
    不再人肉镜像维护）。真实评测进程忽略返回值（进程退出即还原）：

    - ``core.mcp.session_state._SESSIONS_BASE`` 指向 results/_sessions/，避免 arm A 的
      load_tools 持久化往真实 data/sessions/ 写评测垃圾（_state_file 调用时读模块全局，
      赋值即生效）。
    - 关 ``usage_tracking`` / ``cost_quota``（进程内 settings 单例，只影响本评测进程）：
      否则真实评测每 case 往生产 llm_usage_records 插一行账单、并向 Redis 累计配额。
    - mock 工具 + discover_tools 注册进进程级 ToolRegistry；注册池与 run_case 用的
      池子同源（build_pool(TOOL_POOL_SIZE)），裁剪时 registry 不残留池外可调用工具。

    各步幂等（mkdir exist_ok / 赋值 / register 同名覆盖），重复调用安全——还原闭包
    还原的是**首次调用前**的原值，故不要嵌套调用。
    """
    import core.mcp.session_state as _ss

    from settings import get_settings

    _settings = get_settings()
    _orig_base = _ss._SESSIONS_BASE
    _orig_usage = _settings.usage_tracking.enabled
    _orig_quota = _settings.cost_quota.enabled

    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _ss._SESSIONS_BASE = config.SESSIONS_DIR
    _settings.usage_tracking.enabled = False
    _settings.cost_quota.enabled = False

    from . import synthetic_tools
    from .discovery import register_discover_tool
    from core.agent.registry import get_tool_registry

    _registry = get_tool_registry()
    _pool = synthetic_tools.build_pool(config.TOOL_POOL_SIZE)
    synthetic_tools.register_mock_tools(_registry, _pool)
    register_discover_tool(_registry)
    _names = [a.wrapped_name for a in _pool] + ["discover_tools"]

    def _restore() -> None:
        for name in _names:
            _registry.unregister(name)
        _settings.usage_tracking.enabled = _orig_usage
        _settings.cost_quota.enabled = _orig_quota
        _ss._SESSIONS_BASE = _orig_base

    return _restore


def _metrics_from_outcome(
    task: dict,
    arm: str,
    outcome: Any,
    *,
    system_prompt_chars: int,
    wall_ms: int,
    discover_count: int,
    loaded_names: set[str],
) -> dict:
    """从 LoopOutcome + 臂侧状态提取指标。

    ``guessed_calls``：被调用的领域工具中**未经 load/discover 揭示**的名字——registry
    全量注册（与生产 manager 同口径），模型可以只看 manifest 凭猜名直调；这个指标把
    "发现后调用"与"猜名直调"分开，避免 recall 虚高误读。
    """
    expected = set(task["expected_tools"])
    tools_used = list(outcome.tools_used)
    # 命中率只看领域工具（mcp_*）：meta 工具（load_tools/discover_tools）不算误召
    domain_used = [t for t in tools_used if str(t).startswith("mcp_")]
    hits = expected & set(domain_used)
    usage = outcome.total_usage  # LoopOutcome 恒有（default_factory）
    return {
        "task_id": task["id"],
        "arm": arm,
        "error": None,
        "completed": outcome.completed,
        "rounds": outcome.rounds,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "tools_used": tools_used,
        "domain_tools_used": domain_used,
        "recall": round(len(hits) / len(expected), 4) if expected else 0.0,
        "missed_tools": sorted(expected - set(domain_used)),
        "false_calls": sorted(set(domain_used) - expected),
        "guessed_calls": sorted(set(domain_used) - loaded_names),
        "discover_calls": discover_count,
        "load_tools_used": "load_tools" in tools_used,
        "system_prompt_chars": system_prompt_chars,
        "wall_ms": wall_ms,
        "final_text": (outcome.final_text or "")[:500],
    }


async def run_case(
    task: dict,
    arm: str,
    *,
    loop_fn=None,
    model: str | None = None,
    max_iterations: int | None = None,
    run_tag: str = "",
) -> dict:
    """跑单个 (task, arm)，返回指标 dict。loop_fn 可注入 mock loop（离线测试）。

    前置：调用方须已 ``ensure_setup()``（run_eval.main / 测试 fixture 各自负责，
    process 级初始化不藏在 per-case 里）。
    fail-fast 口径：任务 expected_tools 不在合成池内直接抛 ValueError（池子被
    TOOL_POOL_SIZE 裁剪过时提示先跑全量）；LLM 调用失败记 error 不中断整批。
    """
    from . import synthetic_tools
    from .discovery import DISCOVER_TOOLS_SCHEMA, DiscoverToolLoader, _CURRENT_DISCOVER_LOADER

    if arm not in ARMS:
        raise ValueError(f"未知 arm: {arm}（可选 {ARMS}）")

    pool = synthetic_tools.build_pool(config.TOOL_POOL_SIZE)
    pool_names = {a.wrapped_name for a in pool}
    missing = [t for t in task["expected_tools"] if t not in pool_names]
    if missing:
        raise ValueError(
            f"任务 {task['id']} 的 expected_tools 不在合成池内: {missing}"
            f"（当前 TOOL_POOL_SIZE={config.TOOL_POOL_SIZE}，对比评测请用全量 80）"
        )

    from core.agentic.dynamic_tools import LOAD_TOOLS_SCHEMA, _CURRENT_LOADER
    from core.context import UnifiedContext
    from core.mcp.deferred_tools import DeferredToolLoader, render_deferred_tools_manifest
    from core.stream_bus import StreamBus

    sid = f"demo-{arm}-{task['id']}" + (f"-{run_tag}" if run_tag else "")
    ctx = UnifiedContext(
        course_id="general",
        user_id="eval",
        user_message=task["prompt"],
        session_id=sid,
        mode="chat",
        enabled_tools=[],
    )

    discover_count = 0
    loader_token = None
    discover_token = None
    discover_loader: DiscoverToolLoader | None = None
    deferred_loader = None

    rec: dict[str, Any]
    t0 = time.perf_counter()
    try:
        if arm == "deferred_exact":
            # 与生产组装的一处已知偏差：生产 chat 还会经 build_tool_hint_text 追加
            # load_tools 的 hint 块（prompting/__init__.py）；manifest 头部已自带
            # load_tools 用法说明，demo 直接拼 manifest，不再补该块。
            system_prompt = _BASE_PROMPT + "\n\n" + render_deferred_tools_manifest(pool)
            tool_schemas: list[dict[str, Any]] = [LOAD_TOOLS_SCHEMA]
            # 评测口径：每 case 干净起点（loaded=空集），不读 results/_sessions 里
            # 上次运行的持久化——同 sid 重跑才可复现；load() 触发的 _persist 落在
            # 重定向后的 _SESSIONS_BASE，不污染真实 data/sessions/。
            loader = DeferredToolLoader(pool=pool, session_id=sid, loaded=set())
            deferred_loader = loader
            tool_schemas.extend(loader.initial_schemas())
            loader.bind_live_schemas(tool_schemas)
            loader_token = _CURRENT_LOADER.set(loader)
        else:
            system_prompt = _BASE_PROMPT + "\n\n" + _DISCOVER_PROMPT_BLOCK
            tool_schemas = [DISCOVER_TOOLS_SCHEMA]
            discover_loader = DiscoverToolLoader(pool=pool, top_k=config.DISCOVER_TOP_K)
            discover_loader.bind_live_schemas(tool_schemas)
            discover_token = _CURRENT_DISCOVER_LOADER.set(discover_loader)

        if loop_fn is None:
            from core.agentic.loop import run_agent_loop

            loop_fn = run_agent_loop

        outcome = await loop_fn(
            context=ctx,
            stream=StreamBus(),
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            model=model or config.MODEL,
            max_iterations=max_iterations if max_iterations is not None else config.MAX_ITERATIONS,
        )
        if discover_loader is not None:
            discover_count = len(discover_loader.discover_queries)
        loaded_names: set[str] = set()
        if deferred_loader is not None:
            loaded_names = deferred_loader.loaded_names
        elif discover_loader is not None:
            loaded_names = discover_loader.loaded_names
        rec = _metrics_from_outcome(
            task, arm, outcome,
            system_prompt_chars=len(system_prompt),
            wall_ms=int((time.perf_counter() - t0) * 1000),
            discover_count=discover_count,
            loaded_names=loaded_names,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[eval_mcp_discovery] %s/%s 运行失败", arm, task["id"])
        rec = {
            "task_id": task["id"],
            "arm": arm,
            "error": repr(exc),
            "wall_ms": int((time.perf_counter() - t0) * 1000),
        }
    finally:
        from .discovery import reset_discover_loader as _rd
        from core.agentic.dynamic_tools import reset_deferred_loader as _rl

        if loader_token is not None:
            _rl(loader_token)
        if discover_token is not None:
            _rd(discover_token)

    return rec


__all__ = ["ARMS", "ensure_setup", "run_case"]
