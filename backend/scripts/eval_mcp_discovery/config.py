"""MCP 主动发现 vs deferred 清单对比 demo 配置。

复用 eval_context / eval_rag 的 sys.path + .env 注入范式（scripts/ 是 namespace 包，
python -m scripts.eval_mcp_discovery.run_eval 直接跑）。本文件必须在该包其它模块
import 生产代码之前先被 import（bootstrap 副作用）。
"""
import os
import sys
from pathlib import Path

# 让 import 找到 backend 根（scripts/eval_mcp_discovery/ → backend/）
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# 加载 backend/.env（override=False 尊重上层预设；与 eval_context/config.py 同款）
try:
    from dotenv import load_dotenv

    load_dotenv(Path(_BACKEND_ROOT) / ".env", override=False)
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
EVAL_DIR: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = EVAL_DIR / "results"
# deferred load_tools 的 per-session 持久化目录（重定向 session_state._SESSIONS_BASE，
# 避免往真实 data/sessions/ 写评测垃圾）
SESSIONS_DIR: Path = RESULTS_DIR / "_sessions"

# ---------------------------------------------------------------------------
# 旋钮（env 前缀 EVAL_MCP_DISCOVERY_）
# ---------------------------------------------------------------------------
# 模型：默认生产 chat 主模型，env 可覆盖
MODEL: str | None = os.getenv("EVAL_MCP_DISCOVERY_MODEL")  # None → settings.llm.text_model
# 合成工具池规模（全池 80；调小用于快速 smoke，run_eval 会过滤掉 expected 工具不在池内的任务）
TOOL_POOL_SIZE: int = int(os.getenv("EVAL_MCP_DISCOVERY_POOL_SIZE", "80"))
# discover_tools 单次语义检索返回 top-k
DISCOVER_TOP_K: int = int(os.getenv("EVAL_MCP_DISCOVERY_TOP_K", "4"))
# 单回合最大 LLM 轮数（与生产 DEFAULT_MAX_ITERATIONS 对齐）
MAX_ITERATIONS: int = int(os.getenv("EVAL_MCP_DISCOVERY_MAX_ITER", "10"))
# 单条 case 间延迟（秒），防 rate limit
QUERY_DELAY: float = float(os.getenv("EVAL_MCP_DISCOVERY_DELAY", "1"))
