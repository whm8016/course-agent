"""RAG 评测系统配置。

读取项目主配置（backend/config.py）中的 DashScope / LightRAG 参数，
在此基础上定义评测专用的默认值。
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 让 import 能找到 backend 根目录（scripts/eval_rag/ → backend/）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# ---------------------------------------------------------------------------
# 课程与模式
# ---------------------------------------------------------------------------
COURSE_ID: str = "circuit_analysis"

MODES: list[str] = ["naive", "local", "global", "mix"]

# ---------------------------------------------------------------------------
# RAGAS 指标
# ---------------------------------------------------------------------------
METRICS_ALL: list[str] = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
]

# 零 LLM 成本指标，适合快速验证
METRICS_NO_LLM: list[str] = ["context_precision", "context_recall"]

# ---------------------------------------------------------------------------
# DashScope LLM / Embedding（RAGAS 内部使用）
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.getenv("TEXT_MODEL", "qwen-plus")
LLM_API_KEY: str = os.getenv("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL: str = os.getenv(
    "EMBEDDING_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

EMBED_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
EMBED_API_KEY: str = os.getenv("EMBEDDING_API_KEY") or LLM_API_KEY
EMBED_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL") or LLM_BASE_URL

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
EVAL_DIR: Path = Path(__file__).resolve().parent
CACHE_DIR: Path = EVAL_DIR / "cache"
RESULTS_DIR: Path = EVAL_DIR / "results"
DATASET_PATH: Path = EVAL_DIR / "qa_dataset.json"

# ---------------------------------------------------------------------------
# 查询间隔（秒）——避免 DashScope 限流
# ---------------------------------------------------------------------------
QUERY_DELAY: float = float(os.getenv("EVAL_QUERY_DELAY", "1.5"))
