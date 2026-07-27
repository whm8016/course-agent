"""回归：rag_runner 缓存必须按 course_id 隔离。

历史 bug：cache key 只含 {qid}_{mode}，不含 course_id → 基线索引和新切块索引共用
同一份缓存，对比评测时新索引会直接读到基线的 answer/contexts，结论失效。
本测试锁死「按课程分目录」的隔离语义。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 补 backend 根到 sys.path，使 scripts.eval_rag 可导入
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scripts.eval_rag import config, rag_runner  # noqa: E402


def test_cache_isolated_per_course(tmp_path, monkeypatch):
    """不同 course_id 的缓存互不可见（隔离的核心）。"""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    payload_a = {"answer": "基线答案", "contexts": ["c1"], "retrieve_ms": 10, "query_ms": 20}

    rag_runner.save_cache("s01", "fact", "mycourse", payload_a)

    # 同课程命中
    hit = rag_runner.load_cache("s01", "fact", "mycourse")
    assert hit is not None
    assert hit["answer"] == "基线答案"

    # 不同课程必须 miss —— 这正是旧 bug 失效的地方
    assert rag_runner.load_cache("s01", "fact", "mycourse_rf600") is None


def test_cache_path_under_course_subdir(tmp_path, monkeypatch):
    """缓存文件落在 {CACHE_DIR}/{course_id}/ 下，不再散落在根目录。"""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    rag_runner.save_cache("s02", "relationship", "mycourse_rf600", {"answer": "x"})

    expected = tmp_path / "mycourse_rf600" / "s02_relationship_v2.json"
    assert expected.exists()
    # 根目录不残留（旧 bug 的文件直接堆在根）
    assert not (tmp_path / "s02_relationship_v2.json").exists()


def test_cache_path_sanitizes_unsafe_course_id(tmp_path, monkeypatch):
    """course_id 含路径分隔符等特殊字符时做无害化，防止越出 cache 目录。"""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    p = rag_runner._cache_path("s01", "fact", "evil/../../etc")
    # 特殊字符（含 / 和 .）被替换成 _，不会向上穿越
    assert ".." not in p.parts
    assert tmp_path in p.parents  # 仍落在 cache 目录内
