"""表A 两臂主表口径回归：两臂名单、引用正则、源片段命中率、contexts 形状、缓存精排后缀。

历史 bug（本文件锁死）：
  1. has_citation 的「根据.{0,6}资料」分支把闭卷臂拒答话术「根据现有资料暂无法确认」
     误判成引用 → no_rag 可追溯率虚高 0.2222。
  2. 缓存键不带精排状态：RERANK__ENABLED=true 重跑时读到无精排旧答案，主表结论失效。
  3. pg 线（llamaindex_pg）contexts 用 ``\\n\\n`` 拼接成整串单条，按 ``\\n\\n---\\n\\n``
     拆不开 → rank 恒 0（Top-1≡Top-5、RAGChecker/RAGAS 粒度退化）。拆分修复在
     rag_runner._split_context_string（按共享 [证据N] 头二道拆），load_cache 对旧
     blob 缓存归一——本文件在 rag_runner 层与评测层各锁一环。
"""
from __future__ import annotations

import importlib
import os

import pytest

from scripts.eval_rag import config, rag_runner
from scripts.eval_rag import run_table_ab


# ---------------------------------------------------------------------------
# 两臂名单
# ---------------------------------------------------------------------------
def test_table_a_modes_collapsed_to_two_arms():
    """主表只留闭卷 vs 本系统；naive_rag（流水线相同无对照价值）/oracle_rag 已砍。"""
    assert run_table_ab.TABLE_A_MODES == ["no_rag", "fact"]


# ---------------------------------------------------------------------------
# 可追溯率正则：拒答话术不是引用
# ---------------------------------------------------------------------------
def test_has_citation_refusal_phrase_not_counted():
    # 历史 bug 根因：拒答话术命中「根据.{0,6}资料」被当成引用
    assert not run_table_ab.has_citation("根据现有资料暂无法确认该器件的量程。")
    assert not run_table_ab.has_citation("根据现有资料中暂未提及考核方式。")
    assert not run_table_ab.has_citation("无法从现有资料确认。")


def test_has_citation_positive_and_negative():
    assert run_table_ab.has_citation("量程为22种。[证据1]")
    assert run_table_ab.has_citation("来源：实验讲义第三章")
    assert run_table_ab.has_citation("详见参考资料部分。")
    assert not run_table_ab.has_citation("量程为22种，共八个子实验。")
    assert not run_table_ab.has_citation("")
    # 「参考资料」的近亲拒答句不算引用（真引用用法不受影响）
    assert not run_table_ab.has_citation("参考资料中未提及该型号的量程。")
    assert not run_table_ab.has_citation("参考资料中暂未覆盖考核方式。")


# ---------------------------------------------------------------------------
# contexts 形状（rag_runner 层）：pg 线 blob 拆分 + 旧缓存归一
# ---------------------------------------------------------------------------
def test_split_context_string_pg_line_blob():
    """pg 线（llamaindex_pg）用 \n\n 拼接、无 --- 分隔：按共享的 [证据N] 头拆开。"""
    blob = "[证据1｜来源: a.docx]\n第一块\n\n[证据2｜来源: b.docx]\n第二块"
    assert rag_runner._split_context_string(blob) == [
        "[证据1｜来源: a.docx]\n第一块", "[证据2｜来源: b.docx]\n第二块",
    ]


def test_split_context_string_already_split_unchanged():
    """lightrag 线（--- 分隔）与单头块拆完不变；无头串原样保留。"""
    assert rag_runner._split_context_string("[证据1]a\n\n---\n\n[证据2]b") == [
        "[证据1]a", "[证据2]b",
    ]
    assert rag_runner._split_context_string("单条") == ["单条"]
    assert rag_runner._split_context_string("") == []


def test_load_cache_normalizes_legacy_blob_contexts(tmp_path, monkeypatch):
    """旧 pg 线缓存是整串单条 context（拆分修复前落盘）：load_cache 必须归一成逐块。"""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    blob = "[证据1｜来源: a.docx]\n第一块\n\n[证据2｜来源: b.docx]\n第二块"
    rag_runner.save_cache("q1", "fact", "mycourse", {"answer": "x", "contexts": [blob]})
    loaded = rag_runner.load_cache("q1", "fact", "mycourse")
    assert len(loaded["contexts"]) == 2
    assert loaded["contexts"][0].startswith("[证据1")


# ---------------------------------------------------------------------------
# 源片段命中率（零 LLM，评测层）：contexts 已由 rag_runner 保证逐条形状
# ---------------------------------------------------------------------------
def _mk_results(*ctx_lists: list[str]) -> list[dict]:
    return [{"contexts": list(c)} for c in ctx_lists]


def test_source_hit_rates_top1_and_top5(monkeypatch):
    """rank0 命中算 Top-1；首个命中落在前 5 内算 Top-5；无 gold 映射的题不入 n。"""
    qa = [
        {"id": "q1", "question": "q", "ground_truth": "g"},
        {"id": "q2", "question": "q", "ground_truth": "g"},
        {"id": "q3", "question": "q", "ground_truth": "g"},  # 无 gold 映射
    ]
    # extract_anchors("AAAA1111 BBBB") → 滑窗 AAAA/AA11/1111 + BBBB 共 4 锚点
    gold = {"q1": "AAAA1111 BBBB", "q2": "CCCC2222 DDDD"}
    monkeypatch.setattr(rag_runner, "load_oracle_contexts", lambda: gold)

    results = _mk_results(
        ["AAAA 与 AA11 及 1111", "无关"],          # q1：rank0 覆盖 3/4 ≥0.5 → Top-1（也计入 Top-5）
        ["只有 CCCC", "无关", "无关", "CCCC CC22 2222 DDDD 全有"],  # q2：rank3 全覆盖 → 仅 Top-5
        ["随便"],                                    # q3：无映射，不参与
    )
    # Top-1 ⊆ Top-5：q1、q2 都在前 5 内命中 → top5=1.0；只有 q1 是 rank0 → top1=0.5
    assert run_table_ab.source_hit_rates(qa, results) == {"top1": 0.5, "top5": 1.0, "n": 2}


def test_source_hit_rates_rank_semantics_on_blob_cache(tmp_path, monkeypatch):
    """端到端：旧 blob 缓存经 load_cache 归一后，rank 语义落在拆后块上（rank2 → 仅 Top-5）。"""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rag_runner, "load_oracle_contexts", lambda: {"q1": "AAAA1111 BBBB"})
    blob = (
        "[证据1｜来源: a.docx]\n只有 AAAA\n\n"
        "[证据2｜来源: b.docx]\n无关\n\n"
        "[证据3｜来源: c.docx]\nAAAA AA11 1111 BBBB 全有"
    )
    rag_runner.save_cache("q1", "fact", "mycourse", {"answer": "x", "contexts": [blob]})
    qa = [{"id": "q1", "question": "q", "ground_truth": "g"}]
    res = [rag_runner.load_cache("q1", "fact", "mycourse")]
    assert run_table_ab.source_hit_rates(qa, res) == {"top1": 0.0, "top5": 1.0, "n": 1}


def test_source_hit_rates_rank_cap_at_5(monkeypatch):
    """首个命中落在 rank≥5 不计 Top-5（k=10/20 sweep 下 contexts 更长）。"""
    qa = [{"id": "q1", "question": "q", "ground_truth": "g"}]
    monkeypatch.setattr(rag_runner, "load_oracle_contexts", lambda: {"q1": "AAAA1111 BBBB"})
    ctxs = ["无关"] * 6 + ["AAAA AA11 1111 BBBB"]  # 首个命中 rank6
    assert run_table_ab.source_hit_rates(qa, [{"contexts": ctxs}]) == {
        "top1": 0.0, "top5": 0.0, "n": 1,
    }


def test_source_hit_rates_coverage_below_threshold(monkeypatch):
    """覆盖 <0.5 不算命中（与 make_oracle_gold 的 min_cover 同口径）。"""
    qa = [{"id": "q1", "question": "q", "ground_truth": "g"}]
    monkeypatch.setattr(rag_runner, "load_oracle_contexts", lambda: {"q1": "AAAA1111 BBBB"})
    # 只覆盖 1/4 = 0.25 < 0.5
    assert run_table_ab.source_hit_rates(qa, _mk_results(["只有 AAAA"])) == {
        "top1": 0.0, "top5": 0.0, "n": 1,
    }


def test_source_hit_rates_empty_gold(monkeypatch):
    monkeypatch.setattr(rag_runner, "load_oracle_contexts", lambda: {})
    qa = [{"id": "q1", "question": "q", "ground_truth": "g"}]
    assert run_table_ab.source_hit_rates(qa, _mk_results(["x"])) == {}


def test_source_hit_rates_empty_contexts_never_hit(monkeypatch):
    """闭卷臂 contexts 恒空 → 恒 0（compose 时标「—」，但 JSON 里有确定数字）。"""
    qa = [{"id": "q1", "question": "q", "ground_truth": "g"}]
    monkeypatch.setattr(rag_runner, "load_oracle_contexts", lambda: {"q1": "AAAA1111 BBBB"})
    assert run_table_ab.source_hit_rates(qa, _mk_results([])) == {"top1": 0.0, "top5": 0.0, "n": 1}


# ---------------------------------------------------------------------------
# 缓存键精排后缀（_rerank_enabled 与 build_rerank_fn 同读 settings 的 lru_cache
# 快照——切环境变量后必须刷新快照才生效，这与 run_table_ab.main 先设 env 再触发
# 首次 get_settings 的时序约束一致）
# ---------------------------------------------------------------------------
@pytest.fixture()
def rerank_settings(monkeypatch):
    """提供 set_rerank(value|None)：设/删 RERANK__ENABLED 并同步刷新 settings 快照。"""
    from settings import get_settings

    def _set(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv("RERANK__ENABLED", raising=False)
        else:
            monkeypatch.setenv("RERANK__ENABLED", value)
        get_settings.cache_clear()

    get_settings.cache_clear()
    yield _set
    get_settings.cache_clear()


def test_cache_path_rerank_suffix(rerank_settings, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    rerank_settings("true")
    assert rag_runner._cache_path("s01", "fact", "mycourse").name == "s01_fact_v2_rr.json"
    # 精排与 top_k 后缀叠加
    assert rag_runner._cache_path("s01", "fact", "mycourse", top_k=10).name == \
        "s01_fact_v2_k10_rr.json"
    # 闭卷/oracle 臂不经检索，answer 与精排无关：不加后缀，旧缓存继续命中省 LLM
    assert rag_runner._cache_path("s01", "no_rag", "mycourse").name == "s01_no_rag_v2.json"
    assert rag_runner._cache_path("s01", "oracle_rag", "mycourse").name == \
        "s01_oracle_rag_v2.json"


def test_cache_path_no_suffix_when_rerank_off(rerank_settings, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    rerank_settings(None)
    # 生产/其他脚本默认关（.env 亦是 false）：保持旧文件名，向后兼容已有缓存
    assert rag_runner._cache_path("s01", "fact", "mycourse").name == "s01_fact_v2.json"


def test_rerank_cache_isolation_roundtrip(rerank_settings, tmp_path, monkeypatch):
    """开精排后必须读不到无精排旧缓存（本次两臂主表的核心隔离语义）。"""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    rerank_settings(None)
    rag_runner.save_cache("s01", "fact", "mycourse", {"answer": "无精排旧答案"})
    rerank_settings("true")
    assert rag_runner.load_cache("s01", "fact", "mycourse") is None


# ---------------------------------------------------------------------------
# 环境卫生：import run_table_ab 不得污染进程环境
# ---------------------------------------------------------------------------
def test_import_does_not_flip_rerank_env(monkeypatch):
    """RERANK__ENABLED 只在 main() 里置 true：模块被 import（测试/复用）时不得设置，
    否则同进程其他走 rag_runner 缓存的代码会拿到意外的 _rr 后缀。先固定为 false
    再 reload——load_dotenv(override=False) 不会覆盖已存在的环境变量，测试不依赖
    开发者本机 .env 的内容。"""
    monkeypatch.setenv("RERANK__ENABLED", "false")
    importlib.reload(run_table_ab)
    assert os.environ["RERANK__ENABLED"].strip().lower() not in ("1", "true", "yes", "on")
