"""表A（两臂主表）+ 表B（越界拒答）评测主脚本。

表A 两条臂（同一回答模型 EVAL_ANSWER_MODEL，只差证据来源）：
  1. no_rag  闭卷直答（模型自带知识天花板）
  2. fact    本系统：PG 线全流水线（结构化切块 + dense + PG tsvector + RRF + 精排）

naive_rag（检索流水线与本系统完全相同，F1 0.780 vs 0.781 无对照价值）与 oracle_rag
（25/27 映射，两道空题反把上界拉到最低）已从主表砍掉——「检索改进拿到多少」由七级
消融表（run_eval --ablation）承担；两臂实现仍留在 rag_runner，需要时可加回。

跨臂指标（不需要检索片段，两臂可比，写进摘要）：
  - RAGChecker overall：precision / recall / f1（陈述级，claim 分解+蕴含判断；
    三者的比对矩阵只依赖 GT 与回答、不依赖检索片段 → 闭卷臂也可算，且是其最关键证据）
  - 关键要点覆盖率（数字+专名锚点在回答中出现的比例，纯程序比对零 LLM）
  - 可追溯率（回答引用来源的比例；no_rag 恒 0）

仅 fact 臂报（闭卷臂无检索片段，定义上不适用，表注写明）：
  - RAGChecker generator：self_knowledge（恒为 1）/ hallucination（恒 = 1-precision）
    / faithfulness——不当数字比
  - 源片段 Top-1 / Top-5 命中率（oracle_gold 锚点比对，零 LLM，n=有映射题数）

表B：20 道越界题（gold=应拒答）+ 27 道课内题混跑各臂，正则判拒答（零 LLM 判分）：
  - 正确拒答率：越界题中命中拒答模板的比例
  - 误拒率：课内题中命中拒答模板的比例
  - 无证据强答率：越界题中未拒答的比例（=1-正确拒答率，分开报便于对照）

用法（backend 下，需 .env 凭据）：
    # 表A 两臂主表（27 课内题；fact 臂自动开精排，见 main 注释）
    python -P scripts/eval_rag/run_table_ab.py --course mycourse
    # 表B（20 越界 + 27 课内 × 指定臂）
    python -P scripts/eval_rag/run_table_ab.py --course mycourse --table b
    # A-3 top_k 敏感性扫描（只跑 fact 臂 × 各 k）
    python -P scripts/eval_rag/run_table_ab.py --course mycourse --sweep 3,5,10,20
    # 三张子表汇总成 markdown 主表（自动取 results/ 最新 JSON）
    python -P scripts/eval_rag/compose_table_a.py

产物：scripts/eval_rag/results/table_a_<ts>.json / table_b_<ts>.json /
      table_a_sweep_<ts>.json / table_a_composed_<ts>.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

load_dotenv(_BACKEND_ROOT / ".env", override=False)

from scripts.eval_rag import config  # noqa: E402

if config.LLM_API_KEY and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = config.LLM_API_KEY
os.environ.setdefault("LIGHTRAG__ENABLED", "true")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 表A 两臂（naive_rag/oracle_rag 已砍，理由见模块 docstring；实现仍在 rag_runner）
TABLE_A_MODES = ["no_rag", "fact"]

# 生产拒答话术（chat.yaml / tool_dispatch.py [系统指令]）：命中任一即算「含拒答话术」。
# 判「整条拒答」（vs 回答主体+尾注承认细节不知的部分拒答）须再满足长度门：
#  - 回答很短（≤120 字）且含话术 → 整条拒答（拒答话术通常就是一句话）
#  - 回答长但话术出现在开头 80 字内 → 主体即拒答（先拒后不再答）
#  - 其余为「部分拒答」：主体作答+尾部承认某细节资料未覆盖，不算整条拒答
_REFUSAL_PATTERNS = [
    "暂无法确认",
    "无法确认",
    "未检索到",
    "资料中未提及",
    "资料中未提供",
    "现有资料中暂未",
    "无法从现有资料",
    "建议你查阅",
    "建议查阅",
    "向.*老师.*咨询",
]


def is_refusal(answer: str) -> bool:
    if not answer:
        return False
    hits = [p for p in _REFUSAL_PATTERNS if re.search(p, answer)]
    if not hits:
        return False
    if len(answer) <= 120:
        return True
    # 主体拒答：开头 80 字内即出现话术（拒答句是回答第一句）
    return any(re.search(p, answer[:80]) for p in _REFUSAL_PATTERNS)


# ---------------------------------------------------------------------------
# 关键要点覆盖率（零 LLM）：gold 锚点（数字/百分号/带撇号编号/≥2字英文/≥3字中文词）
# 在回答中出现的比例
# ---------------------------------------------------------------------------
_KEYPOINT_RE = re.compile(
    r"\d+(?:\.\d+)?%?"          # 数字/百分数（22、80%、1-1′ 之外的纯数字）
    r"|[A-Za-z]{2,}(?:-\d+)?"   # ≥2 字母的英文型号（DG1022、FFT）
    r"|[一-鿿]{3,}"             # ≥3 字中文词
)


def extract_keypoints(text: str) -> list[str]:
    return _KEYPOINT_RE.findall(text or "")


def keypoint_coverage(answer: str, gold: str) -> float:
    kps = extract_keypoints(gold)
    if not kps:
        return 0.0
    hit = sum(1 for k in kps if k in (answer or ""))
    return hit / len(kps)


# ---------------------------------------------------------------------------
# 可追溯率：回答带来源标注（[证据N] / 来源: / 参考资料 引用样式的比例）
# ---------------------------------------------------------------------------
def has_citation(answer: str) -> bool:
    # 不能认「根据…资料」：拒答话术「根据现有资料暂无法确认」会被当成引用
    # （历史 bug：闭卷臂可追溯率虚高到 0.2222）。「参考资料」保留但排除其后紧跟
    # 「中未/中暂未」的近亲拒答句（「参考资料中未提及…」不算引用）。
    return bool(re.search(r"\[证据\d+\]|来源[：:]|参考资料(?!中[未暂])", answer or ""))


# ---------------------------------------------------------------------------
# 源片段命中率（零 LLM）：检索片段覆盖 gold 锚点 ≥0.5 即命中，按 rank 算 Top-1/Top-5
# ---------------------------------------------------------------------------
def source_hit_rates(
    qa_items: list[dict], results: list[dict], min_cover: float = 0.5
) -> dict:
    """对单臂结果算源片段 Top-1 / Top-5 命中率，返回 {"top1","top5","n"}（无映射返回 {}）。

    gold 锚点取自 oracle_gold.json 的源片段（make_oracle_gold 离线生成的题→源片段映射，
    25/27 题有映射）：复用其 extract_anchors 抽锚点串，某条检索片段覆盖 ≥min_cover 比例
    的锚点即算该题命中，取首个命中的 rank——Top-1 即 rank0 命中，Top-5 即前 5 条内命中
    （rank ≥5 的命中不计，k>5 的 sweep 里才可能出现）。只对 fact 臂有意义（闭卷臂
    contexts 恒空，恒 0，compose 时标「—」）。

    contexts 的「逐片段一条」形状由 rag_runner 保证（_split_context_string 按 --- 与
    [证据N] 头两道拆分，load_cache 对旧 blob 缓存亦归一），此处直接按序遍历。
    """
    from scripts.eval_rag.make_oracle_gold import extract_anchors
    from scripts.eval_rag.rag_runner import load_oracle_contexts

    gold = load_oracle_contexts()
    if not gold:
        return {}
    n = top1 = top5 = 0
    for item, res in zip(qa_items, results):
        chunk = gold.get(item["id"])
        if not chunk:
            continue
        anchors = extract_anchors(chunk)
        if not anchors:
            continue
        n += 1
        for rank, ctx in enumerate(res.get("contexts", [])[:5]):
            cov = sum(1 for a in anchors if a in ctx) / len(anchors)
            if cov >= min_cover:
                top5 += 1
                if rank == 0:
                    top1 += 1
                break
    if not n:
        return {}
    return {"top1": round(top1 / n, 4), "top5": round(top5 / n, 4), "n": n}


# ---------------------------------------------------------------------------
# RAGChecker 指标（custom_llm_api_func 绕开 checker max_tokens=10 硬编码：
# DeepSeek 推理模型 reasoning tokens 会先吃满 10 token，content 必空）
# ---------------------------------------------------------------------------
def _build_ragchecker() -> "object":
    from litellm import batch_completion

    # 判分模型：优先 EVAL_JUDGE_MODEL/DeepSeek；余额不足（Insufficient Balance）时
    # 自动退 FALLBACK__*（qwen-plus，DashScope）。判分是相对比较，模型档位在臂间一致即可。
    base_url = os.getenv("LLM__BASE_URL", "https://api.deepseek.com/v1")
    model = f"openai/{os.getenv('EVAL_JUDGE_MODEL', config.JUDGE_LLM_MODEL)}"
    api_key = os.getenv("LLM__API_KEY") or config.LLM_API_KEY

    def _probe() -> bool:
        try:
            import litellm

            litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                temperature=0, max_tokens=8, api_base=base_url,
            )
            return True
        except Exception as e:
            logger.warning("主判分模型不可用（%s），退 FALLBACK", str(e)[:120])
            return False

    if not _probe():
        fb_key = os.getenv("FALLBACK__API_KEY", "")
        fb_url = os.getenv("FALLBACK__BASE_URL", "")
        fb_model = os.getenv("FALLBACK__MODEL", "")
        if not (fb_key and fb_url and fb_model):
            raise RuntimeError("主判分模型不可用且无 FALLBACK 配置")
        base_url, model, api_key = fb_url, f"openai/{fb_model}", fb_key

    # litellm openai/ 前缀读 OPENAI_API_KEY；RAGChecker 的 openai_api_key 形参只在
    # 非空时写入 os.environ（传空串会被跳过）——这里显式设，双保险。
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    def custom_api(prompts: list[str]) -> list[str]:
        rs = batch_completion(
            model=model,
            messages=[[{"role": "user", "content": p}] for p in prompts],
            temperature=0, max_tokens=4096,
            api_base=base_url,
        )
        out = []
        for r in rs:
            c = r.choices[0].message.content if hasattr(r, "choices") else None
            out.append(c or "")
        return out

    from ragchecker import RAGChecker
    from ragchecker.container import RetrievedDoc, RAGResult, RAGResults

    rc = RAGChecker(
        extractor_name=model, checker_name=model,
        openai_api_key=api_key,
        batch_size_extractor=4, batch_size_checker=8,
        extractor_max_new_tokens=4096,
        joint_check=True, joint_check_num=5,
        custom_llm_api_func=custom_api,
    )
    return rc, RAGResult, RAGResults, RetrievedDoc


def ragchecker_metrics(
    qa_items: list[dict], results: list[dict], metrics: list[str] | None = None,
) -> dict[str, float]:
    """对单臂结果算 RAGChecker 指标（均值，×100 → 归一 0-1）。"""
    rc, RAGResult, RAGResults, RetrievedDoc = _build_ragchecker()
    metrics = metrics or ["precision", "recall", "f1"]
    rr = []
    for item, res in zip(qa_items, results):
        if not res.get("answer"):
            continue
        rr.append(RAGResult(
            query_id=item["id"], query=item["question"],
            gt_answer=item.get("ground_truth", ""),
            response=res["answer"],
            retrieved_context=[
                RetrievedDoc(doc_id=f"c{i}", text=c)
                for i, c in enumerate(res.get("contexts", []))
            ],
        ))
    if not rr:
        return {}
    out = rc.evaluate(RAGResults(results=rr), metrics=metrics)
    flat: dict[str, float] = {}
    for group in out.values():
        for k, v in group.items():
            flat[k] = round(float(v) / 100.0, 4)
    return flat


# ---------------------------------------------------------------------------
# 表A
# ---------------------------------------------------------------------------
def _zero_llm_metrics(qa_items: list[dict], res: list[dict]) -> dict:
    """零 LLM 程序指标（表A 各臂与 sweep 各 k 共用；闭卷臂 source_hit 恒 0，compose 标「—」）。"""
    return {
        "keypoint_coverage": round(sum(
            keypoint_coverage(r["answer"], q["ground_truth"])
            for q, r in zip(qa_items, res)
        ) / len(qa_items), 4),
        "citation_rate": round(
            sum(1 for r in res if has_citation(r["answer"])) / len(res), 4
        ),
        "source_hit": source_hit_rates(qa_items, res),
    }


async def run_table_a(args) -> dict:
    from scripts.eval_rag.rag_runner import run_all_modes

    qa_items = json.loads(config.DATASET_PATH.read_text("utf-8"))
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}  # 精确匹配，防子串误跳
    modes = [m for m in TABLE_A_MODES if m not in skip]

    all_results = await run_all_modes(
        args.course, qa_items, modes,
        no_cache=args.no_cache, top_k=args.top_k,
        retriever_backend="llamaindex_pg",
    )

    summary: dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "course": args.course,
        "modes": modes,
        "n": len(qa_items),
        "arms": {},
    }
    for mode in modes:
        res = all_results[mode]
        arm = _zero_llm_metrics(qa_items, res)
        # RAGChecker（LLM 判分，按需）
        if not args.no_llm_metrics:
            # precision/recall/f1 两臂都算——比对只依赖 GT 与回答，是闭卷臂最关键的证据
            mets = ["precision", "recall", "f1"]
            # 片段类指标只在带检索臂上算：闭卷臂 contexts 恒空，按定义 self_knowledge
            # 恒 1、hallucination 恒 =1-precision，表里写「—」不当数字比
            if mode != "no_rag":
                mets += ["self_knowledge", "hallucination", "faithfulness"]
            logger.info("RAGChecker %s: %s", mode, mets)
            try:
                arm.update(ragchecker_metrics(qa_items, res, mets))
            except Exception as e:
                logger.error("RAGChecker %s 失败: %s", mode, e)
        summary["arms"][mode] = arm

    # 延迟
    from scripts.eval_rag import stats
    summary["latency"] = {
        m: stats.compute_latency_distribution(all_results[m]) for m in modes
    }
    return summary


# ---------------------------------------------------------------------------
# A-3：fact 臂 top_k 敏感性扫描
# ---------------------------------------------------------------------------
async def run_topk_sweep(args) -> dict:
    from scripts.eval_rag.rag_runner import run_all_modes

    qa_items = json.loads(config.DATASET_PATH.read_text("utf-8"))
    ks = sorted({int(x) for x in args.sweep.split(",") if x.strip()})
    summary: dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "course": args.course,
        "mode": "fact",
        "n": len(qa_items),
        "ks": ks,
        "sweep": {},
    }
    for k in ks:
        logger.info("=== top_k=%d (fact) ===", k)
        all_results = await run_all_modes(
            args.course, qa_items, ["fact"],
            no_cache=args.no_cache, top_k=k,
            retriever_backend="llamaindex_pg",
        )
        res = all_results["fact"]
        entry = _zero_llm_metrics(qa_items, res)
        if not args.no_llm_metrics:
            # f1 需 answer2response+response2answer，hallucination 再加 retrieved2response，
            # 三条中间矩阵一次算齐；precision/recall 顺带产出，不额外花判分钱
            try:
                entry.update(ragchecker_metrics(
                    qa_items, res, ["precision", "recall", "f1", "hallucination"],
                ))
            except Exception as e:
                logger.error("RAGChecker sweep k=%d 失败: %s", k, e)
        summary["sweep"][str(k)] = entry
    return summary


# ---------------------------------------------------------------------------
# 表B
# ---------------------------------------------------------------------------
async def run_table_b(args) -> dict:
    from scripts.eval_rag.rag_runner import run_all_modes

    qa_items = json.loads(config.DATASET_PATH.read_text("utf-8"))
    ood_items = json.loads(
        (config.EVAL_DIR / "ood_dataset.json").read_text("utf-8")
    )
    modes = [m.strip() for m in args.b_modes.split(",")]
    # 拼错臂名不报错会静默按 fact 类检索臂跑（rag_runner 对未知 mode 走兜底分支），
    # 产出一条错误命名的臂进表B 结论——入口处 fail-fast
    known = {"no_rag", "naive_rag", "oracle_rag", "fact", "relationship"}
    unknown = [m for m in modes if m not in known]
    if unknown:
        raise SystemExit(f"--b-modes 含未知臂 {unknown}，可用：{sorted(known)}")

    all_results = await run_all_modes(
        args.course, qa_items + ood_items, modes,
        no_cache=args.no_cache, top_k=args.top_k,
        retriever_backend="llamaindex_pg",
        course_by_mode={"naive_rag": args.naive_course} if args.naive_course else {},
    )

    n_in, n_ood = len(qa_items), len(ood_items)
    summary: dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "course": args.course,
        "modes": modes,
        "n_in_scope": n_in, "n_ood": n_ood,
        "refusal_patterns": _REFUSAL_PATTERNS,
        "arms": {},
    }
    for mode in modes:
        res = all_results[mode]
        in_res, ood_res = res[:n_in], res[n_in:]
        # 拒答判定只算一遍：率与 qids 明细共用（is_refusal 是正则扫描，别对同一答案跑两次）
        ref_in = [is_refusal(r["answer"]) for r in in_res]
        ref_ood = [is_refusal(r["answer"]) for r in ood_res]
        summary["arms"][mode] = {
            # 越界题：拒答=对
            "correct_refusal_rate": round(sum(ref_ood) / n_ood, 4),
            # 课内题：拒答=错（误拒）
            "false_refusal_rate": round(sum(ref_in) / n_in, 4),
            # 越界题未拒答（无证据强答）
            "unrefused_ood_rate": round(1 - sum(ref_ood) / n_ood, 4),
            "refused_ood": sum(ref_ood), "refused_in_scope": sum(ref_in),
            # 误拒/未拒答明细，便于人工复核
            "false_refusal_qids": [
                q["id"] for q, ref in zip(qa_items, ref_in) if ref
            ],
            "unrefused_ood_qids": [
                q["id"] for q, ref in zip(ood_items, ref_ood) if not ref
            ],
        }
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description="表A 两臂主表 / A-3 top_k 扫描 / 表B 越界拒答评测")
    parser.add_argument("--table", choices=["a", "b"], default="a")
    parser.add_argument("--course", default="mycourse")
    parser.add_argument("--naive-course", default=None,
                        help="naive_rag 臂的 course_id（须预先按固定字数切块索引）")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip", default="", help="跳过的臂（逗号分隔，精确匹配，如 no_rag）")
    parser.add_argument("--no-llm-metrics", action="store_true",
                        help="表A 只跑零 LLM 程序指标（keypoint/citation），跳过 RAGChecker")
    parser.add_argument("--sweep", default=None, metavar="K1,K2,...",
                        help="A-3 top_k 敏感性扫描（如 3,5,10,20）：只跑 fact 臂 × 各 k，"
                             "报 f1/hallucination（缓存键自带 _k{K}，各 k 互不污染）")
    parser.add_argument("--b-modes", default="no_rag,fact",
                        help="表B 参与的臂（默认闭卷 vs 本系统）")
    args = parser.parse_args()

    # 本系统臂（fact）走精排：.env 默认 RERANK__ENABLED=false（生产关，eval 验证有收益
    # 后再上线），load_dotenv 已把 false 读进环境 → setdefault 无效，必须直接覆盖。
    # 只改本评测进程；放 main() 而非模块级，模块被 import（如测试）时不污染进程环境。
    # 须在首次 get_settings()（检索阶段才发生）之前执行，pydantic-settings 启动时快照。
    os.environ["RERANK__ENABLED"] = "true"
    # fail-fast：开关开了但 build_rerank_fn() 仍返回 None（缺 EMBEDDING__API_KEY）时，
    # fact 臂会静默退化成无精排——缓存照样写 _rr、表里照样标「含精排」，口径全错。
    # 精排 API 单次调用失败属于运行时降级（hybrid_retriever 按设计回退融合结果），不在此拦截。
    from core.rag.rerank import build_rerank_fn

    if build_rerank_fn() is None:
        raise SystemExit(
            "RERANK__ENABLED=true 但 build_rerank_fn() 返回 None（缺 EMBEDDING__API_KEY）："
            "fact 臂会退化为无精排，主表口径失效——先在 .env 配好 embedding key 再跑"
        )

    if args.sweep:
        summary = await run_topk_sweep(args)
        name = "table_a_sweep"
    elif args.table == "a":
        summary = await run_table_a(args)
        name = "table_a"
    else:
        summary = await run_table_b(args)
        name = "table_b"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = config.RESULTS_DIR / f"{name}_{ts}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    logger.info("Summary → %s", out)
    logger.info(json.dumps(summary.get("arms", {}), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
