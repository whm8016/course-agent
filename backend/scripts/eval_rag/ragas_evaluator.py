"""RAGAS Evaluator —— ragas 0.4.3 evaluate() 适配。

LLM：llm_factory(model, client=OpenAI(...)) → InstructorLLM。阅卷走 DeepSeek（_build_openai_client）。
Embedding：_DimensionalOpenAIEmbeddings 固定注入 dimensions（修 answer_relevancy=0，Bug 2），
           走千问独立 client（_build_embed_client），与 LLM 不同 provider。
Metrics：ragas 0.4.3 的 evaluate() 用 isinstance(m, Metric) 校验。collections 版类基类是
         BaseMetric（不是 Metric），传给 evaluate 会被拒绝。故标准指标用 ragas.metrics
         模块级单例（LLM/embedding 由 evaluate(llm=, embeddings=) 注入），factual_correctness/
         noise_sensitivity 标准单例已移除、用 ragas.metrics._xxx 旧实现类（仍是 Metric），
         AspectCritic 自定义 definition 构造。

注意：dataset_generator 合成走 ragas 原生 TestsetGenerator(llm=, embedding_model=)（async embedding
用 AsyncOpenAI），与阅卷 evaluate 共用 _build_ragas_llm/_build_ragas_embeddings，不走 langchain。
"""
from __future__ import annotations

import logging
from typing import Any

try:
    from ragas.embeddings import OpenAIEmbeddings
except ImportError:  # ragas 未装时 import evaluator 仍可用（调用 evaluate 时才报错）
    OpenAIEmbeddings = object  # type: ignore[assignment,misc]

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 固定 dimensions 的 Embedding（Bug 2 修复）
# ---------------------------------------------------------------------------
class _DimensionalOpenAIEmbeddings(OpenAIEmbeddings):
    """固定注入 dimensions 的 OpenAIEmbeddings（DashScope text-embedding-v3 需要）。

    ragas OpenAIEmbeddings 透传 **kwargs 到 client.embeddings.create，但 metric 调用
    embed 时不传 dimensions。DashScope text-embedding-v3 在不传 dimensions 时（尤其
    langchain 旧路径）可能返回异常向量 → answer_relevancy 内部 cosine 相似度全 0。
    在 4 个 embed 方法里 setdefault dimensions，确保显式传。
    """

    def __init__(self, client: Any, model: str, dimensions: int, cache: Any = None):
        super().__init__(client=client, model=model, cache=cache)
        self._dimensions = dimensions

    def embed_text(self, text: str, **kwargs: Any):
        kwargs.setdefault("dimensions", self._dimensions)
        return super().embed_text(text, **kwargs)

    async def aembed_text(self, text: str, **kwargs: Any):
        kwargs.setdefault("dimensions", self._dimensions)
        return await super().aembed_text(text, **kwargs)

    def embed_texts(self, texts: list[str], **kwargs: Any):
        kwargs.setdefault("dimensions", self._dimensions)
        return super().embed_texts(texts, **kwargs)

    async def aembed_texts(self, texts: list[str], **kwargs: Any):
        kwargs.setdefault("dimensions", self._dimensions)
        return await super().aembed_texts(texts, **kwargs)

    # langchain 接口适配：answer_relevancy 等标准指标调 embed_query/embed_documents，
    # 而 ragas OpenAIEmbeddings 只有 embed_text/embed_texts。二者返回类型一致，直接转发。
    def embed_query(self, text: str, **kwargs: Any):
        return self.embed_text(text, **kwargs)

    def embed_documents(self, texts: list[str], **kwargs: Any):
        return self.embed_texts(texts, **kwargs)

    async def aembed_query(self, text: str, **kwargs: Any):
        return await self.aembed_text(text, **kwargs)

    async def aembed_documents(self, texts: list[str], **kwargs: Any):
        return await self.aembed_texts(texts, **kwargs)


# ---------------------------------------------------------------------------
# OpenAI client（DashScope 兼容）—— LLM 与 Embedding 共用
# ---------------------------------------------------------------------------
def _build_openai_client():
    """构造 LLM 专用 OpenAI 兼容 client（DeepSeek，供阅卷/合成 LLM 用）。"""
    from openai import OpenAI

    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def _build_embed_client():
    """构造 embedding 专用 OpenAI 兼容 client（千问 DashScope，与 LLM 不同 provider）。

    不能复用 _build_openai_client：LLM 走 DeepSeek base_url，embedding 走千问，
    混用会把 embedding 请求打到 DeepSeek 报错（混合 provider 下的隐藏坑）。
    """
    from openai import OpenAI

    return OpenAI(api_key=config.EMBED_API_KEY, base_url=config.EMBED_BASE_URL)


def _build_embed_async_client():
    """async embedding client（千问 DashScope），供合成路径的 async embedding。

    ragas OpenAIEmbeddings 的 aembed_* 要求 async client（同步 client 会 raise
    "Cannot use aembed_texts() with a synchronous client"）。合成 generate_with_chunks
    走 async 路径，故单独用 AsyncOpenAI；阅卷 evaluate 用同步 _build_embed_client 不变。
    """
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=config.EMBED_API_KEY, base_url=config.EMBED_BASE_URL)


# ---------------------------------------------------------------------------
# RAGAS InstructorLLM / Embeddings（阅卷 evaluate + 合成 generate 共用）
# ---------------------------------------------------------------------------
def _build_ragas_llm(model: str | None = None, *, max_tokens: int | None = None):
    """llm_factory → InstructorBaseRagasLLM（collections metrics 只接受这种）。

    model 默认 JUDGE_LLM_MODEL（阅卷用强推理 deepseek-v4-pro）；合成出题可传
    GEN_LLM_MODEL（flash，省成本）。
    max_tokens 默认 JUDGE_LLM_MAX_TOKENS（4096）；合成路径传 GEN_LLM_MAX_TOKENS（8192）。
    必须显式传：ragas 默认 max_tokens=1024，NERExtractor / claims 分解等结构化输出会超限被截断
    → instructor IncompleteOutputException（合成 NER 步骤直接拉崩 generate_with_chunks）。
    """
    from ragas.llms import llm_factory

    return llm_factory(
        model=model or config.JUDGE_LLM_MODEL,
        client=_build_openai_client(),
        max_tokens=max_tokens or config.JUDGE_LLM_MAX_TOKENS,
    )


def _build_ragas_embeddings(*, async_client: bool = False):
    """RAGAS embeddings（固定 dimensions，走千问 embed client）。

    async_client=True 用 AsyncOpenAI（供合成 generate_with_chunks 的 async 路径）；
    默认 False 用同步 client（供阅卷 evaluate，sync embed_text）。
    """
    client = _build_embed_async_client() if async_client else _build_embed_client()
    return _DimensionalOpenAIEmbeddings(
        client=client,
        model=config.EMBED_MODEL,
        dimensions=config.EMBED_DIMENSIONS,
    )


# ---------------------------------------------------------------------------
# 获取 collections 指标对象
# ---------------------------------------------------------------------------
# AspectCritic 自定义领域指标的评判标准（definition 注入 instruction）
_TEACHING_ACCURACY_DEFINITION = (
    "回答在教学上准确、无误导：专业术语使用正确，公式/定理引用准确，"
    "不存在可能误导学生的表述（概念混淆、量纲错误、符号错误等）。"
)
_SAFETY_DEFINITION = (
    "回答不包含可能导致实验安全事故的错误指导，例如短路保护提示缺失、"
    "电压极性错误、测量量程选择不当、危险操作未加警示。"
)


def _get_metrics(metric_names: list[str], llm: Any, embeddings: Any) -> list[Any]:
    """根据名称列表返回 ragas 0.4.3 指标（evaluate 要求 isinstance(m, Metric)）。

    ragas 0.4.3 的 evaluate() 用 isinstance(m, Metric) 校验。collections 版类
    （Faithfulness(llm=...) 等）基类是 BaseMetric 而非 Metric，会被拒绝。故：
      - 标准指标用 ragas.metrics 模块级单例（faithfulness/context_precision/context_recall/
        answer_relevancy），LLM 与 embedding 由 evaluate(llm=, embeddings=) 统一注入；
      - factual_correctness / noise_sensitivity 标准单例已移除，用 ragas.metrics._xxx
        旧实现类构造（仍是 Metric，自带 llm）；
      - AspectCritic 需自定义 definition，构造实例（是 Metric）。

    支持指标：
      - context_precision / context_recall：检索质量（tier1）
      - faithfulness：生成防幻觉（tier2）
      - factual_correctness：事实对齐度（tier2，claim 分解 + NLI，mode=f1）
      - noise_sensitivity：无关文档误答敏感度（tier3，mode=irrelevant）
      - answer_relevancy：回答相关性（需 embedding）
      - teaching_accuracy / safety：AspectCritic 领域定制（binary 1/0）
    """
    from ragas.metrics import (
        AspectCritic,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.metrics._factual_correctness import FactualCorrectness
    from ragas.metrics._noise_sensitivity import NoiseSensitivity

    factory: dict[str, Any] = {
        # 标准单例：LLM/embedding 由 evaluate(llm=, embeddings=) 注入，不在此构造
        "context_precision": lambda: context_precision,
        "context_recall": lambda: context_recall,
        "faithfulness": lambda: faithfulness,
        "answer_relevancy": lambda: answer_relevancy,
        # 标准单例已移除的进阶指标：用旧实现类构造（仍是 Metric，自带 llm）
        "factual_correctness": lambda: FactualCorrectness(llm=llm, mode="f1"),
        "noise_sensitivity": lambda: NoiseSensitivity(llm=llm, mode="irrelevant"),
        # AspectCritic 领域定制（需 definition）
        "teaching_accuracy": lambda: AspectCritic(
            name="teaching_accuracy",
            definition=_TEACHING_ACCURACY_DEFINITION,
            llm=llm,
        ),
        "safety": lambda: AspectCritic(
            name="safety", definition=_SAFETY_DEFINITION, llm=llm
        ),
    }
    metrics: list[Any] = []
    for name in metric_names:
        f = factory.get(name)
        if f is None:
            logger.warning("未知指标: %s，跳过", name)
            continue
        try:
            metrics.append(f())
        except Exception as e:
            logger.error("构造指标 %s 失败: %s", name, e)
    return metrics


# ---------------------------------------------------------------------------
# 构建评测数据集
# ---------------------------------------------------------------------------
def _build_eval_dataset(qa_items: list[dict], mode_results: list[dict]) -> Any:
    """构建 EvaluationDataset（ragas v0.2+ SingleTurnSample）。"""
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = []
    for item, result in zip(qa_items, mode_results):
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=result.get("answer", ""),
                retrieved_contexts=result.get("contexts", []),
                reference=item["ground_truth"],
            )
        )
    return EvaluationDataset(samples=samples)


# ---------------------------------------------------------------------------
# 解析评测结果：从单次 evaluate 提取 (整体均值, 逐条分数)
# ---------------------------------------------------------------------------
def _safe_float(v: Any) -> float:
    """把 ragas 分数转 float，NaN/None/异常 → 0.0（评测需可比的数值，不容 NaN）。"""
    import math
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _is_invalid_score(v: Any) -> bool:
    """ragas 分数是否是评测器异常（NaN/None/非数值）。

    ragas Executor 把判分异常静默转成 np.nan，与"真判 0"在最终分数上无法区分——
    这里用原始值是否 NaN/None/非数值 来识别"判崩了"，供均值剔除与排除清单使用。
    """
    import math
    try:
        f = float(v)
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def _avg_from_per_question(
    per_q: list[dict[str, float]], metric_names: list[str],
    invalid: dict[str, list[int]] | None = None,
) -> dict[str, float]:
    """从逐条分数算各指标均值。

    invalid 给出每个指标"判崩了（NaN）"的样本索引，这些不计入均值（分母减一），
    避免 RAGAS 评测器异常把假性 0 分灌进均值、系统性拉低 faithfulness/recall。
    """
    if not per_q:
        return {m: 0.0 for m in metric_names}
    invalid = invalid or {m: [] for m in metric_names}
    out = {}
    for m in metric_names:
        bad = set(invalid.get(m, []))
        valid = [row.get(m, 0.0) for i, row in enumerate(per_q) if i not in bad]
        out[m] = sum(valid) / len(valid) if valid else 0.0
    return out


def _extract_scores(
    result: Any, metric_names: list[str]
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, list[int]]]:
    """从 ragas evaluate() 返回值提取 (整体均值, 逐条分数列表, 各指标判崩样本索引)。

    逐条分数直接来自单次 evaluate 的结果（每条样本一行/一项），无需逐条重跑——
    这是 Bug 3 的修复核心：旧实现要么把整体均值复制 N 遍充作逐条，要么逐条重跑 N 次 API。
    metric 列名与 metric_names 一致（collections 指标的 name 属性决定列名）。

    per_q 结构不变（{metric: float}），判崩（NaN）样本填 0.0 保持 CSV 可用，不波及下游；
    判崩题号单独放进 invalid（{metric: [idx...]}），供均值剔除与排除清单。
    """

    def _build(scores_iter):
        per_q = []
        invalid = {m: [] for m in metric_names}
        for i, s in enumerate(scores_iter):
            # 兼容 dict 与 pandas Series（二者都有 .get）
            if not hasattr(s, "get"):
                continue
            row = {}
            for m in metric_names:
                raw = s.get(m)
                if _is_invalid_score(raw):
                    invalid[m].append(i)
                    row[m] = 0.0
                else:
                    row[m] = float(raw)
            per_q.append(row)
        return per_q, invalid

    empty_invalid = {m: [] for m in metric_names}
    # ragas 0.4: EvaluationResult.scores 是 list[dict]，每条样本一项
    scores_list = getattr(result, "scores", None)
    if isinstance(scores_list, list) and scores_list:
        per_q, invalid = _build(scores_list)
        return _avg_from_per_question(per_q, metric_names, invalid), per_q, invalid

    # to_pandas fallback（ragas 0.2/0.4 通用）
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        per_q, invalid = _build(row for _, row in df.iterrows())
        return _avg_from_per_question(per_q, metric_names, invalid), per_q, invalid

    # fallback: dict 形式（无逐条信息）
    if isinstance(result, dict):
        return ({k: _safe_float(v) for k, v in result.items()}, [], empty_invalid)
    return {}, [], empty_invalid


# ---------------------------------------------------------------------------
# 核心评测函数
# ---------------------------------------------------------------------------
def evaluate_mode(
    qa_items: list[dict],
    mode_results: list[dict],
    metric_names: list[str],
) -> dict[str, Any]:
    """对某模式的查询结果计算 RAGAS 指标。

    返回 {"avg": {metric: 均值}, "per_question": [{metric: 分数}, ...]}。
    逐条分数来自单次 evaluate 的结果（每条样本一项），无需逐条重跑。
    """
    llm = _build_ragas_llm()
    embeddings = _build_ragas_embeddings()
    metrics = _get_metrics(metric_names, llm=llm, embeddings=embeddings)
    if not metrics:
        logger.error("没有有效的指标，跳过评测")
        return {"avg": {}, "per_question": []}

    eval_ds = _build_eval_dataset(qa_items, mode_results)
    try:
        from ragas import evaluate as ragas_evaluate

        logger.info("使用 ragas 0.4 collections API (InstructorLLM + EvaluationDataset)")
        result = ragas_evaluate(
            dataset=eval_ds,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
        )
        avg, per_q, invalid = _extract_scores(result, metric_names)
        # Cost 估算依据：ragas EvaluationResult.total_tokens（本轮该模式总消耗）
        # ragas 0.4.3 的 total_tokens 是方法（非属性），需兼容调用
        _tt = getattr(result, "total_tokens", 0)
        try:
            _tt = _tt() if callable(_tt) else _tt
        except Exception:
            _tt = 0
        total_tokens = int(_tt or 0)
        return {"avg": avg, "per_question": per_q, "total_tokens": total_tokens, "invalid": invalid}
    except Exception as e:
        logger.error("ragas evaluate 失败: %s", e)
        return {"avg": {}, "per_question": [], "invalid": {}}
