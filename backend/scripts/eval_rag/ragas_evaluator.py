"""RAGAS Evaluator —— 对各模式的查询结果计算 RAGAS 指标。

支持 ragas >= 0.2（自动检测 API 版本）：
  - v0.4+: 使用 EvaluationDataset + SingleTurnSample，LangChain LLM 直接传入
  - v0.2.x: 使用 EvaluationDataset + evaluate()
  - v0.1.x: 使用 HuggingFace Dataset + evaluate()

指标：context_precision, context_recall, faithfulness, answer_relevancy
"""
from __future__ import annotations

import logging
import os
from typing import Any

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM / Embedding 配置
# ---------------------------------------------------------------------------
def _build_llm():
    """构建 LangChain ChatOpenAI 实例，直接传给 ragas evaluate()。"""
    from langchain_openai import ChatOpenAI

    # 确保 OPENAI_API_KEY 环境变量存在（LangChain 内部会检查）
    api_key = config.LLM_API_KEY
    if api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key

    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=api_key,
        base_url=config.LLM_BASE_URL,
        temperature=0,
    )


def _build_embeddings():
    """构建 LangChain OpenAIEmbeddings 实例。"""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=config.EMBED_MODEL,
        api_key=config.EMBED_API_KEY,
        base_url=config.EMBED_BASE_URL,
    )


# ---------------------------------------------------------------------------
# 获取指标对象
# ---------------------------------------------------------------------------
def _get_metrics(metric_names: list[str]) -> list[Any]:
    """根据名称列表获取 RAGAS 指标对象（已初始化的实例）。

    ragas 0.4 中 `from ragas.metrics import context_precision` 返回实例（有 deprecation 警告但可用）。
    """
    try:
        import ragas.metrics as m
        mapping = {
            "context_precision": getattr(m, "context_precision", None),
            "context_recall": getattr(m, "context_recall", None),
            "faithfulness": getattr(m, "faithfulness", None),
            "answer_relevancy": getattr(m, "answer_relevancy", None),
        }
        metrics = []
        for name in metric_names:
            obj = mapping.get(name)
            if obj is None:
                logger.warning("未知指标: %s，跳过", name)
                continue
            metrics.append(obj)
        if metrics:
            return metrics
    except Exception as e:
        logger.warning("旧版 ragas.metrics import 失败: %s", e)

    # 新版 fallback
    logger.info("尝试新版 ragas.metrics.collections import...")
    try:
        from ragas.metrics.collections.context_precision import ContextPrecision
        from ragas.metrics.collections.context_recall import ContextRecall
        from ragas.metrics.collections.faithfulness import Faithfulness
        from ragas.metrics.collections.answer_relevancy import AnswerRelevancy

        llm = _build_llm()
        mapping = {
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm),
        }
        metrics = []
        for name in metric_names:
            obj = mapping.get(name)
            if obj is None:
                logger.warning("未知指标: %s，跳过", name)
                continue
            metrics.append(obj)
        return metrics
    except Exception as e:
        logger.error("获取 metrics 失败: %s", e)
        return []


# ---------------------------------------------------------------------------
# 构建评测数据集
# ---------------------------------------------------------------------------
def _build_eval_dataset(
    qa_items: list[dict], mode_results: list[dict]
) -> Any:
    """构建 EvaluationDataset（ragas v0.2+）。"""
    try:
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
    except ImportError:
        return None


def _build_legacy_dataset(
    qa_items: list[dict], mode_results: list[dict]
) -> Any:
    """构建 HuggingFace Dataset（ragas v0.1.x 兼容）。"""
    from datasets import Dataset

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for item, result in zip(qa_items, mode_results):
        data["question"].append(item["question"])
        data["answer"].append(result.get("answer", ""))
        data["contexts"].append(result.get("contexts", []))
        data["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(data)


# ---------------------------------------------------------------------------
# 核心：解析评测结果
# ---------------------------------------------------------------------------
def _parse_eval_result(result: Any, metric_names: list[str]) -> dict[str, float]:
    """从 ragas evaluate() 返回值中提取指标分数。"""
    # ragas 0.4: 返回 EvaluationResult，有 .scores 属性
    if hasattr(result, "scores"):
        scores_list = result.scores
        if scores_list and isinstance(scores_list, list):
            avg: dict[str, float] = {}
            for m in metric_names:
                vals = [s.get(m, 0.0) for s in scores_list if isinstance(s, dict)]
                # 过滤 NaN
                import math
                vals = [v for v in vals if not math.isnan(v)]
                avg[m] = sum(vals) / len(vals) if vals else 0.0
            return avg

    # ragas 0.2: to_pandas()
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        skip_cols = {"user_input", "response", "reference", "retrieved_contexts",
                      "question", "answer", "contexts", "ground_truth"}
        avg = {}
        for col in df.columns:
            if col not in skip_cols:
                import math
                vals = [v for v in df[col] if not math.isnan(v)]
                avg[col] = sum(vals) / len(vals) if vals else 0.0
        return avg

    # fallback: 直接当 dict 处理
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if isinstance(v, (int, float))}

    return {}


# ---------------------------------------------------------------------------
# 核心评测函数
# ---------------------------------------------------------------------------
def evaluate_mode(
    qa_items: list[dict],
    mode_results: list[dict],
    metric_names: list[str],
) -> dict[str, float]:
    """对某模式的查询结果计算 RAGAS 指标，返回 {metric_name: avg_score}。"""
    metrics = _get_metrics(metric_names)
    if not metrics:
        logger.error("没有有效的指标，跳过评测")
        return {}

    llm = _build_llm()
    embeddings = _build_embeddings()

    # ---- 尝试新版 API (v0.2+) ----
    eval_ds = _build_eval_dataset(qa_items, mode_results)
    if eval_ds is not None:
        try:
            from ragas import evaluate as ragas_evaluate

            logger.info("使用 ragas v0.2+ API (EvaluationDataset)")
            result = ragas_evaluate(
                dataset=eval_ds,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
            )
            return _parse_eval_result(result, metric_names)
        except Exception as e:
            logger.warning("ragas v0.2+ API 失败: %s，尝试旧版 API", e)

    # ---- 旧版 API (v0.1.x) ----
    dataset = _build_legacy_dataset(qa_items, mode_results)
    try:
        from ragas import evaluate as ragas_evaluate

        logger.info("使用 ragas v0.1.x API (HuggingFace Dataset)")
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
        return _parse_eval_result(result, metric_names)
    except Exception as e:
        logger.error("ragas evaluate 失败: %s", e)
        return {}


def evaluate_mode_per_question(
    qa_items: list[dict],
    mode_results: list[dict],
    metric_names: list[str],
) -> list[dict[str, float]]:
    """对某模式逐条计算指标，返回 [{metric_name: score}]。

    用于 CSV 中每行的指标值。
    注意：逐条评测会显著增加 API 调用次数和成本。
    """
    per_question_results: list[dict[str, float]] = []

    for idx, (item, result) in enumerate(zip(qa_items, mode_results)):
        logger.info("逐条评测 %d/%d...", idx + 1, len(qa_items))
        scores = evaluate_mode([item], [result], metric_names)
        per_question_results.append(scores)

    return per_question_results
