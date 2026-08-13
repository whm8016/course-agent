"""课程主题图构建器（生产路径）：从讲义切块抽主题 + 前置边，灌 course_topic 表。

从 ``scripts/build_course_topics.py`` 下沉而来——生产代码（worker/API）不能 import scripts/
（脚本顶部有 import 期 env setdefault 副作用）。抽取逻辑的单一真相源在此，脚本的 from-chunks
分支与 probe 都改调本模块，避免脚本/生产两份抽取逻辑漂移。

核心入口：
- ``load_course_chunks(course_id)``：定位摄入审计 latest.json，解析章节前缀成 [{text,section,order_idx}]
- ``build_topic_graph(course_id, db, force)``：skip-if-exists / force 重建 / 强制算 embedding

embedding 是门控硬依赖（``proactive._match_topic`` 靠 embedding 最近邻，NULL 直接淘汰 → 门控全
unknown 不拼），故本模块不给关闭 embedding 的选项——能建索引（检索也靠 embedding）就一定能
建主题图；embed_model 解析失败即 fail-fast（由 worker deadletter 兜底）。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

JSON_DIR = Path("data/course_topics")


async def _aget_embed(embed_model, text: str) -> list[float] | None:
    if embed_model is None:
        return None
    try:
        return await embed_model._aget_text_embedding(text)
    except Exception as exc:
        logger.warning("[course_topics] embedding 失败（置 NULL）：%s", exc)
        return None


def export_json(course_id: str, topics: list[dict], edges: list[dict]) -> Path:
    """导出课程级 JSON artifact（表真相源的投影，供评测复现/教师核对/论文附录）。"""
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    out = JSON_DIR / f"course_{course_id}_topics.json"
    payload = {
        "course_id": course_id,
        "topics": topics,
        "edges": edges,
        "note": "表为真相源，本 JSON 为投影（评测复现/教师核对/论文附录用）",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[course_topics] JSON 导出 → %s（%d 主题 %d 边）", out, len(topics), len(edges))
    return out


# -------------------- from-chunks 离线抽取（有课程库数据时跑）--------------------

_EXTRACT_TOPICS_PROMPT = """你是课程结构分析助手。下面是一节实验讲义的文本（开头标注了章节）。
请抽出本节实际教学的主题，每个主题给一句话定义。返回严格 JSON（不要 markdown 围栏）：
{"topics": [{"label": "主题名", "definition": "一句话定义"}]}
规则：只抽本节确实教的内容；主题粒度对齐一个可考核的知识点（如「戴维南等效」而非「电路」）；
没有明确主题返回空数组；不编造。"""

_EXTRACT_PREREQ_PROMPT = """你是课程结构分析助手。判断 A 是否是 B 的直接前置（学 B 前应先掌握 A）。
A：{a}
B：{b}
返回严格 JSON：{{"is_prereq": true|false, "confidence": 0.0-1.0}}
规则：只判「A 是 B 的直接前置」这一方向；不确定或双向都依赖时 is_prereq=false（宁可漏连不乱连）。"""


async def _llm_json(llm, model: str, prompt: str) -> dict | None:
    try:
        # deepseek-v4-pro 等 reasoning 模型先吐 reasoning_content 再吐 content；
        # max_tokens 太小会被 reasoning 占满→content 空(finish=length)→json.loads 崩。
        resp = await llm.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=4096, stream=False,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("[course_topics] LLM 抽取失败：%s", exc)
        return None


def _pack_blocks(group: list[dict], max_chars_per_call: int) -> list[list[dict]]:
    """把一章的 chunks 按 chunk 边界打包成若干块，每块累计字符 <= max_chars_per_call。

    不切断单个 chunk 内部（保知识点完整）；单 chunk 本身超预算时单独成块（兜底不丢内容）。
    取代旧 ``[:1500]``/``[:4000]`` 双层硬截断——超长章节是「多调几次 LLM」而非「丢掉后半章」。
    纯函数、可单测。
    """
    blocks: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    for ck in group:
        n = len(ck["text"]) + 1  # +1 为 join 的换行
        if cur and cur_len + n > max_chars_per_call:
            blocks.append(cur)
            cur, cur_len = [], 0
        cur.append(ck)
        cur_len += n
    if cur:
        blocks.append(cur)
    return blocks


async def extract_topics_from_chunks(
    course_id: str, chunks: list[dict], llm, model: str, embed_model=None,
    max_chars_per_call: int = 5000,
) -> list[dict]:
    """从讲义 chunk 按「章节」聚合抽主题。chunk: {text, section, order_idx}。

    一章内按 chunk 边界分块（每块 <= max_chars_per_call 字符），每块独立问一次 LLM，
    跨块重复主题交给 merge_synonymous 去重。单 chunk 不截断、章节超长也不丢内容。
    max_chars_per_call 权衡：太大→reasoning 模型输出预算(4096)被推理占满→content 截断；
    太小→调用次数多、跨块重复多。默认 5000（probe 实测可调）。
    """
    by_section: dict[str, list[dict]] = {}
    for ck in chunks:
        sec = ck.get("section") or ck.get("source") or "未知章节"
        by_section.setdefault(sec, []).append(ck)

    topics: list[dict] = []
    order = 0
    for sec, group in by_section.items():
        for block in _pack_blocks(group, max_chars_per_call):
            text = "\n".join(ck["text"] for ck in block)
            data = await _llm_json(llm, model, f"章节：{sec}\n\n{text}\n\n{_EXTRACT_TOPICS_PROMPT}")
            for t in (data or {}).get("topics", []) if isinstance(data, dict) else []:
                label = str(t.get("label") or "").strip()
                if not label:
                    continue
                emb = await _aget_embed(embed_model, f"{label}：{t.get('definition','')}")
                topics.append({
                    "topic_id": "",  # 合并去重后定（label 归一 + embedding 相似）
                    "label": label,
                    "definition": str(t.get("definition") or "").strip(),
                    "source_section": sec,
                    "order_idx": order,
                    "embedding": emb,
                })
                order += 1
    return topics


def merge_synonymous(topics: list[dict], sim_threshold: float = 0.88) -> list[dict]:
    """合并同义主题：label 字符串归一 + embedding 余弦相似度去重。

    保留 order_idx 最小者，定义取较长者。纯函数、可单测。归一化与余弦复用
    course_topic_store（与门控读出侧同口径，避免漂移）。
    """
    from core.memory.course_topic_store import _norm_label, cosine

    merged: list[dict] = []
    seen: list[tuple[str, list[float]]] = []
    for t in sorted(topics, key=lambda x: x["order_idx"]):
        n = _norm_label(t["label"])
        dup = None
        for i, (sn, se) in enumerate(seen):
            if n == sn or (t["embedding"] and se and cosine(t["embedding"], se) >= sim_threshold):
                dup = i
                break
        if dup is not None:
            if len(t["definition"]) > len(merged[dup]["definition"]):
                merged[dup]["definition"] = t["definition"]
            continue
        seen.append((n, t["embedding"] or []))
        merged.append(t)
    for i, t in enumerate(merged):
        t["order_idx"] = i
        t["topic_id"] = _norm_label(t["label"])[:32] or f"topic{i}"
    return merged


async def extract_prereq_edges(
    course_id: str, topics: list[dict], llm, model: str
) -> list[dict]:
    """抽前置边：仅在 order_idx 在前 + 同章或相邻章候选对上判定（Goel 协议）。

    不做 O(n²) 全配对——讲义顺序在前 + 章节邻近才作候选，含糊/双向丢弃。
    """
    edges: list[dict] = []
    for i, dst in enumerate(topics):
        for src in topics[:i]:  # 只看顺序在前的
            # 章节邻近约束：同章或相邻（order 差小）
            if abs(src["order_idx"] - dst["order_idx"]) > 3:
                continue
            data = await _llm_json(
                llm, model,
                _EXTRACT_PREREQ_PROMPT.format(a=src["label"], b=dst["label"]),
            )
            if isinstance(data, dict) and data.get("is_prereq") and (data.get("confidence") or 0) >= 0.6:
                edges.append({
                    "src": src["topic_id"], "dst": dst["topic_id"],
                    "relation": "prerequisite", "confidence": float(data.get("confidence")),
                })
    return edges


def load_course_chunks(course_id: str) -> list[dict]:
    """定位该课程的摄入切块审计 latest.json，解析成 [{text, section, order_idx}]。

    优先 pgvector 后端落盘（``settings.paths.ingest_chunks_dir/course_{id}/``），回退 LightRAG
    （``lightrag_store/course_{id}/ingest_chunks/``，复用 ``ingestion._lightrag_ingest_chunks_dir``
    避免路径逻辑两份）。审计 JSON 的 ``chunks`` 是**纯字符串列表**，每条开头带
    ``【章节: ...】`` 前缀（``ingestion._build_source_prefix`` 注入）；逐条 ``parse_source_prefix``
    拆出 section（按章聚合抽主题用）与纯正文 text。

    两后端落盘都受 settings 开关门控（``chunking.save_pg_ingest_chunks`` / ``lightrag.save_ingest_chunks``）；
    关闭或未索引过 → 文件缺失 → 抛 FileNotFoundError（提示先建索引）。
    """
    from core.rag.ingestion import _lightrag_ingest_chunks_dir
    from core.rag.source_utils import parse_source_prefix
    from settings import get_settings

    s = get_settings()
    candidates = [
        Path(s.paths.ingest_chunks_dir) / f"course_{course_id}" / "latest.json",
        _lightrag_ingest_chunks_dir(course_id) / "latest.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"课程 {course_id} 没有摄入切块审计 JSON（latest.json）。请先对该课程建立知识库索引"
            f"（POST /courses/{course_id}/index），并确认未关闭 ingest chunks 落盘开关。"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks = raw.get("chunks") if isinstance(raw, dict) else raw
    if not chunks:
        raise ValueError(
            f"课程 {course_id} 的审计 JSON 无 chunk（{path}）；索引可能空跑，请检查上传文件。"
        )
    parsed: list[dict] = []
    for i, ck in enumerate(chunks):
        section, body = parse_source_prefix(ck)
        parsed.append({"text": body, "section": section or "未知章节", "order_idx": i})
    logger.info("[course_topics] 加载切块审计 course=%s path=%s chunks=%d", course_id, path, len(parsed))
    return parsed


async def build_topic_graph(course_id: str, db, *, force: bool = False) -> dict:
    """从课程讲义切块抽主题图并灌 ``course_topic`` / ``course_topic_edge``。

    - skip-if-exists：表已有该课程主题且非 force → 返回 already_exists，不烧 LLM。
    - force：先删旧边与旧主题（避开 ``uq_course_topic`` 唯一约束）再重建。
    - embedding 强制算（``get_embed_model``）——门控靠 embedding 做问句最近邻，NULL 直接淘汰。
      embed_model 解析失败即 raise（fail-fast，由 worker deadletter 兜底）。
    """
    from sqlalchemy import delete, func, select

    from core.db.database import CourseTopic, CourseTopicEdge

    existing = (
        await db.execute(
            select(func.count()).select_from(CourseTopic).where(CourseTopic.course_id == course_id)
        )
    ).scalar_one()
    if existing and not force:
        logger.info("[course_topics] course=%s 已有 %d 主题，skip（force=True 重建）", course_id, existing)
        return {"status": "already_exists", "course_id": course_id, "topics": int(existing)}
    if force:
        await db.execute(delete(CourseTopicEdge).where(CourseTopicEdge.course_id == course_id))
        await db.execute(delete(CourseTopic).where(CourseTopic.course_id == course_id))
        await db.flush()

    chunks = load_course_chunks(course_id)

    try:
        from core.rag.llamaindex.pg_store import get_embed_model

        embed_model = get_embed_model()
    except Exception as exc:
        # 门控硬依赖：embed_model 不可用 = 产物对门控无用（全 unknown）。fail-fast 让 worker 兜底。
        raise RuntimeError(f"embed_model 不可用，无法构建有效主题图（门控依赖 embedding）：{exc}") from exc

    from core.llm.llm import client as llm
    from settings import get_settings
    s = get_settings()
    model = s.course_topic.extract_model or s.llm.text_model
    logger.info(
        "[course_topics] build_topic_graph course=%s model=%s chunks=%d force=%s",
        course_id, model, len(chunks), force,
    )

    topics = await extract_topics_from_chunks(course_id, chunks, llm, model, embed_model)
    topics = merge_synonymous(topics)
    edges = await extract_prereq_edges(course_id, topics, llm, model)

    for t in topics:
        db.add(CourseTopic(
            course_id=course_id, topic_id=t["topic_id"], label=t["label"],
            definition=t["definition"], source_section=t["source_section"],
            order_idx=t["order_idx"], embedding=t["embedding"],
        ))
    for e in edges:
        db.add(CourseTopicEdge(
            course_id=course_id, src_topic_id=e["src"], dst_topic_id=e["dst"],
            relation=e["relation"], confidence=e["confidence"],
        ))
    await db.commit()
    export_json(course_id, topics, edges)
    logger.info("[course_topics] build_topic_graph done course=%s %d主题 %d边", course_id, len(topics), len(edges))
    return {"status": "built", "course_id": course_id, "topics": len(topics), "edges": len(edges)}
