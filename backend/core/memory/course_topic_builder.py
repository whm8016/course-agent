"""课程主题图构建器（生产路径）：从讲义切块同抽「主题 + 课文锚定先修边」，灌 course_topic 表。

从 ``scripts/build_course_topics.py`` 下沉而来——生产代码（worker/API）不能 import scripts/
（脚本顶部有 import 期 env setdefault 副作用）。抽取逻辑的单一真相源在此，脚本的 from-chunks
分支与 probe 都改调本模块，避免脚本/生产两份抽取逻辑漂移。

抽取主循环对齐 LightRAG ``operate.extract_entities`` / KGGen ``steps/_2_get_relations``：
- 原文随行：主题与边从同一块讲义文本一次抽出，边带原文引句（evidence）
- 闭集端点（KGGen ``parse_relations_response`` 口径）：边端点 = 本块新主题 ∪ 前面块已抽出的
  主题——旧点可当端点但不重复成节点，跨章「在 X 的基础上」这类最有证据的边由此而来
- 抽完再去重：merge_synonymous 归并后经别名表把边端点重映射到规范 topic_id
不做「语义 top-k 候选 + 两两判定」——已废弃（凭标签猜边无证据根基）。

核心入口：
- ``load_course_chunks(course_id)``：定位摄入审计 latest.json，解析章节前缀成 [{text,section,order_idx}]
- ``build_topic_graph(course_id, db, force)``：skip-if-exists / force 重建 / 强制算 embedding

embedding 是门控硬依赖（``proactive._match_topic`` 靠 embedding 最近邻，NULL 直接淘汰 → 门控全
unknown 不拼），故本模块不给关闭 embedding 的选项——能建索引（检索也靠 embedding）就一定能
建主题图；embed_model 解析失败即 fail-fast（由 worker deadletter 兜底）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
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


def export_json(
    course_id: str, topics: list[dict], edges: list[dict],
    failed_sections: list[str] | None = None, stats: dict | None = None,
) -> Path:
    """导出课程级 JSON artifact（表真相源的投影，供评测复现/教师核对/论文附录）。

    failed_sections / stats 为本次构建的抽取失败章节与边统计——GET payload 经
    ``read_failed_sections``（本模块，mtime 缓存）容错读回展示给教师（不加表列：
    展示字段不值一次迁移）。旧调用方（scripts/build_course_topics.py）不传则落空默认。
    """
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    out = JSON_DIR / f"course_{course_id}_topics.json"
    payload = {
        "course_id": course_id,
        "topics": topics,
        "edges": edges,
        "failed_sections": failed_sections or [],
        "stats": stats or {},
        "note": "表为真相源，本 JSON 为投影（评测复现/教师核对/论文附录用）",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[course_topics] JSON 导出 → %s（%d 主题 %d 边）", out, len(topics), len(edges))
    return out


# 读侧缓存：GET 主题图每次都要 failed_sections，但 artifact 含 embedding 可达数百 KB——
# (path, mtime_ns) 缓存使未变化的构建零重读；重建后 mtime 变化自动失效。
_failed_sections_cache: dict[str, tuple[int, list[str]]] = {}


def read_failed_sections(course_id: str) -> list[str]:
    """读回构建时落盘的抽取失败章节（与 export_json 同模块——文件名契约单一真相源）。

    文件缺失 / 解析失败 / 字段缺失一律返回 []：纯展示字段，不因 IO 问题打断 GET。
    """
    path = JSON_DIR / f"course_{course_id}_topics.json"
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return []
    cached = _failed_sections_cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        failed = data.get("failed_sections") if isinstance(data, dict) else None
        result = [str(s) for s in failed] if isinstance(failed, list) else []
    except Exception:
        result = []
    _failed_sections_cache[str(path)] = (mtime, result)
    return result


# -------------------- from-chunks 离线抽取（有课程库数据时跑）--------------------

_EXTRACT_BLOCK_PROMPT = """---Role---
你是课程知识图谱专家，负责从课程讲义文本中抽取可考核的知识点及其先修关系。

---Goal---
抽出本块讲义中新教的知识点（KC），以及文本中明示的先修依赖。

---Instructions---
【知识点 topics】
1. 粒度锚定：一个知识点 = 可单独考核的最小知识单元。
   - 正例：「戴维南等效」「三端变阻器的连接方式」「一阶RC电路的零输入响应」
   - 反例：「电路」「实验」「基本理论」（过宽不可考核，禁止输出这类泛化词）
2. label 必须逐字使用讲义中出现的原词——不翻译、不改写、不缩写、不自行概括。
3. 共指消解：讲义中「本章」「该定理」「它」等指代，在输出里解析为所指知识点的完整名称。
4. topics 只收本块**新教**的知识点；「已有知识点清单」里的旧点不要再输出进 topics；
   没有新知识点返回空数组；宁少勿凑数，不编造。

【先修边 edges】
5. 只抽文本中**明示**的先修依赖——「在X的基础上」「复习/回顾X」「由X推导」「需先掌握X」
   「利用X求Y」这类讲义亲口写出的依赖。凭常识猜测、文本没写的不要输出。
6. 每条边 src=前置知识点，dst=依赖它的知识点；strength ∈ [0,1] 表示依赖的明确程度；
   evidence 逐字摘抄块原文中体现该依赖的句子（只抄一句，不改写、不拼接）。
7. 边的两端可以是本块新教的知识点，也可以是「已有知识点清单」中的旧点——如第 5 章写
   「在元件伏安特性的基础上测等效内阻」，src 用清单里的「元件伏安特性」。两端都不在
   清单、也不在本块新点里的，不要输出该边。
8. 不确定是不是先修关系的不要输出（宁漏勿错）。

---已有知识点清单---
{known_list}

---Output---
输出严格 JSON（无 markdown 围栏、无任何其他文字）：
{{"topics": [{{"label": "知识点名", "definition": "一句话定义"}}], "edges": [{{"src": "前置点", "dst": "后继点", "strength": 0.8, "evidence": "原文句"}}]}}"""

# gleaning 补抽（LightRAG extract_entities 固定单次模式）：首轮解析成功后追加「只补遗漏」
# 一轮。单消息调用没有第一轮的对话历史，规则须自包含重述（不能写「规则与第一轮相同」），
# 否则模型只凭 Goal 一句话自由发挥。带原文与第一轮结果，避免凭空重复输出。
_GLEANING_PROMPT = """---Role---
你是课程知识图谱专家，正在复查同一块讲义的知识点+先修边抽取结果。

---Goal---
只补抽第一轮**遗漏**的知识点与先修边，不重复、不改写第一轮已有条目；没有遗漏返回空。

---Instructions---
【知识点 topics】
- 只收本块讲义中**新教**的可考核知识点（可单独考核的最小知识单元）；label 逐字用讲义原词；
  「已有知识点清单」里的旧点不进 topics；泛化词（「电路」「实验」）不收。
【先修边 edges】
- 只抽文本**明示**的先修依赖（「在X的基础上」「复习X」「由X推导」等写法）；src=前置，
  dst=后继；strength ∈ [0,1]；evidence 逐字摘抄块原文一句；两端须在本块新点或清单旧点内。
- 没有遗漏就返回空：{{"topics": [], "edges": []}}

---第一轮结果---
{first_round}

---已有知识点清单---
{known_list}

---Output---
输出严格 JSON（无围栏）：
{{"topics": [{{"label": "知识点名", "definition": "一句话定义"}}], "edges": [{{"src": "前置点", "dst": "后继点", "strength": 0.8, "evidence": "原文句"}}]}}"""

# 库函数签名默认（直接调用方/probe 不传时用）；部署级默认在 settings.course_topic.gleaning
_DEFAULT_GLEANING = 1
# strength 缺失/不可解析时的兜底置信（GraphRAG _process_result 对 weight 解析失败兜底同款）
_DEFAULT_STRENGTH = 0.75
# embedding 并发上限（照 consolidation._JUDGE_CONCURRENCY 模式——全仓 fan-out 都有 cap）。
# 裸 gather 在大课程数百主题时一次打满嵌入端点，429 风暴被 _aget_embed 吞成 None →
# 静默产出 embedding 全 NULL 的残废图（门控永远匹配不上），限流才能看见。
_EMBED_CONCURRENCY = 8


async def _llm_json(
    llm, model: str, prompt: str, track: dict[str, str] | None = None
) -> dict | None:
    """LLM → JSON：传输层走 ``with_retry_and_circuit``（指数退避+熔断），解析层自重试一次。

    网络抖动/限流由可靠性层兜（可重试异常自动退避重试一整波）；传输整波耗尽/熔断打开
    即返回 None（不再发第二波——外层重试只会翻倍退避时长，熔断打开时更是必失败）；
    200-但非 JSON 由外层再试一次。返回 None 后上层记 failed_section 或丢该轮 gleaning，
    不向上抛。专属熔断器名 ``topic_build``：后台课程级构建失败不烧对话主链路的
    default 熔断器（vision_describe 同款思路）。
    """
    from core.llm.json_extract import extract_json_from_llm
    from core.llm.reliability import (
        CircuitOpenError,
        LLMRetryError,
        RetryConfig,
        get_llm_circuit_breaker,
        with_retry_and_circuit,
    )

    breaker = get_llm_circuit_breaker("topic_build")
    retry_config = RetryConfig()
    for attempt in (1, 2):  # 200-但非 JSON：重试一次
        try:
            # deepseek-v4-pro 等 reasoning 模型先吐 reasoning_content 再吐 content；
            # max_tokens 太小会被 reasoning 占满→content 空(finish=length)。
            resp = await with_retry_and_circuit(
                llm.chat.completions.create,
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4096, stream=False,
                retry_config=retry_config, circuit_breaker=breaker,
            )
        except (CircuitOpenError, LLMRetryError) as exc:
            logger.warning("[course_topics] LLM 传输失败（可靠性层整波耗尽，放弃本块）：%s", exc)
            return None
        try:
            if track is not None:
                # 用量埋点：主题图抽取是 ARQ 后台课程级调用（无 user），逐次记
                from core.analytics.token_usage import track_llm_response

                await track_llm_response(resp, model=model, mode="topic_build", **track)
            # 围栏/前后噪声/尾逗号容错走全库单一真相源（judge/question/research 同款）
            data = extract_json_from_llm(resp.choices[0].message.content or "")
        except Exception as exc:  # 埋点/解析偶发崩不废整轮——按本次失败重试
            logger.warning("[course_topics] 响应处理失败（第 %d 次）：%s", attempt, exc)
            continue
        if data:
            return data
        logger.warning("[course_topics] LLM 输出非 JSON（第 %d 次）", attempt)
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


def _parse_reply(data) -> tuple[list[dict], list[dict]]:
    """解析一轮块抽取回复的原始 topics / edges（只做字段清洗，不做校验）。纯函数。"""
    topics: list[dict] = []
    edges: list[dict] = []
    if not isinstance(data, dict):
        return topics, edges
    for t in data.get("topics", []) or []:
        if not isinstance(t, dict):
            continue
        label = str(t.get("label") or "").strip()
        if label:
            topics.append({
                "label": label,
                "definition": str(t.get("definition") or "").strip(),
            })
    for e in data.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        src = str(e.get("src") or "").strip()
        dst = str(e.get("dst") or "").strip()
        if src and dst:
            edges.append({
                "src": src, "dst": dst,
                "strength": e.get("strength"),
                "evidence": str(e.get("evidence") or "").strip(),
            })
    return topics, edges


def _validate_block_edges(
    raw_edges: list[dict], text: str, closed_norm: set[str],
) -> tuple[list[dict], int]:
    """块内边统一校验：闭集端点 + 引句子串 + strength 钳位。纯函数。

    闭集 = 前面块已抽主题 ∪ 本块全部轮次新主题（KGGen ``parse_relations_response``
    口径：已抽实体表；旧点合法当端点，未知端点丢边计数）。evidence 非本块原文子串 →
    置空保留边（端点错=硬伤丢边，引句差=软伤去引句——judge._validate 的 verbatim 思路）。
    """
    from core.memory.course_topic_store import _norm_label

    out: list[dict] = []
    dropped = 0
    for e in raw_edges:
        if _norm_label(e["src"]) not in closed_norm or _norm_label(e["dst"]) not in closed_norm:
            dropped += 1
            logger.warning("[course_topics] 边端点不在闭集，丢弃：%s → %s", e["src"], e["dst"])
            continue
        try:
            conf = min(max(float(e["strength"]), 0.0), 1.0)
            if math.isnan(conf):  # NaN 钻得过 min/max 钳位（json.loads 收裸 NaN 字面量）
                raise ValueError("strength is NaN")
        except (TypeError, ValueError):
            conf = _DEFAULT_STRENGTH
        ev = e["evidence"]
        if ev and ev not in text:
            ev = ""
        out.append({
            "src_label": e["src"], "dst_label": e["dst"],
            "confidence": conf, "evidence": ev,
        })
    return out, dropped


def _render_known(known: dict[str, str], extra_labels: list[str]) -> str:
    """渲染 prompt 的「已有知识点清单」；空清单给占位说明（第一块）。"""
    labels = list(known.values()) + list(extra_labels)
    if not labels:
        return "（暂无——本块是第一块）"
    return "\n".join(f"- {l}" for l in labels)


async def extract_block_graph(
    course_id: str, sec: str, text: str, known: dict[str, str], llm, model: str,
    *, gleaning: int = _DEFAULT_GLEANING,
) -> tuple[list[dict], list[dict], int] | None:
    """单块抽取：新主题 + 课文锚定先修边。返回 None = 两试皆败（上层记 failed_section）。

    known: 归一 label → 原始 label，前面块累积的全局闭集——旧点可当边端点但不重复成
    节点，跨章先修边（「在 X 的基础上」）由此而来。多轮（首轮 + gleaning）的边对
    「known ∪ 本块全部新主题」统一校验：round1 的边可引用 round2 才补抽出的点。
    """
    from core.memory.course_topic_store import _norm_label

    data = await _llm_json(
        llm, model,
        f"章节：{sec}\n\n{text}\n\n" + _EXTRACT_BLOCK_PROMPT.format(known_list=_render_known(known, [])),
        track={"course_id": course_id},
    )
    if data is None:
        return None
    all_topics, raw_edges = _parse_reply(data)
    for _ in range(gleaning):
        first_round = json.dumps({"topics": all_topics, "edges": raw_edges}, ensure_ascii=False)
        gdata = await _llm_json(
            llm, model,
            f"章节：{sec}\n\n{text}\n\n" + _GLEANING_PROMPT.format(
                first_round=first_round,
                known_list=_render_known(known, [t["label"] for t in all_topics]),
            ),
            track={"course_id": course_id},
        )
        if gdata is None:
            break  # 补抽失败不废首轮——gleaning 是增益不是依赖
        gt, ge = _parse_reply(gdata)
        all_topics += gt
        raw_edges += ge

    block_topics: list[dict] = []
    block_norms = set(known)  # 主题去重也含旧点：同点跨块重复出现不重复成节点
    for t in all_topics:
        n = _norm_label(t["label"])
        if n and n not in block_norms:
            block_norms.add(n)
            block_topics.append(t)
    valid_edges, dropped = _validate_block_edges(raw_edges, text, block_norms)
    return block_topics, valid_edges, dropped


async def extract_graph_from_chunks(
    course_id: str, chunks: list[dict], llm, model: str, embed_model=None,
    max_chars_per_call: int = 5000, gleaning: int = _DEFAULT_GLEANING,
) -> tuple[list[dict], list[dict], dict]:
    """从讲义 chunk 按章节聚合抽「主题 + 课文锚定先修边」。chunk: {text, section, order_idx}。

    块顺序 = 章节首现序 → 章内 chunk 序（讲义阅读序）：已抽清单随块累积，后面块的边
    可以引用前面块的点。一章内按 chunk 边界分块（每块 <= max_chars_per_call 字符），
    单 chunk 不截断。max_chars_per_call 权衡：太大→reasoning 模型输出预算(4096)被推理
    占满→content 截断；太小→调用次数多、跨块重复多。默认 5000（probe 实测可调）。

    返回 (topics, raw_edges, meta)：topics 待 merge_synonymous 去重定 topic_id；
    raw_edges 端点是 label，merge 后经别名表重映射成 topic_id；meta 含
    failed_sections（两试皆败的块，"章节名（第n块）"，随 GET payload 暴露给教师）
    与 edges_dropped_invalid（闭集校验丢边数）。
    """
    by_section: dict[str, list[dict]] = {}
    for ck in chunks:
        sec = ck.get("section") or ck.get("source") or "未知章节"
        by_section.setdefault(sec, []).append(ck)

    topics: list[dict] = []
    raw_edges: list[dict] = []
    failed_sections: list[str] = []
    dropped_invalid = 0
    known: dict[str, str] = {}  # 归一 label → label：全局闭集，随块累积
    order = 0
    from core.memory.course_topic_store import _norm_label

    for sec, group in by_section.items():
        for bi, block in enumerate(_pack_blocks(group, max_chars_per_call), 1):
            text = "\n".join(ck["text"] for ck in block)
            res = await extract_block_graph(
                course_id, sec, text, known, llm, model, gleaning=gleaning,
            )
            if res is None:
                failed_sections.append(f"{sec}（第{bi}块）")
                continue
            block_topics, block_edges, dropped = res
            dropped_invalid += dropped
            for t in block_topics:
                topics.append({
                    "topic_id": "",  # 合并去重后定（label 归一 + embedding 相似）
                    "label": t["label"],
                    "definition": t["definition"],
                    "source_section": sec,
                    "order_idx": order,
                    "embedding": None,  # 批量 gather 后回填
                })
                order += 1
                known[_norm_label(t["label"])] = t["label"]
            raw_edges.extend(block_edges)
    # embedding 并发（_aget_embed 自吞异常返 None，单条失败不拖垮整批）——
    # 串行逐个 await 数十次网络往返会白占 worker 槽位；Semaphore 限流防嵌入端点 429
    if topics and embed_model is not None:
        sem = asyncio.Semaphore(_EMBED_CONCURRENCY)

        async def _embed_one(t: dict) -> list[float] | None:
            async with sem:
                return await _aget_embed(embed_model, f"{t['label']}：{t['definition']}")

        embs = await asyncio.gather(*(_embed_one(t) for t in topics))
        for t, emb in zip(topics, embs):
            t["embedding"] = emb
    if failed_sections:
        logger.warning("[course_topics] course=%s %d 块抽取失败：%s",
                       course_id, len(failed_sections), "; ".join(failed_sections))
    return topics, raw_edges, {
        "failed_sections": failed_sections,
        "edges_dropped_invalid": dropped_invalid,
    }


def merge_synonymous(
    topics: list[dict], sim_threshold: float = 0.88
) -> tuple[list[dict], dict[str, str]]:
    """合并同义主题：label 字符串归一 + embedding 余弦相似度去重。

    保留 order_idx 最小者，定义取较长者，embedding 取非空者（首个为 NULL 而重复项
    有值时用重复项的——NULL 进表后 _match_topic 直接淘汰，该主题永远匹配不上）。
    返回 (merged, alias)：alias 把每个输入 label 的归一形式映射到最终 topic_id——
    课文边端点是抽取时的原始 label，去重后须经别名表重映射到规范 topic_id。
    纯函数、可单测。归一化与余弦复用 course_topic_store（与门控读出侧同口径）。
    """
    from core.memory.course_topic_store import _norm_label, cosine

    merged: list[dict] = []
    norm_to_idx: dict[str, int] = {}
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
            if not merged[dup]["embedding"] and t["embedding"]:
                merged[dup]["embedding"] = t["embedding"]
                seen[dup] = (seen[dup][0], t["embedding"])
            norm_to_idx[n] = dup
            continue
        norm_to_idx[n] = len(merged)
        seen.append((n, t["embedding"] or []))
        merged.append(t)
    for i, t in enumerate(merged):
        t["order_idx"] = i
        t["topic_id"] = _norm_label(t["label"])[:32] or f"topic{i}"
    alias = {n: merged[i]["topic_id"] for n, i in norm_to_idx.items()}
    return merged, alias


def _mk_edge(
    src_id: str, dst_id: str, confidence: float,
    relation: str = "prerequisite", evidence: str | None = None,
) -> dict:
    """构造一条前置边 dict（src=前置，dst=被前置——门控回溯方向契约）。

    evidence 恒有键：课文锚定边带原文引句，闭包推断边/seed 边为 None（无证据）——
    消费方统一 ``e["evidence"]`` 直取，不区分键缺失与 None。
    """
    return {
        "src": src_id, "dst": dst_id,
        "relation": relation, "confidence": round(float(confidence), 4),
        "evidence": evidence or None,
    }


def _edge_better(a: dict, b: dict) -> bool:
    """同 (src,dst) 重复边取舍：置信高者优先；置信打平时有证据者优先。纯函数。"""
    if a["confidence"] != b["confidence"]:
        return a["confidence"] > b["confidence"]
    return bool(a["evidence"]) and not b["evidence"]


def finalize_edges(raw_edges: list[dict], alias: dict[str, str]) -> tuple[list[dict], int]:
    """课文边 label → topic_id：别名重映射、自环剔除、(src,dst) 去重。纯函数。

    自环 = 重映射后 src==dst（两写法合并成同一主题）。方向冲突（不同块对同一对点
    给出相反方向）不在此仲裁——两个方向都保留，交给 break_cycles 删环删低置信边。
    返回 (edges, dropped)；dropped 含重映射失败（闭集校验后理论上不发生，防御性计数）。
    """
    from core.memory.course_topic_store import _norm_label

    by_pair: dict[tuple[str, str], dict] = {}
    dropped = 0
    for e in raw_edges:
        src = alias.get(_norm_label(e["src_label"]))
        dst = alias.get(_norm_label(e["dst_label"]))
        if not src or not dst or src == dst:
            dropped += 1
            continue
        edge = _mk_edge(src, dst, e["confidence"], evidence=e["evidence"] or None)
        cur = by_pair.get((src, dst))
        if cur is None or _edge_better(edge, cur):
            by_pair[(src, dst)] = edge
    return list(by_pair.values()), dropped


def break_cycles(edges: list[dict]) -> list[dict]:
    """删环保 DAG：DFS 找环，删环上 confidence 最低的边，重复直到无环。

    门控 get_prereq_predecessors 的 BFS 有 preds 集合防死循环，运行时遇环不炸；
    但先修语义是严格偏序（反对称+传递，CSUR 2025 综述），成环属 LLM 判定噪声，
    删最低置信边保留最可信结构。顺序契约：须在 close_transitive 之前调用
    （DAG 上做闭包才不引环）。纯函数、可单测。
    """
    remaining = list(edges)  # 只 list 拷贝（remove 用），边 dict 本身不改
    while True:
        adj: dict[str, list[str]] = {}
        for e in remaining:
            adj.setdefault(e["src"], []).append(e["dst"])
        color: dict[str, int] = {}  # 0=未访问 1=在当前路径 2=已完成
        cycle: list[str] | None = None

        def dfs(u: str, path: list[str]) -> list[str] | None:
            color[u] = 1
            path.append(u)
            for v in adj.get(u, []):
                if color.get(v, 0) == 0:
                    found = dfs(v, path)
                    if found:
                        return found
                elif color.get(v) == 1:
                    return path[path.index(v):] + [v]  # 环：从 v 起绕回 v
            path.pop()
            color[u] = 2
            return None

        for node in list(adj):
            if color.get(node, 0) == 0:
                cycle = dfs(node, [])
                if cycle:
                    break
        if not cycle:
            return remaining
        ring = set(zip(cycle, cycle[1:]))
        victim = min(
            (e for e in remaining if (e["src"], e["dst"]) in ring),
            key=lambda e: e["confidence"],
        )
        logger.warning(
            "[course_topics] 检测到环 %s → 删最低置信边 %s→%s（conf=%.2f）",
            "→".join(cycle), victim["src"], victim["dst"], victim["confidence"],
        )
        remaining.remove(victim)


def close_transitive(edges: list[dict]) -> list[dict]:
    """传递闭包补边：A→B、B→C ⇒ 补 A→C（confidence = 各路径 min 置信的最大值）。

    门控读侧（get_prereq_predecessors）本就 BFS 算全闭包，运行时预警效果等价；
    此步主要服务学生端可视化（远距离前置直接可见）与教师 safety_weight 标注粒度。
    推断边 relation="prerequisite_inferred" 与 LLM 直接判定的 "prerequisite" 区分
    （前端"直接前置"/教师核对须过滤推断边，confidence 语义也不同：|score| vs 路径 min）。
    已有直接边不覆盖（LLM 直接判定优先）。须在 break_cycles 之后调用。纯函数、可单测。
    """
    adj: dict[str, list[tuple[str, float]]] = {}
    for e in edges:
        adj.setdefault(e["src"], []).append((e["dst"], float(e["confidence"])))
    existing = {(e["src"], e["dst"]) for e in edges}
    added: list[dict] = []
    for start in adj:
        # 从 start 出发的 widest-path（每条路径取 min 置信，取各路径最大者）
        reach: dict[str, float] = {}
        stack = list(adj[start])
        while stack:
            node, conf = stack.pop()
            cur = reach.get(node)
            if cur is not None and cur >= conf:
                continue
            reach[node] = conf
            for nxt, c2 in adj.get(node, []):
                stack.append((nxt, min(conf, c2)))
        for dst, conf in reach.items():
            if (start, dst) not in existing:
                added.append(_mk_edge(start, dst, conf, relation="prerequisite_inferred"))
    return edges + added


def assemble_edges(
    raw_edges: list[dict], alias: dict[str, str], topics: list[dict],
    *, dropped_invalid: int,
) -> tuple[list[dict], dict]:
    """课文边 → 入库边：重映射去重 → 删环 → 闭包 → stats。纯函数、无 LLM。

    边全部来自课文锚定抽取（不做两两判定）。stats 五键：
    - text_mined_edges：过闭集校验与去重后的直接边数
    - edges_dropped_invalid：块级闭集丢边 + 重映射/自环丢边（质量信号：闭集对不上多
      说明模型端点写法漂移，可观察调 prompt）
    - cycle_broken / transitive_added：删环数 / 闭包补边数
    - cross_section_edges：跨章直接边数（验收口径只数直接边——推断边经 fan-in/fan-out
      抬高指标，且推断跨章边必伴随直接跨章边，只数直接边不损失「0→正」检出能力）
    """
    direct, finalize_dropped = finalize_edges(raw_edges, alias)
    acyclic = break_cycles(direct)
    closed = close_transitive(acyclic)
    sec_of = {t["topic_id"]: t.get("source_section") for t in topics}
    cross = sum(1 for e in acyclic if sec_of.get(e["src"]) != sec_of.get(e["dst"]))
    stats = {
        "text_mined_edges": len(direct),
        "edges_dropped_invalid": dropped_invalid + finalize_dropped,
        "cycle_broken": len(direct) - len(acyclic),
        "transitive_added": len(closed) - len(acyclic),
        "cross_section_edges": cross,
    }
    logger.info(
        "[course_topics] 前置边：课文锚定 %d（闭集/自环弃 %d）→ 删环 %d → 闭包 +%d → 总 %d（跨章 %d）",
        stats["text_mined_edges"], stats["edges_dropped_invalid"],
        stats["cycle_broken"], stats["transitive_added"], len(closed), cross,
    )
    return closed, stats


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
    gleaning = s.course_topic.gleaning
    logger.info(
        "[course_topics] build_topic_graph course=%s model=%s chunks=%d force=%s gleaning=%s",
        course_id, model, len(chunks), force, gleaning,
    )

    topics, raw_edges, meta = await extract_graph_from_chunks(
        course_id, chunks, llm, model, embed_model, gleaning=gleaning,
    )
    if not topics:
        # 全量失败保护：所有块两试皆败（坏 key / 持续限流熔断）时 force 已删旧图——
        # 不 commit 空图冒充 built；raise 让 worker deadletter，事务回滚保住旧图。
        # 部分失败不算（topics 非空），教师经 failed_sections 可见可重试。
        raise RuntimeError(
            f"course={course_id} 全部块抽取失败（{len(chunks)} chunks，LLM 不可用？），不落空图"
        )
    topics, alias = merge_synonymous(topics)
    edges, edge_stats = assemble_edges(
        raw_edges, alias, topics, dropped_invalid=meta["edges_dropped_invalid"],
    )

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
            evidence=e["evidence"],
        ))
    await db.commit()
    export_json(
        course_id, topics, edges,
        failed_sections=meta["failed_sections"], stats=edge_stats,
    )
    logger.info("[course_topics] build_topic_graph done course=%s %d主题 %d边", course_id, len(topics), len(edges))
    return {
        "status": "built", "course_id": course_id,
        "topics": len(topics), "edges": len(edges),
        "cross_section_edges": edge_stats["cross_section_edges"],
        "failed_sections": meta["failed_sections"],
    }
