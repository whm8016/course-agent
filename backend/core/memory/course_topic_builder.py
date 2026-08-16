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

import asyncio
import json
import logging
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

_EXTRACT_TOPICS_PROMPT = """---Role---
你是课程知识图谱专家，负责从课程讲义文本中抽取可考核的知识点。

---Goal---
抽出本节讲义中实际教学的知识点（KC），每个给一句话定义。

---Instructions---
1. 粒度锚定：一个知识点 = 可单独考核的最小知识单元。
   - 正例：「戴维南等效」「三端变阻器的连接方式」「一阶RC电路的零输入响应」
   - 反例：「电路」「实验」「基本理论」（过宽不可考核，禁止输出这类泛化词）
2. label 必须逐字使用讲义中出现的原词——不翻译、不改写、不缩写、不自行概括。
3. 共指消解：讲义中「本章」「该定理」「它」等指代，在输出里解析为所指知识点的完整名称。
4. 只抽本节确实教的内容；没有明确知识点返回空数组；宁少勿凑数，不编造。
5. 输出严格 JSON（无 markdown 围栏、无任何其他文字）：
{"topics": [{"label": "知识点名", "definition": "一句话定义"}]}"""

_JUDGE_PREREQ_BATCH_PROMPT = """---Role---
你是课程结构分析专家，判定知识点之间的先修关系（学 B 之前应先掌握 A，则 A 是 B 的前置）。

---Goal---
对 ---Pairs--- 中每一对知识点 (A, B)，判断 A 与 B 的直接先修方向。

---Instructions---
1. 每对输出 direction 四选一：
   - "forward"：A 是 B 的直接前置（掌握 A 显著有助于理解 B，B 的教学默认学生已会 A）
   - "reverse"：B 是 A 的直接前置
   - "none"：无直接先修关系（含：仅相关但可并行学、互不依赖）
   - "uncertain"：说不清（先修关系微妙、专家也会有分歧的情形）
2. 同时输出 score ∈ [-1, 1]：+1 = 强烈认为 A 前置 B；-1 = 强烈认为 B 前置 A；0 = 无关系/说不清。
3. 非对称互斥：若 A 前置 B，则 B 不得前置 A。若两个方向看起来都成立，说明二者并非
   直接先修（可并行学习），direction 取 "none"。
4. 讲义章节顺序（下方标注）仅是参考证据，不是判据——跨章知识点之间同样可能存在先修
   关系（如第 2 章的「元件伏安特性」是第 5 章「等效内阻测量」的前置）。
5. 判据是知识内容本身的依赖关系，不是字面相似。
6. 输出严格 JSON（无围栏）：{{"pairs": [{{"id": <原样返回输入id>, "direction": "...", "score": 0.0}}]}}
   id 原样返回输入的 id，不要重编号、不要漏。

---Pairs---
课程章节顺序：{sections_summary}

候选对：
{pairs_json}

---Output---
输出 JSON："""

# 批量判定 8 对/批（Batch Prompting, arXiv:2301.08721：6-10 样本/批成本降 ~5x 性能持平）
_BATCH_SIZE = 8
# 建边最低置信：|score| < 0.5 不建（旧实现 0.6 偏严；配四分类+uncertain 复判放宽到 0.5）
_EDGE_SCORE_MIN = 0.5
# embedding 并发上限（照 consolidation._JUDGE_CONCURRENCY 模式——全仓 fan-out 都有 cap）。
# 裸 gather 在大课程数百主题时一次打满嵌入端点，429 风暴被 _aget_embed 吞成 None →
# 静默产出 embedding 全 NULL 的残废图（门控永远匹配不上），限流才能看见。
_EMBED_CONCURRENCY = 8


async def _llm_json(
    llm, model: str, prompt: str, track: dict[str, str] | None = None
) -> dict | None:
    from core.llm.json_extract import extract_json_from_llm

    for attempt in (1, 2):  # 网络抖动/JSON 偶发崩：重试一次，两试皆败返回 None
        try:
            # deepseek-v4-pro 等 reasoning 模型先吐 reasoning_content 再吐 content；
            # max_tokens 太小会被 reasoning 占满→content 空(finish=length)。
            resp = await llm.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4096, stream=False,
            )
            if track is not None:
                # 用量埋点：主题图抽取是 ARQ 后台课程级调用（无 user），逐次记
                from core.analytics.token_usage import track_llm_response

                await track_llm_response(resp, model=model, mode="topic_build", **track)
            # 围栏/前后噪声/尾逗号容错走全库单一真相源（judge/question/research 同款）
            data = extract_json_from_llm(resp.choices[0].message.content or "")
            if data:
                return data
            logger.warning("[course_topics] LLM 输出非 JSON（第 %d 次）", attempt)
        except Exception as exc:
            logger.warning("[course_topics] LLM 抽取失败（第 %d 次）：%s", attempt, exc)
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
) -> tuple[list[dict], list[str]]:
    """从讲义 chunk 按「章节」聚合抽主题。chunk: {text, section, order_idx}。

    一章内按 chunk 边界分块（每块 <= max_chars_per_call 字符），每块独立问一次 LLM，
    跨块重复主题交给 merge_synonymous 去重。单 chunk 不截断、章节超长也不丢内容。
    max_chars_per_call 权衡：太大→reasoning 模型输出预算(4096)被推理占满→content 截断；
    太小→调用次数多、跨块重复多。默认 5000（probe 实测可调）。

    返回 (topics, failed_sections)：重试后仍抽取失败的块记入 failed_sections
    （"章节名（第n块）"），随 GET payload 暴露给教师——主题图有洞时教师可见可重试。
    """
    by_section: dict[str, list[dict]] = {}
    for ck in chunks:
        sec = ck.get("section") or ck.get("source") or "未知章节"
        by_section.setdefault(sec, []).append(ck)

    topics: list[dict] = []
    failed_sections: list[str] = []
    order = 0
    for sec, group in by_section.items():
        for bi, block in enumerate(_pack_blocks(group, max_chars_per_call), 1):
            text = "\n".join(ck["text"] for ck in block)
            data = await _llm_json(
                llm, model, f"章节：{sec}\n\n{text}\n\n{_EXTRACT_TOPICS_PROMPT}",
                track={"course_id": course_id},
            )
            if data is None:
                failed_sections.append(f"{sec}（第{bi}块）")
            for t in (data or {}).get("topics", []) if isinstance(data, dict) else []:
                label = str(t.get("label") or "").strip()
                if not label:
                    continue
                topics.append({
                    "topic_id": "",  # 合并去重后定（label 归一 + embedding 相似）
                    "label": label,
                    "definition": str(t.get("definition") or "").strip(),
                    "source_section": sec,
                    "order_idx": order,
                    "embedding": None,  # 批量 gather 后回填
                })
                order += 1
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
    return topics, failed_sections


def merge_synonymous(topics: list[dict], sim_threshold: float = 0.88) -> list[dict]:
    """合并同义主题：label 字符串归一 + embedding 余弦相似度去重。

    保留 order_idx 最小者，定义取较长者，embedding 取非空者（首个为 NULL 而重复项
    有值时用重复项的——NULL 进表后 _match_topic 直接淘汰，该主题永远匹配不上）。
    纯函数、可单测。归一化与余弦复用 course_topic_store（与门控读出侧同口径）。
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
            if not merged[dup]["embedding"] and t["embedding"]:
                merged[dup]["embedding"] = t["embedding"]
                seen[dup] = (seen[dup][0], t["embedding"])
            continue
        seen.append((n, t["embedding"] or []))
        merged.append(t)
    for i, t in enumerate(merged):
        t["order_idx"] = i
        t["topic_id"] = _norm_label(t["label"])[:32] or f"topic{i}"
    return merged


def build_edge_candidates(topics: list[dict], k: int = 10) -> list[tuple[int, int]]:
    """生成前置判定候选对：embedding 余弦 top-k 相似主题（无向，无顺序/章节约束）。

    语义相关是先修的必要条件——ACE（JEDM 2024）实测语义排序候选 recall 0.684 vs
    随机 0.385，真实课程图直接先修边仅占全部组合 5-15%；先修对几乎必然语义相近，
    用相似度预筛把 LLM 判定集中在高概率区。取代旧「order 差 ≤3」硬窗口（跨章边
    全丢）；讲义顺序不再当过滤器，只作判定 prompt 里的参考证据（Auto-Req 协议）。

    embedding 缺失（网络抖动置 NULL）的主题退回「同章全部 + order 差 ≤3」保底，
    防零候选。返回去重排序后的 (i, j) 索引对（i < j）。纯函数、可单测。
    """
    from core.memory.course_topic_store import cosine

    n = len(topics)
    pair_set: set[tuple[int, int]] = set()
    # 半矩阵一次算一对（余弦对称）——每对一次 cosine 调用（含双方范数），较全矩阵省一半
    sims: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if not topics[i].get("embedding"):
            continue
        for j in range(i + 1, n):
            if not topics[j].get("embedding"):
                continue
            s = cosine(topics[i]["embedding"], topics[j]["embedding"])
            sims[i][j] = sims[j][i] = s
    for i in range(n):
        if topics[i].get("embedding"):
            top = sorted((sims[i][j], j) for j in range(n) if j != i and topics[j].get("embedding"))
            for _, j in top[-k:]:  # 取 top-k（sorted 升序，末 k 个即最大）
                pair_set.add((i, j) if i < j else (j, i))
        else:
            sec = topics[i].get("source_section")
            for j in range(n):
                if j == i:
                    continue
                same_sec = topics[j].get("source_section") == sec
                near = abs(topics[i]["order_idx"] - topics[j]["order_idx"]) <= 3
                if same_sec or near:
                    pair_set.add((i, j) if i < j else (j, i))
    return sorted(pair_set)


def _mk_edge(src_id: str, dst_id: str, confidence: float, relation: str = "prerequisite") -> dict:
    """构造一条前置边 dict（src=前置，dst=被前置——门控回溯方向契约）。"""
    return {
        "src": src_id, "dst": dst_id,
        "relation": relation, "confidence": round(float(confidence), 4),
    }


async def judge_pairs_batch(
    course_id: str, pairs: list[tuple[int, int]], topics: list[dict], llm, model: str,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """批量四分类判定候选对，返回 (edges, uncertain_pairs)。

    direction=forward → src=A；reverse → src/dst 翻转（LLM 可判出讲义顺序与先修
    顺序相反的情形，旧实现方向被讲义顺序锁死）；confidence=|score|（arXiv:2504.08856
    的非对称分数设计：一个数字同时编码方向与置信，配四分类主判据防两头点头）。

    逐项容错：返回缺 id / 非法 direction / score 不可解析 → 该项按 uncertain 处理
    （留给上层复判），不整批失败。|score| < _EDGE_SCORE_MIN 不建边。
    """
    if not pairs:
        return [], []
    sections = list(dict.fromkeys(
        t.get("source_section") or "未知章节" for t in topics
    ))
    sections_summary = " → ".join(sections)

    edges: list[dict] = []
    uncertain: list[tuple[int, int]] = []
    for lo in range(0, len(pairs), _BATCH_SIZE):
        batch = pairs[lo:lo + _BATCH_SIZE]
        pairs_payload = [
            {
                "id": nid,
                "a": {"label": topics[i]["label"], "section": topics[i].get("source_section", "")},
                "b": {"label": topics[j]["label"], "section": topics[j].get("source_section", "")},
            }
            for nid, (i, j) in enumerate(batch)
        ]
        data = await _llm_json(
            llm, model,
            _JUDGE_PREREQ_BATCH_PROMPT.format(
                sections_summary=sections_summary,
                pairs_json=json.dumps(pairs_payload, ensure_ascii=False),
            ),
            track={"course_id": course_id},
        )
        by_id: dict[int, dict] = {}
        for item in (data or {}).get("pairs", []) if isinstance(data, dict) else []:
            if isinstance(item, dict):
                try:
                    by_id[int(item.get("id"))] = item
                except (TypeError, ValueError):
                    continue
        for nid, (i, j) in enumerate(batch):
            item = by_id.get(nid)
            if item is None:
                logger.warning("[course_topics] 批量判定缺 id=%d（转复判）", nid)
                uncertain.append((i, j))
                continue
            direction = str(item.get("direction") or "").strip().lower()
            try:
                score = abs(float(item.get("score") or 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if direction in ("forward", "reverse") and score >= _EDGE_SCORE_MIN:
                if direction == "forward":
                    edges.append(_mk_edge(topics[i]["topic_id"], topics[j]["topic_id"], score))
                else:
                    edges.append(_mk_edge(topics[j]["topic_id"], topics[i]["topic_id"], score))
            elif direction == "uncertain":
                uncertain.append((i, j))
    return edges, uncertain


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


async def extract_prereq_edges(
    course_id: str, topics: list[dict], llm, model: str
) -> tuple[list[dict], dict]:
    """抽前置边：embedding top-k 语义候选 → 批量四分类 → uncertain 复判 → 删环 → 闭包。

    取代旧「order 差 ≤3 逐对判定」——旧窗口跨章边系统性丢失（真实课程 32 边跨章仅
    3 条，门控 4 正例 0/4 no_candidate）。返回 (edges, stats)；stats 记候选数/直接边/
    复判/删环/闭包补边/跨章边数（跨章边数是本修复的核心验收指标）。
    """
    pairs = build_edge_candidates(topics)
    direct, uncertain = await judge_pairs_batch(course_id, pairs, topics, llm, model)
    rechecked: list[dict] = []
    if uncertain:
        logger.info("[course_topics] 复判 %d 个 uncertain 对", len(uncertain))
        rechecked, still = await judge_pairs_batch(course_id, uncertain, topics, llm, model)
        if still:
            logger.info("[course_topics] %d 对复判仍 uncertain，弃（宁漏勿错）", len(still))
        direct = direct + rechecked

    acyclic = break_cycles(direct)
    closed = close_transitive(acyclic)
    sec_of = {t["topic_id"]: t.get("source_section") for t in topics}
    cross = sum(1 for e in closed if sec_of.get(e["src"]) != sec_of.get(e["dst"]))
    stats = {
        "candidates": len(pairs),
        "direct_edges": len(direct) - len(rechecked),
        "rechecked_edges": len(rechecked),
        "cycle_broken": len(direct) - len(acyclic),
        "transitive_added": len(closed) - len(acyclic),
        "cross_section_edges": cross,
    }
    logger.info(
        "[course_topics] 前置边 course=%s：候选 %d → 直接 %d + 复判 %d → 删环 %d → 闭包 +%d → 总 %d（跨章 %d）",
        course_id, stats["candidates"], stats["direct_edges"], stats["rechecked_edges"],
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
    logger.info(
        "[course_topics] build_topic_graph course=%s model=%s chunks=%d force=%s",
        course_id, model, len(chunks), force,
    )

    topics, failed_sections = await extract_topics_from_chunks(
        course_id, chunks, llm, model, embed_model
    )
    topics = merge_synonymous(topics)
    edges, edge_stats = await extract_prereq_edges(course_id, topics, llm, model)

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
    export_json(course_id, topics, edges, failed_sections=failed_sections, stats=edge_stats)
    logger.info("[course_topics] build_topic_graph done course=%s %d主题 %d边", course_id, len(topics), len(edges))
    return {
        "status": "built", "course_id": course_id,
        "topics": len(topics), "edges": len(edges),
        "cross_section_edges": edge_stats["cross_section_edges"],
        "failed_sections": failed_sections,
    }
