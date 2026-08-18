"""表E live 层接线自检（CI 离线，零真实 LLM/embedding）。

五件事：
1. 生产缝 2：``pipeline_common`` 把 ``ctx.metadata["_eval_now"]`` 传进
   ``stitch_for_turn(now=...)``——不设键时传 None（墙钟，线上行为不变）。
2. 案例数据 vs 现网图快照一致性：topic 存在、evidence 在先修闭包、复发口径
   落库可长出、剧本轮数=期望档数、时间升序（30 例全量，纯数据零 DB）。
3. 全流程接线（mock _ask_model + mock judge + 桩 embed 单例）跑 2 例：观测由
   **consolidate 生产写路径**长出（episode_ids 非空 = 走了巩固，不是 runner
   INSERT 装学生）、episode.created_at = 剧本钟、门控读真折叠。
4. 冷却三步流（probe1 → commit → probe2 拨钟）在 mock 下 reason=cooldown。
5. 两条「次数」口径不许合并：进闸（C1）一次即候选、复发（C2）三十天里两次
   （RECURRENCE_MIN_COUNT）——单次失手不进复发集、够不着保底 0.7。

不进 pytest 默认门的（凭据/手动）：全量 30 例 live 跑（run_stitch_e2e_live 正文）。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# ---------- 1. pipeline_common 的 _eval_now 缝 ----------


def _mk_ctx(user_id="u-seam", course_id="c-seam", eval_now=object()):
    """eval_now=object() 当哨兵：不传进 stitch_for_turn 的 kwarg 就还是这个对象。"""
    from core.context import UnifiedContext

    metadata = {"turn_id": "t-seam"}
    if eval_now is not None:
        metadata["_eval_now"] = eval_now
    return UnifiedContext(
        course_id=course_id, user_id=user_id, session_id="s", user_message="叠加定理怎么验证？",
        conversation_history=[], mode="chat", enabled_tools=[], rag_mode="",
        memory_context="", metadata=metadata,
    )


@pytest.mark.parametrize(
    "eval_now, expected_now",
    [
        (1_786_900_000.0, 1_786_900_000.0),  # 设数值键 → 原样传（评测与暴露轮同钟）
        (None, None),  # 不设键（生产路径）→ 门控内部取墙钟，行为不变
        ("1.8e9", None),  # ↓ 类型守卫：非数值（bool 是 int 子类也要拒）回墙钟，
        (True, None),     #   拨钟值进算术会 TypeError、被吞致 mastery 层整轮静默
        ([1.8e9], None),
    ],
)
async def test_eval_now_passed_to_stitch_for_turn(eval_now, expected_now):
    """缝 2：pipeline_common 把 metadata._eval_now 传给 stitch_for_turn(now=...)。"""
    from core.memory.proactive import StitchBrief

    with (
        patch("core.llm.prompts.get_course_prompt", AsyncMock(return_value="")),
        patch(
            "core.memory.proactive.stitch_for_turn",
            AsyncMock(return_value=StitchBrief(False, None, "unknown", 0.0, 1.0)),
        ) as mock_stitch,
        patch(
            "core.memory.mem0_client.build_memory_context", AsyncMock(return_value="")
        ),
    ):
        from core.pipeline_common import build_common_context_layers

        await build_common_context_layers(_mk_ctx(eval_now=eval_now))

    assert mock_stitch.await_args.kwargs.get("now") == expected_now


def test_ws_entry_strips_underscore_metadata_keys():
    """WS 入口剥离客户端预置的下划线内部键（_eval_now 可拨门控钟绕过冷却）。"""
    from api.run import _sanitize_client_metadata

    raw = {
        "_eval_now": 1e18, "_pending_stitch_cooldown": {"x": True},
        "count": 3, "sources": ["kb"], "_timing": "oops",
    }
    clean = _sanitize_client_metadata(raw)
    assert clean == {"count": 3, "sources": ["kb"]}
    # 非 dict（含 None/字符串）一律归 {}——下游 .get/.setdefault 才不炸
    assert _sanitize_client_metadata("not-a-dict") == {}
    assert _sanitize_client_metadata(None) == {}


# ---------- 2. 案例数据一致性（纯数据，零 DB）----------


def test_live_cases_match_graph_and_scripts():
    from collections import defaultdict

    from scripts.eval_memory.stitch_cases_table_e import entry_observations, prereq_predecessors
    from scripts.eval_memory.stitch_cases_table_e_live import (
        STITCH_CASES_LIVE, live_case_now, load_graph_live,
    )
    from scripts.eval_memory.stitch_scripts_table_e import SCRIPTS, expose_sessions
    from core.memory.mastery_state import POOR_OUTCOME_MAX
    from core.memory.proactive import RECURRENCE_MIN_COUNT, RECURRENCE_WINDOW_DAYS
    from core.memory.course_topic_store import PREREQ_RELATIONS

    g = load_graph_live()
    tids = {t["topic_id"] for t in g["topics"]}
    edges = g["edges"]
    base = 1_786_953_600.0  # 任意锚：相对时间推导与锚无关
    n_turns = 0

    # 结构口径守卫：全图先修闭包不超出直连（63 图实测零「仅传递可达」祖先）。
    # 下次重建图长出隔级链时这条会红——「缺口证据全深度 1 / 论文按直连报」的
    # 口径（见案例文件头注）随即须重审，不许静默漂移。
    direct: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e["relation"] in PREREQ_RELATIONS:
            direct[e["dst"]].add(e["src"])
    for t in sorted(tids):
        assert prereq_predecessors(t, edges) == direct.get(t, set()), (
            f"{t} 存在仅传递可达的先修（隔级链）——直连口径失效，深度声明须重审"
        )

    assert len(STITCH_CASES_LIVE) == 30
    assert len({c["id"] for c in STITCH_CASES_LIVE}) == 30, "case id 重复"
    for c in STITCH_CASES_LIVE:
        cid = c["id"]
        assert cid in SCRIPTS, f"{cid} 缺暴露轮剧本"
        for m in c["mastery"]:
            assert m["topic_id"] in tids, f"{cid}: mastery 不在现网图 {m['topic_id']}"
            assert len(m["levels"]) >= 1
        if c["matched_topic"] is not None:
            assert c["matched_topic"] in tids, f"{cid}: matched 不在现网图"
        ev = c["gold"].get("evidence_topic")
        if ev:
            assert ev in tids, f"{cid}: evidence 不在现网图 {ev}"
            if c["gold"]["stitch"]:
                clo = prereq_predecessors(c["matched_topic"], edges) if c["matched_topic"] else set()
                assert ev == c["matched_topic"] or ev in clo, (
                    f"{cid}: evidence {ev} 既非 matched 也不在其先修闭包"
                )

        # 复发口径：观测落库后 30 天窗内同主题 poor 计数过 RECURRENCE_MIN_COUNT 的
        # 集合 == case 声明（计数镜像 _read_error_repeated 的 GROUP BY/HAVING 形状，
        # 门槛常量与探测钟公式都单源自生产侧模块，别在测试里重写字面量）
        now = live_case_now(c, base)
        poor_cnt: dict[str, int] = defaultdict(int)
        for m in c["mastery"]:
            for o in entry_observations(c, m, base=base):
                if o["outcome"] <= POOR_OUTCOME_MAX and o["created_at"] >= now - RECURRENCE_WINDOW_DAYS * 86400:
                    poor_cnt[m["topic_id"]] += 1
        emergent = {t for t, n in poor_cnt.items() if n >= RECURRENCE_MIN_COUNT}
        assert emergent == set(c["error_repeated"]), (
            f"{cid}: 复发口径不符 emergent={sorted(emergent)} declared={sorted(c['error_repeated'])}"
        )

        # 剧本：轮数=期望档数、时间升序（expose_sessions 内部已自检轮数，这里再锁排序）
        sessions = expose_sessions(c, base)
        ats = [s["at"] for s in sessions]
        assert ats == sorted(ats), f"{cid}: 暴露 session 非升序"
        n_turns += sum(len(s["turns"]) for s in sessions)

    assert 70 <= n_turns <= 90, f"暴露轮总数 {n_turns} 超出预算口径（70-90）"


def test_expose_sessions_rejects_script_mismatch():
    """剧本与案例打架 → ValueError（数据 bug 先炸，不静默跑）。"""
    from scripts.eval_memory.stitch_cases_table_e_live import STITCH_CASES_LIVE
    from scripts.eval_memory.stitch_scripts_table_e import expose_sessions

    case = dict(next(c for c in STITCH_CASES_LIVE if c["id"] == "neg_single_poor"))
    case["mastery"] = [dict(case["mastery"][0], levels=["poor", "poor"])]  # 期望 2 档、剧本 1 轮
    with pytest.raises(ValueError, match="剧本"):
        expose_sessions(case, 1_786_953_600.0)


# ---------- 3/4. 全流程接线（mock 面：embed 单例/_ask_model/judge/mem0）----------


class _OneHotEmbed:
    """文本 → 首个命中关键词主题的 one-hot（与自身余弦恰 1，其余 0）——确定性匹配。"""

    def __init__(self, topic_ids: list[str], keyword_map: dict[str, tuple[str, ...]]):
        self._ids = topic_ids
        self._kw = keyword_map

    async def _aget_query_embedding(self, query: str) -> list[float]:
        dim = len(self._ids)
        idx = next(
            (self._ids.index(t) for t, kws in self._kw.items() if any(k in query for k in kws)),
            None,
        )
        if idx is None:
            return [0.0] * dim
        return [1.0 if i == idx else 0.0 for i in range(dim)]


@pytest.fixture
async def live_graph_db():
    """内存库种现网图（one-hot embedding 供匹配），返回 topic 顺序。"""
    from core.db.database import close_db, init_db

    from scripts.eval_memory.stitch_cases_table_e_live import load_graph_live

    await init_db()
    g = load_graph_live()
    from core.db.database import AsyncSessionLocal, CourseTopic, CourseTopicEdge

    async with AsyncSessionLocal() as s:
        for e in g["edges"]:
            s.add(CourseTopicEdge(
                course_id=g["course_id"], src_topic_id=e["src"], dst_topic_id=e["dst"],
                relation=e["relation"],
            ))
        ids = [t["topic_id"] for t in g["topics"]]
        for i, t in enumerate(g["topics"]):
            s.add(CourseTopic(
                course_id=g["course_id"], topic_id=t["topic_id"], label=t["label"],
                embedding=[1.0 if j == i else 0.0 for j in range(len(ids))],
                difficulty_b=t.get("difficulty_b", 0.0),
                safety_weight=t.get("safety_weight", 1.0),
            ))
        await s.commit()
    try:
        yield g["course_id"], ids
    finally:
        await close_db()


async def test_live_runner_wiring_gap_and_cooldown(live_graph_db):
    """2 例全流程：观测走 consolidate 生产写路径长出（非 runner INSERT）、拨钟、门控、冷却三步。"""
    from core.db.database import AsyncSessionLocal, MasteryObservation, MemoryEpisode
    from core.memory.judge import JudgeResult, Judgement
    from sqlalchemy import select

    from scripts.eval_memory import run_stitch_e2e_live
    from scripts.eval_memory.stitch_cases_table_e_live import STITCH_CASES_LIVE
    from scripts.eval_memory.stitch_scripts_table_e import SCRIPTS, expose_sessions

    course_id, topic_ids = live_graph_db
    # 措辞↔主题映射单源自数据（剧本每句自带 topic_id、case.query 自带 matched_topic，
    # 整句做子串关键词——consolidation 喂 match_topics 的是「学生句\n助手句」拼接文本，
    # 必含学生原话）：改问句措辞不再需要手同步关键词表；匹配质量归 --check-match 真向量
    kw: dict[str, tuple[str, ...]] = {}
    for turns in SCRIPTS.values():
        for tid, q in turns:
            kw.setdefault(tid, [])
            kw[tid].append(q)
    for c in STITCH_CASES_LIVE:
        if c["matched_topic"]:
            kw.setdefault(c["matched_topic"], [])
            kw[c["matched_topic"]].append(c["query"])
    kw = {t: tuple(qs) for t, qs in kw.items()}
    fake_embed = _OneHotEmbed(topic_ids, kw)

    def _judge_side_effect(segment_turns, topics, track=None):
        # mock 判定器：请求哪个主题就判哪档——poor 化的案例全 poor（按请求序给）
        return JudgeResult(
            judgements=[
                Judgement(topic_id=t, level="poor", confidence=0.9,
                          evidence_quote=segment_turns[0][0][:20])
                for t, _label, _def in topics
            ],
            abstained=[], error_patterns=[], dropped_invalid_topic=0, dropped_bad_quote=0,
        )

    async def _fake_match_topics(text, course_id_, db, threshold=None):
        return [
            (t, 0.9) for t, kws in kw.items() if any(k in text for k in kws)
        ][:1]

    async def _fake_ask_model(**kwargs):
        return {
            "assistant_msg": f"教师讲解（mock）：{kwargs['user_msg'][:18]}……",
            "tools_used": ["rag"], "memory_context_len": 0, "error": None,
        }

    with (
        patch("core.rag.llamaindex.pg_store._embed_model", fake_embed),
        patch.object(run_stitch_e2e_live, "_probe_retrieval", AsyncMock()),
        patch("scripts.demo_mem0_student_qa._ask_model", AsyncMock(side_effect=_fake_ask_model)),
        patch("core.memory.topic_match.match_topics", AsyncMock(side_effect=_fake_match_topics)),
        patch("core.memory.judge.judge_segment", AsyncMock(side_effect=_judge_side_effect)),
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
        patch("core.memory.mem0_client.get_memory",
              side_effect=RuntimeError("no mem0 in wiring test")),
        # run() finally 里的 close_db 会把 :memory: 整库带走——测试要在 run 之后
        # 查库验「观测由巩固长出」，故挡掉（teardown 由 fixture 用真 close_db 收尾）
        patch("core.db.database.close_db", AsyncMock()),
    ):
        summary = await run_stitch_e2e_live.run(
            case_ids=["pos_gap_rseries_superposition", "cooldown_1h"]
        )

    assert summary["suite"] == "table_e_e2e_live"
    assert summary["cases"] == 2
    by_id = {r["id"]: r for r in summary["details"]}

    # --- 例 1：前置缺口（电阻串并联 2×poor、dsl 28、问叠加定理的验证方法）---
    gap = by_id["pos_gap_rseries_superposition"]
    assert gap["match_ok"] is True
    assert gap["matched_topic_actual"] == "叠加定理的验证方法"
    assert gap["gate"]["stitch"] is True
    assert gap["gate"]["reason"] == "prereq_gap"
    assert gap["gate"]["evidence"] == "电阻串并联电路"
    assert gap["judge_levels"] == {"电阻串并联电路": ["poor", "poor"]}
    assert gap["extract_ok"] is True
    assert gap["expose_turns"] == 2
    assert gap["rag_hit"] is True

    # 观测由 consolidate 长出（episode_ids 非空 = 走了巩固写路径，runner 没 INSERT）
    # user_id 编号与 runner 的全局 enumerate 单源（套件里增删案例不用改这里）
    uids = {c["id"]: f"lv_{i:02d}" for i, c in enumerate(STITCH_CASES_LIVE)}
    user_gap = uids["pos_gap_rseries_superposition"]
    async with AsyncSessionLocal() as s:
        obs = (
            await s.execute(
                select(MasteryObservation).where(MasteryObservation.user_id == user_gap)
            )
        ).scalars().all()
    assert len(obs) == 2
    assert all(o.source == "llm_judge" and o.episode_ids for o in obs)
    # episode.created_at = 剧本钟（dsl 28 → 28/60 天前，锚 base），非墙钟
    case_gap = next(c for c in STITCH_CASES_LIVE if c["id"] == "pos_gap_rseries_superposition")
    want_ats = sorted(sess["at"] for sess in expose_sessions(case_gap, summary["base"]))
    got_ats = sorted(o.created_at for o in obs)
    assert got_ats == want_ats
    async with AsyncSessionLocal() as s:
        eps = (
            await s.execute(select(MemoryEpisode).where(MemoryEpisode.user_id == user_gap))
        ).scalars().all()
    assert sorted(e.created_at for e in eps) == want_ats

    # --- 例 2：冷却三步（probe1 拼了 → commit 落冷却戳 → probe2 拨 +1h 被压住）---
    cd = by_id["cooldown_1h"]
    assert cd["match_ok"] is True
    assert cd["cooldown_committed"] is True
    assert cd["pre_gate"]["stitch"] is True
    assert cd["pre_gate"]["reason"] == "recurrence"
    assert cd["gate"]["reason"] == "cooldown"
    assert cd["gate"]["stitch"] is False
    assert cd["cooldown_fn"] is True  # gold=该提、门控冷却沉默：预定 FN（设计争议）
    assert cd["judge_levels"] == {"戴维南定理": ["poor", "poor"]}

    # 冷却戳确已落库（commit_stitch_cooldown 写的是墙钟，>0 校验非空即可）
    async with AsyncSessionLocal() as s:
        from core.db.database import KnowledgeMastery

        km = (
            await s.execute(
                select(KnowledgeMastery).where(
                    KnowledgeMastery.user_id == uids["cooldown_1h"],
                    KnowledgeMastery.kp_id == "戴维南定理",
                )
            )
        ).scalars().one()
    assert km.last_stitched_at is not None

    # 基线锚：本子集两例 gold 全 True（无负例，tnr 分母 0 记 0）→ never BA 0、always BA 0.5
    assert summary["arms"]["never"]["balanced_accuracy"] == 0.0
    assert summary["arms"]["always"]["balanced_accuracy"] == 0.5


# ---------- 5. 两条「次数」口径不许合并（进闸一次 / 复发两次）----------


async def test_single_poor_never_fires_recurrence():
    """门控锁：所问主题窗内只有一次 poor——不进复发集、够不着保底 0.7、刚栽完沉默。

    进闸（C1）一次判未掌握即可进候选（WEAK_MARGIN）；复发（C2）是三十天里两次
    （RECURRENCE_MIN_COUNT）——两条「次数」是两回事，合并的后果是单次失手借
    复发保底 0.7 直接开口，绕过「刚判薄弱说明正在学」的遗忘耦合。
    """
    from core.db.database import (
        AsyncSessionLocal, MasteryObservation, User, close_db, init_db,
    )
    from core.memory.mastery_state import LEVEL_OUTCOME, W_JUDGE
    from core.memory.proactive import (
        RECURRENCE_MIN_COUNT, RECURRENCE_PNEED_FLOOR, _read_error_repeated, decide_stitch,
    )

    # 口径前提：本测试守护的是「两次才算复发」；想把常量降回 1，先来改这条注释
    assert RECURRENCE_MIN_COUNT >= 2

    await init_db()
    user, course, topic = "u-single-poor", "c-single-poor", "戴维南定理"
    now = 1_786_953_600.0
    try:
        async with AsyncSessionLocal() as s:
            s.add(User(id=user, username=user, password_hash="x"))
            s.add(MasteryObservation(
                user_id=user, course_id=course, topic_id=topic, source="llm_judge",
                outcome=LEVEL_OUTCOME["poor"], weight=W_JUDGE, created_at=now - 3600,
            ))
            await s.commit()

        # 读侧：窗内 1 次 poor（哪怕就在一小时前）< 门槛 → 不进复发集
        topic_map = {topic: topic}
        async with AsyncSessionLocal() as s:
            repeated = await _read_error_repeated(user, course, s, topic_map, now=now)
        assert repeated == set()

        # 决策侧：无先修缺口 → 沉默；p_need 够不着保底 0.7，reason 不许是 recurrence
        mastery = {topic: {"risk": 0.65, "theta": -0.62, "p_forget": 0.0, "label": topic}}
        brief = decide_stitch(topic, set(), mastery, repeated, now=now)
        assert brief.should_stitch is False
        assert brief.reason != "recurrence"
        assert brief.p_need < RECURRENCE_PNEED_FLOOR

        # 同档案带刚练过的先修缺口：走先修×遗忘≈0，同样沉默（不是复发开口）
        mastery_gap = dict(mastery, **{"欧姆定律": {"risk": 0.7, "theta": -0.85, "p_forget": 0.0, "label": "欧姆定律"}})
        b2 = decide_stitch(topic, {"欧姆定律"}, mastery_gap, repeated, now=now)
        assert b2.should_stitch is False and b2.reason == "below_tau"
        assert b2.p_need < RECURRENCE_PNEED_FLOOR

        # 反事实锚：若被错当复发（membership），保底 0.7 立即开口——正是要拦的路
        bad = decide_stitch(topic, set(), mastery, {topic}, now=now)
        assert bad.should_stitch is True and bad.reason == "recurrence"
        assert bad.p_need >= RECURRENCE_PNEED_FLOOR

        # 边界对侧：窗内凑满第二条 poor → 读侧长出复发集（常量是真实边界，不是摆设）
        async with AsyncSessionLocal() as s:
            s.add(MasteryObservation(
                user_id=user, course_id=course, topic_id=topic, source="llm_judge",
                outcome=LEVEL_OUTCOME["poor"], weight=W_JUDGE, created_at=now - 2 * 86400,
            ))
            await s.commit()
            repeated2 = await _read_error_repeated(user, course, s, topic_map, now=now)
        assert repeated2 == {topic}
    finally:
        await close_db()
