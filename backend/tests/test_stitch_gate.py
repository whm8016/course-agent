"""学情拼接门控（单阈值两档）+ cooldown + logit 判据。

决策口径（2026-08 由 VoI 三档简化定稿）：p_need >= τ 过闸即一律完整简报，
τ = C_FA/(C_FA + p_need·C_FN)，C_FA=1/C_FN=3 的代数正解即 p_need >= 0.434
（p_need >= 1/(1+3·p_need)）——门槛不随轮变化。

阶段D 改动：C1 候选判据从 risk >= 0.5 换 theta 空间（theta <= WEAK_MARGIN≈-0.2007，
等价「至少一次判 poor」——2026-08-18 由两次下调，判定观测常摊到兄弟主题上凑不满
两次；不用 theta - b——班级难度不决定谁能入闸）。测试 mastery_map
用真实折叠器（fold_observations）算出的状态，不再手拍 0.9。
"""
import time
from unittest.mock import AsyncMock, patch

import pytest

from core.memory.mastery import forget_prob
from core.memory.mastery_state import LEVEL_OUTCOME, W_JUDGE, fold_observations
from core.memory.proactive import (
    WEAK_MARGIN,
    decide_stitch,
)

_PREREQ = {"ohm"}


def _folded_map(levels: list[str], b_topic: float = 0.0) -> dict[str, dict]:
    """真实折叠：给定判定等级序列 → mastery_map 单主题条目（plan 阶段D 验收口径）。

    b_topic 只进 risk/logit（影响 p_need），不进入闸判据（theta）。
    """
    obs = [
        {"created_at": float(i), "outcome": LEVEL_OUTCOME[lv], "weight": W_JUDGE}
        for i, lv in enumerate(levels)
    ]
    s = fold_observations(obs)
    return {
        "ohm": {
            "risk": round(s.risk, 4),
            "theta": round(s.theta, 4),
            "logit": round(s.theta - b_topic, 4),
            "p_forget": 1.0,
            "observation_count": s.n_obs,
        }
    }


# ── 阈值锚点（与 WEAK_MARGIN 同纪律：代价/权重一改测试就红） ─────────────────


def test_tau_threshold_semantic_anchor():
    """p_need >= 0.434 的语义锚点：两次判未掌握后约六天未练（b=0 口径）。

    t=5 天 p_need≈0.40 不过闸；t=7 天 p_need≈0.48 过闸。用真实折叠 + forget_prob
    实算——DEFAULT_C_FA/C_FN 或 W_JUDGE 改动会红，强制重定门槛。
    """
    m = _folded_map(["poor", "poor"])  # risk ≈ 0.632（theta -0.54 越过 WEAK_MARGIN）
    m["ohm"]["p_forget"] = forget_prob(5, 5.0)
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.should_stitch is False, "t=5 天 p_need≈0.40 < 0.434 不应拼"
    assert brief.reason == "below_tau"

    m["ohm"]["p_forget"] = forget_prob(7, 5.0)
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.should_stitch is True, "t=7 天 p_need≈0.48 >= 0.434 应拼"


def test_tau_formula_solves_to_fixed_point():
    """τ 公式在默认代价下的代数正解是 0.434（门槛不随轮变化的证明）。"""
    for p in (0.1, 0.3, 0.434, 0.6, 0.9):
        m = _folded_map(["poor", "poor"])
        m["ohm"]["p_forget"] = p / m["ohm"]["risk"]  # 拼出指定 p_need
        brief = decide_stitch("kcl", _PREREQ, m, set())
        if p < 0.434:
            assert brief.should_stitch is False
        # p >= 1/(1+3p) 的正根 = (-1+√13)/6 ≈ 0.434，门槛恒定
        assert (brief.p_need >= brief.tau) == brief.should_stitch


def test_recurrence_floor_beats_threshold():
    """复发保底 0.7 > 0.434：进复发候选即必开口（两档的必然结果）。"""
    m = _folded_map(["fair"])  # 复发与薄弱判据无关
    m["ohm"]["p_forget"] = 0.0
    brief = decide_stitch("ohm", set(), m, {"ohm"})
    assert brief.should_stitch is True
    assert brief.reason == "recurrence"
    assert brief.p_need >= 0.7


# ── 阶段D 回归：fair 不入闸 / 1次poor入闸 / 2次poor入闸 ────────────────────


def test_one_fair_not_candidate():
    """1 次 fair → theta≈0 远高于 WEAK_MARGIN → no_candidate（旧 >=0.5 会误入）。"""
    brief = decide_stitch("kcl", _PREREQ, _folded_map(["fair"]), set())
    assert brief.should_stitch is False
    assert brief.reason == "no_candidate"


def test_one_poor_is_candidate_and_fires_when_old():
    """语义锚点（2026-08-18 下调后）：1 次 poor 即入闸（theta -0.30 <= WEAK_MARGIN
    -0.2007）；p_forget=1（久未练）时 p_need≈0.57 > 0.434 应拼。2 次 poor 同样入闸。"""
    one = decide_stitch("kcl", _PREREQ, _folded_map(["poor"]), set())
    assert one.reason == "prereq_gap"
    assert one.should_stitch is True
    two = decide_stitch("kcl", _PREREQ, _folded_map(["poor", "poor"]), set())
    # 入闸为候选：p_forget=1 时 p_need≈0.63 > 0.434 应拼
    assert two.reason == "prereq_gap"
    assert two.should_stitch is True


def test_two_poor_old_observation_silent_when_just_seen():
    """刚观测过（p_forget≈0）→ 候选在但 below_tau（遗忘耦合：现在正学着呢）。"""
    m = _folded_map(["poor", "poor"])
    m["ohm"]["p_forget"] = 0.05
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.reason == "below_tau"


# ── cooldown ─────────────────────────────────────────────────────────────


def test_cooldown_blocks_within_24h():
    now = time.time()
    m = _folded_map(["poor"] * 6)
    m["ohm"]["last_stitched_at"] = now - 3600
    brief = decide_stitch("kcl", _PREREQ, m, set(), now=now)
    assert brief.should_stitch is False
    assert brief.reason == "cooldown"


def test_cooldown_expires_after_24h():
    now = time.time()
    m = _folded_map(["poor"] * 6)
    m["ohm"]["last_stitched_at"] = now - 25 * 3600
    brief = decide_stitch("kcl", _PREREQ, m, set(), now=now)
    assert brief.should_stitch is True


# ── safety_weight 乘子（教师 PATCH 真生效，不再是假开关） ────────────────────


def test_safety_weight_zero_silences_topic():
    """safety_weight=0 → p_need 归零，below_tau：教师标 0 = 该主题永不因它预警。"""
    m = _folded_map(["poor", "poor"])
    m["ohm"]["p_forget"] = 1.0
    m["ohm"]["safety_weight"] = 0.0
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.should_stitch is False
    assert brief.reason == "below_tau"
    assert brief.p_need == 0.0


def test_safety_weight_amplifies_marginal_topic():
    """below_tau 边缘候选 × safety_weight 3 → 过闸（漏报代价更高的安全关键主题）。"""
    m = _folded_map(["poor", "poor"])
    m["ohm"]["p_forget"] = 0.5  # p_need ≈ 0.63*0.5 ≈ 0.32 < 0.434（below_tau）
    brief_no_sw = decide_stitch("kcl", _PREREQ, m, set())
    assert brief_no_sw.reason == "below_tau"

    m["ohm"]["safety_weight"] = 3.0
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.should_stitch is True
    assert brief.p_need == min(1.0, brief_no_sw.p_need * 3.0)


def test_safety_weight_default_one_unchanged():
    """无 safety_weight 键（旧调用方/未标注）→ 默认 1.0，决策与原来逐位一致。"""
    m = _folded_map(["poor", "poor"])
    m["ohm"]["p_forget"] = 0.05
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.reason == "below_tau"  # 与 test_two_poor_old_observation 同参同果


# ── P0-3：legacy 行冷却按主键回写 ──────────────────────────────────────────


async def test_mark_stitched_updates_by_row_id():
    """legacy 行（kp_id 是旧 hash，靠 label 对齐进 map）冷却必须按行主键生效。

    旧 bug：_mark_stitched 按 kp_id==topic_id 条件 UPDATE，legacy 行 0 命中，
    last_stitched_at 永远为空 → 同一主题每轮重复拼。改后 UPDATE 条件是主键 id。
    """
    from core.memory.proactive import _mark_stitched

    captured: dict = {}

    class _Result:
        rowcount = 1

    class _DB:
        async def execute(self, stmt):
            captured["stmt"] = stmt
            return _Result()

        async def commit(self):
            pass

    await _mark_stitched(_DB(), "row-legacy-1")
    # 编译出的 WHERE 必须按主键 id 过滤，不含 kp_id 条件
    compiled = str(captured["stmt"].compile(compile_kwargs={"literal_binds": True}))
    assert "id" in compiled
    assert "kp_id" not in compiled


async def test_read_mastery_carries_row_id():
    """_read_mastery 的 map 必须带出行主键 row_id（legacy kp_id 行也要带）。"""
    from core.memory.proactive import _read_mastery

    class _Row:
        def __init__(self, kp_id, label):
            self.id = f"row-{kp_id or label}"
            self.kp_id = kp_id
            self.label = label
            self.theta = -1.0
            self.risk = 0.7
            self.last_observed_at = None
            self.strength = 5.0
            self.observation_count = 2
            self.last_stitched_at = None

    class _Scalars:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class _Res:
        def __init__(self, scalars):
            self._scalars = scalars

        def scalars(self):
            return self._scalars

    legacy_rows = [_Row("kp:ab12cd34ef56", "欧姆定律")]  # kp_id 旧 hash，靠 label 对齐
    # 两次 execute：mastery 行 + difficulty_b 查询
    calls = {"n": 0}

    class _DB:
        async def execute(self, _q):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Res(_Scalars(legacy_rows))
            return _Scalars([])  # b_rows 为空

    topic_map = {"欧姆定律": "ohm"}  # label_norm → topic_id
    out = await _read_mastery("u1", "c1", _DB(), topic_map)
    assert "ohm" in out
    assert out["ohm"]["row_id"] == "row-kp:ab12cd34ef56"


async def test_read_mastery_queries_difficulty_with_aligned_tid():
    """b/safety_weight 查询必须用对齐后的 tid（旧 bug：tids_hint 用原始 kp_id，
    legacy 哈希 id 在 course_topic 查不到 → b 恒 0，班级难度对旧行失效）。"""
    from core.memory.proactive import _read_mastery

    class _Row:
        def __init__(self, kp_id, label):
            self.id = f"row-{kp_id}"
            self.kp_id = kp_id
            self.label = label
            self.theta = -0.6
            self.last_observed_at = None
            self.strength = 5.0
            self.observation_count = 2
            self.last_stitched_at = None

    class _Scalars:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class _Res:
        def __init__(self, scalars):
            self._scalars = scalars

        def scalars(self):
            return self._scalars

    legacy_rows = [_Row("kp:legacy-hash-1", "欧姆定律")]  # kp_id 是旧 hash，靠 label 对齐
    b_row = ("ohm", 0.5, 2.0)  # (topic_id, difficulty_b, safety_weight)
    calls = {"n": 0}
    captured: dict = {}

    class _DB:
        async def execute(self, q):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Res(_Scalars(legacy_rows))
            captured["compiled"] = str(q.compile(compile_kwargs={"literal_binds": True}))
            return _Scalars([b_row])

    topic_map = {"欧姆定律": "ohm"}
    out = await _read_mastery("u1", "c1", _DB(), topic_map)

    # 第二次查询（b/sw）的 IN 列表必须含对齐后的 topic_id，绝不含 legacy 哈希 kp_id
    assert "ohm" in captured["compiled"]
    assert "kp:legacy-hash-1" not in captured["compiled"]
    # b/sw 按 tid 生效进 map（旧行不再恒 b=0）
    assert out["ohm"]["logit"] == round(-0.6 - 0.5, 4)
    assert out["ohm"]["safety_weight"] == 2.0


def test_no_stitch_history_no_cooldown():
    brief = decide_stitch("kcl", _PREREQ, _folded_map(["poor"] * 6), set())
    assert brief.should_stitch is True


def test_class_difficulty_not_drags_fair_student_into_gate():
    """收缩验收：fair 且 theta≈0 的学生不得仅因班级 b 过 C1。

    b=0.5（班难主题）：旧口径 theta-b = -0.5 <= WEAK_MARGIN 会误入闸；新口径比
    theta（≈0）不入闸——「这题班里难」不是「该生薄弱」。b 只进 risk/p_need。
    """
    m = _folded_map(["fair"], b_topic=0.5)
    assert m["ohm"]["logit"] <= WEAK_MARGIN  # 旧口径会过闸（这就是 bug）
    assert m["ohm"]["theta"] > WEAK_MARGIN   # 新口径不入闸
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.reason == "no_candidate"


# WEAK_MARGIN 的数值口径锚（= logit(0.45)）归 test_mastery_state 独占——那里是
# 折叠纪律归属地（_weak_margin_from_proactive + sandwich 三连）；此处只测决策行为。


# ── 曾见错法（错因闭环：简报带最近一条 poor 观测的 evidence_quote）──────────


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def _seed_prereq_gap(user_id: str, evidence_quote: str | None):
    """前置臂档案：kcl→kvl 先修边，kcl 折叠行薄弱（theta=-1）且一周未练。

    evidence_quote=None 时观测日志留两条 poor 占位但 evidence 为空；否则最新一条
    poor 带上它（工作项2 的「学生答 vs 正确答案」格式）。matched 固定 kvl（问 KVL，
    闭包回溯到 kcl 缺口），避开复发臂让场景只依赖观测日志的 evidence 字段。
    """
    import time as _time

    from core.db.database import (
        AsyncSessionLocal, CourseTopic, CourseTopicEdge, KnowledgeMastery,
        MasteryObservation,
    )

    now = _time.time()
    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id="c1", topic_id="kcl", label="基尔霍夫电流定律"))
        s.add(CourseTopic(course_id="c1", topic_id="kvl", label="基尔霍夫电压定律"))
        s.add(CourseTopicEdge(
            course_id="c1", src_topic_id="kcl", dst_topic_id="kvl",
            relation="prerequisite",
        ))
        s.add(KnowledgeMastery(
            user_id=user_id, course_id="c1", kp_id="kcl", label="基尔霍夫电流定律",
            theta=-1.0, n_obs=2, last_observed_at=now - 7 * 86400,
        ))
        if evidence_quote is not None:
            s.add(MasteryObservation(
                user_id=user_id, course_id="c1", topic_id="kcl", source="quiz",
                outcome=0.0, weight=1.0, evidence_quote="旧观测", created_at=now - 8 * 86400,
            ))
            s.add(MasteryObservation(
                user_id=user_id, course_id="c1", topic_id="kcl", source="quiz",
                outcome=0.0, weight=1.0, evidence_quote=evidence_quote,
                created_at=now - 7 * 86400,
            ))
        await s.commit()


async def test_brief_carries_latest_poor_evidence(db):
    """有历史 poor 观测 → 简报追加「曾见错法」，取 created_at 最新那条。"""
    from core.db.database import AsyncSessionLocal
    from core.memory.proactive import stitch_for_turn

    await _seed_prereq_gap("u_ev1", "学生答：I=UR；正确答案：I=U/R")
    async with AsyncSessionLocal() as s:
        with patch("core.memory.proactive._match_topic",
                   AsyncMock(return_value=("kvl", 0.9))):
            brief = await stitch_for_turn("回路电压怎么求", "u_ev1", "c1", s)

    assert brief.should_stitch is True
    assert brief.reason == "prereq_gap"
    assert brief.evidence_topic == "kcl"
    assert "曾见错法" in brief.text
    assert "I=UR" in brief.text          # 学生的错误作答进了简报
    assert "旧观测" not in brief.text     # 只取最新一条（created_at desc limit 1）


async def test_brief_without_evidence_unchanged(db):
    """无观测日志（只有折叠行）→ 简报文本与现状一致（fail-safe 回归）。"""
    from core.db.database import AsyncSessionLocal
    from core.memory.proactive import stitch_for_turn

    await _seed_prereq_gap("u_ev2", evidence_quote=None)
    async with AsyncSessionLocal() as s:
        with patch("core.memory.proactive._match_topic",
                   AsyncMock(return_value=("kvl", 0.9))):
            brief = await stitch_for_turn("回路电压怎么求", "u_ev2", "c1", s)

    assert brief.should_stitch is True
    assert "曾见错法" not in brief.text
    # 现状文本原样（首行 + 薄弱描述），不多任何一行
    assert brief.text.startswith("## 学情提示（跨会话，未问亦带）")
    assert brief.text.count("\n") == 1
