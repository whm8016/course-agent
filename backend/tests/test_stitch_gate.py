"""学情拼接门控（单阈值两档）+ cooldown + logit 判据。

决策口径（2026-08 由 VoI 三档简化定稿）：p_need >= τ 过闸即一律完整简报，
τ = C_FA/(C_FA + p_need·C_FN)，C_FA=1/C_FN=3 的代数正解即 p_need >= 0.434
（p_need >= 1/(1+3·p_need)）——门槛不随轮变化。

阶段D 改动：C1 候选判据从 risk >= 0.5 换 theta 空间（theta <= WEAK_MARGIN≈-0.4055，
等价「至少两次判 poor」；不用 theta - b——班级难度不决定谁能入闸）。测试 mastery_map
用真实折叠器（fold_observations）算出的状态，不再手拍 0.9。
"""
import time

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


# ── 阶段D 回归：fair 不入闸 / 1次poor不入闸 / 2次poor入闸 ────────────────────


def test_one_fair_not_candidate():
    """1 次 fair → theta≈0 远高于 WEAK_MARGIN → no_candidate（旧 >=0.5 会误入）。"""
    brief = decide_stitch("kcl", _PREREQ, _folded_map(["fair"]), set())
    assert brief.should_stitch is False
    assert brief.reason == "no_candidate"


def test_one_poor_not_candidate_two_poor_is():
    """语义锚点：1 次 poor 不入闸（theta -0.30 > WEAK_MARGIN -0.4055），
    2 次 poor 入闸（theta -0.54 <= WEAK_MARGIN）。"""
    one = decide_stitch("kcl", _PREREQ, _folded_map(["poor"]), set())
    assert one.reason == "no_candidate"
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


# ── WEAK_MARGIN 与折叠器的耦合纪律（阶段D 落点） ────────────────────────────


def test_weak_margin_is_negative_and_meaningful():
    """口径锚：WEAK_MARGIN ≈ logit(0.4) ≈ -0.4055（等价 risk >= 0.6）。"""
    import math
    assert math.isclose(WEAK_MARGIN, math.log(0.4 / 0.6), abs_tol=1e-4)
