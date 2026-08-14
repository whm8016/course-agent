"""阶段4→阶段D：班级收缩已删——题目难度 b_topic 即班级先验（plan §1 架构取舍）。

原 k=3 经验贝叶斯收缩被取代：Elo 的 difficulty_b（course_topic，全班共享）本身
携带班级信息——新生 theta=0 时 risk = sigmoid(b_topic) 就是班级先验；U(n) 天然
实现「观测少则少动」。手工收缩与它功能重复且更粗，故删。

难度更新口径（plan「班级难度估计修复」）：b 由 worker.cron_refit_difficulty 日频
全班重折（fold_class_difficulty 吃全部来源观测，按 weight 加权），写路径不再
在线更新。本文件验三点（防有人把收缩加回来）：
1. 写路径无班级聚合查询，risk_used = risk（个人折叠值，无收缩覆写）；
   班级聚合走日频 cron，不在写路径；
2. b_topic 隐先验：theta=0 的行，risk 由 course_topic.difficulty_b 决定（班难→险高）；
3. U(n) 稀疏保护：单次观测位移有限（0.3 logit），新生偶发 poor 不至于深陷。
"""
import pytest

from core.memory.mastery_state import W_JUDGE, fold_observations

_PREREQ = {"ohm"}


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


def test_no_cohort_shrink_risk_used_equals_risk():
    """risk_used 不再被班级均值覆写——个人折叠值直接用（班级聚合走日频 cron，
    不在写路径，见 worker.cron_refit_difficulty）。"""
    s = fold_observations([
        {"created_at": 1.0, "outcome": 0.0, "weight": W_JUDGE},
        {"created_at": 2.0, "outcome": 0.0, "weight": W_JUDGE},
    ])
    risk = round(s.risk, 4)
    assert risk == pytest.approx(0.6326, abs=1e-3)  # 2 次 poor 实算
    # _read_mastery 输出的 risk_used 与 risk 同源（无第二次覆写），由 test_voi_gate 覆盖


def test_difficulty_b_is_class_prior():
    """b_topic 隐班级先验：theta 相同的学生，班难度高的主题 risk 更高。
    这就是「题目难度即班级先验」——替代手工班级收缩的信息通道。"""
    from core.memory.mastery_state import FoldState, sigmoid

    easy = FoldState(theta=0.0, b_topic=-0.5)   # 全班都觉得简单
    hard = FoldState(theta=0.0, b_topic=0.5)    # 全班都觉得难
    assert sigmoid(hard.logit) < sigmoid(easy.logit)
    assert (1 - sigmoid(hard.logit)) > 0.5 > (1 - sigmoid(easy.logit))


def test_sparse_observation_displacement_bounded():
    """U(n) 稀疏保护：单次 poor 位移 0.3 logit（K=1·weight0.6·(0-0.5)），
    新生偶发一次 poor 不会深陷（对比旧 delta 口径可一次 +0.2 且无收敛）。"""
    s = fold_observations([
        {"created_at": 1.0, "outcome": 0.0, "weight": W_JUDGE}
    ])
    assert s.theta == pytest.approx(-0.30, abs=1e-9)
    assert s.theta > -0.4055  # 仍高于 WEAK_MARGIN——1 次 poor 不入闸（C1 比 theta）
