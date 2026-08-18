r"""表E live 评测集：30 例槽位映射**现网图**（63 主题/150 边，2026-08-18 本轮重建），教师规则金标。

与 ``stitch_cases_table_e``（离线 v2 层，193/225 冻结世界）的区别：本文件锚定
``graph_mycourse_live.json``（现网 DB 直出，``export_graph_snapshot --live``），
供第三层 ``run_stitch_e2e_live`` 全栈评测用——写路径走真实 ChatPipeline（RAG+mem0）
→ record_episode → consolidate（真判定器），读路径走生产 ``stitch_for_turn``。

30 个槽位类型与 v2 层一一对应（前置缺口×7 / 复发×3 / 遗忘耦合×3 / 冷却×3 /
干净负例×8 / 多缺口+边界×3 / 时钟三态×3）。2026-08-18 本轮重建（63 点）换对：
叠加定理→叠加定理的验证方法（裸「叠加定理」不在新图，另有「不适用于功率计算」
——探测问句写「怎么验证」防串）；戴维南等效电路参数测量→开路电压的测量方法；
三端变阻器的应用/分压器的调压特性→三端变阻器的研究/分压电路；文氏桥→文氏电路
组成正弦波振荡器；万用表的使用→数字万用表的测量方法；元件伏安特性的测试→线性
电阻元件的伏安特性；deep_chain 换有源一端口网络的外特性、rc_scope 换示波器原理、
仪器兄弟换测频幅/校准（旧先修边在新图不存在，换题不硬凑）。
注意：重建图先修关系密集直连——实测 63 主题无任何「仅传递可达」的祖先（深度 ≥2
缺口结构性不存在），旧 deep_chain 的隔级链退化为 inferred 直连边，本层全部缺口
证据均为深度 1，论文口径按直连报、不再声明「深层」。

``mastery`` 在本层是**期望档案**（暴露轮剧本的目标世界，供 extract_ok 对账），
不再直接落库——live 层禁止 INSERT 观测装学生（写路径评测的全部意义）。条目只带
``topic_id`` 不带 ``label``（live 链路无人读，label 单源在图快照）。时间推导
（days_since_last / 复发间距 1 天 / 非复发 32 天）复用 v2 层 ``entry_observations``
的 base 参数，锚点由 runner 传运行时钟。

金标纪律同 2026-08-16：gold.stitch = 教师规则句子（teacher_rule）的数值化，
不是门控公式自己。**teacher_rule 逐句经用户点头后才是定稿**（见文件头状态行）。
冷却三例金标=该提而门控按防打扰沉默（预定 FN，设计争议点）；时钟三态同档案
只改探测钟（clock_days）。neg_class_hard：现网全班 difficulty_b=0（无班级难度
信号），如实按 b=0 口径记，不为回归改生产图。

金标状态：DRAFT（30 句 teacher_rule 待用户逐句点头，2026-08-17 起草）。
"""
from __future__ import annotations

import json
from pathlib import Path

_SNAPSHOT_LIVE = Path(__file__).resolve().parent / "graph_mycourse_live.json"


def load_graph_live() -> dict:
    """现网图快照（确定性）：{topics:[{topic_id,label,difficulty_b,safety_weight}],
    edges:[{src,dst,relation}]}。缺文件先跑 export_graph_snapshot --live。"""
    if not _SNAPSHOT_LIVE.exists():
        raise SystemExit(
            f"缺 {_SNAPSHOT_LIVE.name}——先跑 "
            "python -m scripts.eval_memory.export_graph_snapshot --live（需现网 DB）"
        )
    return json.loads(_SNAPSHOT_LIVE.read_text(encoding="utf-8"))


def live_case_now(case: dict, base: float) -> float:
    """live 决策钟：base（runner 运行钟）+ 可选 clock_days 偏移（时钟三态）。

    冷却三例不走此口径——commit_stitch_cooldown 落盘的是墙钟，探测钟必须也走
    墙钟才能对上（runner 直接用 time.time() 实测值，见 run_stitch_e2e_live）。"""
    return base + case.get("clock_days", 0) * 86400.0


# ---- 30 例：期望档案 + 教师规则金标（teacher_rule 逐句待用户点头，DRAFT）----
# mastery=期望档案（暴露轮目标，extract_ok 对账用）；gold.stitch=教师规则下的
# 「该不该把未问及的学情拼进本轮回答」；gold.evidence_topic 该提时简报指向。
STITCH_CASES_LIVE: list[dict] = [
    # ===== 一、前置缺口（该提）=====
    {
        "id": "pos_gap_rseries_superposition",
        "label": "前置缺口：电阻串并联薄弱问叠加定理",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["电阻串并联电路"],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "叠加要拆成单源单独作用，每一步都在算串并联；串并联两次没掌握又快一个月没练，该提醒",
        "gold": {"stitch": True, "evidence_topic": "电阻串并联电路"},
    },
    {
        "id": "pos_gap_thevenin_equiv",
        "label": "前置缺口：戴维南定理薄弱问开路电压测量",
        "mastery": [
            {"topic_id": "戴维南定理", "levels": ["poor", "poor", "poor"], "days_since_last": 21},
        ],
        "error_repeated": ["戴维南定理"],
        "query": "开路电压怎么测？要用什么仪表、从哪两端量？",
        "matched_topic": "开路电压的测量方法",
        "teacher_rule": "这正是要用定理的地方，该提醒",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
    },
    {
        "id": "pos_gap_potentiometer",
        "label": "前置缺口：三端变阻器薄弱问分压电路",
        "mastery": [
            {"topic_id": "三端变阻器的研究", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["三端变阻器的研究"],
        "query": "三端变阻器的分压电路实验怎么做？滑动端接在哪里？",
        "matched_topic": "三端变阻器的分压电路",
        "teacher_rule": "接法不对调压根本做不下去，该提醒",
        "gold": {"stitch": True, "evidence_topic": "三端变阻器的研究"},
    },
    {
        "id": "pos_gap_wien_amp",
        "label": "前置缺口：文氏频率特性薄弱问文氏电路正弦波振荡器",
        "mastery": [
            {"topic_id": "文氏电路的频率特性", "levels": ["poor", "poor"], "days_since_last": 21},
        ],
        "error_repeated": ["文氏电路的频率特性"],
        "query": "文氏电路组成的正弦波振荡器怎么搭？振荡频率怎么定？",
        "matched_topic": "文氏电路组成正弦波振荡器",
        "teacher_rule": "该提醒",
        "gold": {"stitch": True, "evidence_topic": "文氏电路的频率特性"},
    },
    {
        "id": "pos_gap_deep_chain",
        "label": "深层缺口退直连：一端口外特性薄弱问开路电压测量（新图 inferred 直连边）",
        "mastery": [
            {"topic_id": "有源一端口网络的外特性", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["有源一端口网络的外特性"],
        "query": "开路电压怎么测？要从哪两端量？",
        "matched_topic": "开路电压的测量方法",
        "teacher_rule": "一端口的外特性没掌握，开路电压也量不明白，该提醒",
        "gold": {"stitch": True, "evidence_topic": "有源一端口网络的外特性"},
    },
    {
        "id": "pos_gap_multimeter_equiv_r",
        "label": "仪器先修：数字万用表薄弱问叠加定理验证",
        "mastery": [
            {"topic_id": "数字万用表的测量方法", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["数字万用表的测量方法"],
        "query": "验证叠加定理的实验怎么做？表怎么接、数据怎么记？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "表都不会用，验证实验寸步难行，该提醒",
        "gold": {"stitch": True, "evidence_topic": "数字万用表的测量方法"},
    },
    {
        "id": "pos_gap_rc_scope_method",
        "label": "前置缺口：示波器原理薄弱问过渡过程观察",
        "mastery": [
            {"topic_id": "示波器的原理与主要技术指标", "levels": ["poor", "poor"], "days_since_last": 21},
        ],
        "error_repeated": ["示波器的原理与主要技术指标"],
        "query": "用示波器观察电路过渡过程该怎么做？主要看什么？",
        "matched_topic": "用示波器观察电路过渡过程的方法",
        "teacher_rule": "要用示波器看过渡过程，示波器原理和指标却没掌握，该提醒",
        "gold": {"stitch": True, "evidence_topic": "示波器的原理与主要技术指标"},
    },
    # ===== 二、复发（该提）=====
    {
        "id": "pos_recurrence_thevenin",
        "label": "复发：戴维南反复错又问戴维南",
        "mastery": [
            {"topic_id": "戴维南定理", "levels": ["poor", "poor"], "days_since_last": 3},
        ],
        "error_repeated": ["戴维南定理"],
        "query": "戴维南定理还是没搞懂，开路电压和独立源置零再讲一遍？",
        "matched_topic": "戴维南定理",
        "teacher_rule": "旧错就在眼前，该提",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
    },
    {
        "id": "pos_recurrence_rc",
        "label": "复发：RC微分电路反复错又问它",
        "mastery": [
            {"topic_id": "rc微分电路", "levels": ["poor", "poor"], "days_since_last": 2},
        ],
        "error_repeated": ["rc微分电路"],
        "query": "RC微分电路的输出波形为什么是尖脉冲？",
        "matched_topic": "rc微分电路",
        "teacher_rule": "反复错又问它，该提",
        "gold": {"stitch": True, "evidence_topic": "rc微分电路"},
    },
    {
        "id": "pos_recurrence_beats_gap",
        "label": "复发与前置缺口竞争：先提反复失手的",
        "mastery": [
            {"topic_id": "叠加定理的验证方法", "levels": ["poor", "poor"], "days_since_last": 2},
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor"], "days_since_last": 30},
        ],
        # 电阻串并联 dsl=30：30 天窗只装得下最后一条 poor（1 天距时倒数第二条也落在
        # 窗外），凑不满复发 ≥2，不进名单——它是本例的先修缺口竞争臂，不是复发证据
        "error_repeated": ["叠加定理的验证方法"],
        "query": "验证叠加定理的时候每个源要单独作用，步骤再说一遍？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "两头都该提时，先提反复失手的那件",
        "gold": {"stitch": True, "evidence_topic": "叠加定理的验证方法"},
    },
    # ===== 三、遗忘耦合（刚练过=不提）=====
    {
        "id": "fc_gap_seen_yesterday",
        "label": "遗忘耦合：薄弱但昨天刚练过",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor"], "days_since_last": 1},
        ],
        "error_repeated": ["电阻串并联电路"],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "知道他薄弱，但昨天刚在练这块——先别打扰，看这轮表现再说",
        "gold": {"stitch": False},
    },
    {
        "id": "fc_gap_week_stale",
        "label": "遗忘耦合：薄弱且整整一周没练",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor"], "days_since_last": 7},
        ],
        "error_repeated": ["电阻串并联电路"],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "两次没掌握又隔了一周，差不多该忘了，该提醒",
        "gold": {"stitch": True, "evidence_topic": "电阻串并联电路"},
    },
    {
        "id": "fc_gap_five_days",
        "label": "遗忘耦合：薄弱但才隔五天",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor"], "days_since_last": 5},
        ],
        "error_repeated": ["电阻串并联电路"],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "才隔五天，还在保鲜期里，不必主动提",
        "gold": {"stitch": False},
    },
    # ===== 四、冷却（预定失败区：金标=该提，门控防打扰沉默）=====
    # live 口径：暴露后先探测一次（墙钟 t0，期望要拼）→ commit_stitch_cooldown
    # （落墙钟）→ 拨到 +N 小时做记分探测。cooldown_hours 由 runner 读。
    {
        "id": "cooldown_1h",
        "label": "冷却争议：1小时前刚提醒过又问",
        "mastery": [
            {"topic_id": "戴维南定理", "levels": ["poor", "poor"], "days_since_last": 3},
        ],
        "error_repeated": ["戴维南定理"],
        "query": "戴维南定理的开路电压还是不会求，再帮我讲一遍？",
        "matched_topic": "戴维南定理",
        "teacher_rule": "一小时内连问两次，说明刚才的提醒没解决问题，该再提",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
        "cooldown_hours": 1,
    },
    {
        "id": "cooldown_20h",
        "label": "冷却争议：20小时前提醒过又问",
        "mastery": [
            {"topic_id": "戴维南定理", "levels": ["poor", "poor"], "days_since_last": 3},
        ],
        "error_repeated": ["戴维南定理"],
        "query": "还是戴维南定理卡住了，独立源置零帮我再看看？",
        "matched_topic": "戴维南定理",
        "teacher_rule": "二十小时后还卡在同一处，该提",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
        "cooldown_hours": 20,
    },
    {
        "id": "cooldown_expired_25h",
        "label": "冷却过期：25小时前提醒过又问",
        "mastery": [
            {"topic_id": "戴维南定理", "levels": ["poor", "poor"], "days_since_last": 3},
        ],
        "error_repeated": ["戴维南定理"],
        "query": "隔了一天再问戴维南定理，开路电压怎么求？",
        "matched_topic": "戴维南定理",
        "teacher_rule": "隔了一整天又问，该提",
        "gold": {"stitch": True, "evidence_topic": "戴维南定理"},
        "cooldown_hours": 25,
    },
    # ===== 五、干净负例（不该提）=====
    {
        "id": "neg_unknown",
        "label": "未知问题：不命中任何课程主题",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["电阻串并联电路"],
        "query": "实验报告下周五几点交到教务处？",
        "matched_topic": None,
        "teacher_rule": "跟课程内容无关的问题，不插学情",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_irrelevant",
        "label": "无关：串并联谐振薄弱问最基础的伏安特性",
        "mastery": [
            {"topic_id": "串并联谐振现象", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["串并联谐振现象"],
        "query": "线性电阻元件的伏安特性怎么测？要接什么仪表？",
        "matched_topic": "线性电阻元件的伏安特性",
        "teacher_rule": "问的是最基础内容，薄弱的是不相干的章节，不提",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_corrected",
        "label": "已纠正：串并联曾两次未掌握后连三次掌握",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor", "good", "good", "good"], "days_since_last": 30},
        ],
        "error_repeated": [],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "补上了就别翻旧账",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_partial_recovery",
        "label": "补一半：两次未掌握只补回一次好表现（先观望）",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor", "good"], "days_since_last": 30},
        ],
        "error_repeated": [],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "刚有起色先观望，净欠账约等于一次失手，别小题大做",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_all_solid",
        "label": "全低险：叠加定理验证的直系先修（抽查三个）全判掌握",
        # 直系先修共 6 个（另有欧姆定律/电流表的使用/线性电阻元件的伏安特性），
        # 档案只判三个、其余无观测（门控视为无候选）——论文不得写成「全部直系先修掌握」
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["good", "good"], "days_since_last": 30},
            {"topic_id": "直流稳压电源的使用", "levels": ["good", "good"], "days_since_last": 30},
            {"topic_id": "数字万用表的测量方法", "levels": ["good", "good"], "days_since_last": 30},
        ],
        "error_repeated": [],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "要用的先修都判了掌握，地基是牢的，不提",
        "gold": {"stitch": False},
    },
    {
        # 2026-08-18 弱线降为「一次判 poor 即入闸」后翻正（原 gold=False）：
        # 一次失手 + 三周未碰 = 该回头看一眼，不再是"别小题大做"。
        # 走的是先修（C1 一次即候选×遗忘耦合），不是复发——窗内只有一次 poor，
        # 复发口径（≥ RECURRENCE_MIN_COUNT）下不进 error_repeated
        "id": "neg_single_poor",
        "label": "单次失手：串并联错过一次、三周没碰",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor"], "days_since_last": 21},
        ],
        "error_repeated": [],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "错过一次又三周没碰，该回头看一眼",
        "gold": {"stitch": True, "evidence_topic": "电阻串并联电路"},
    },
    {
        "id": "neg_class_hard",
        "label": "班级难度：戴维南 fair×3（现网 b=0，无班级难度信号）",
        "mastery": [
            {"topic_id": "戴维南定理", "levels": ["fair", "fair", "fair"], "days_since_last": 30},
        ],
        "error_repeated": [],
        "query": "开路电压怎么测？",
        "matched_topic": "开路电压的测量方法",
        "teacher_rule": "题难不等于这个学生薄弱，不提；现网全班 b=0（无班级难度信号），此例如实按 b=0 口径记",
        "gold": {"stitch": False},
    },
    {
        "id": "neg_instrument_sibling",
        "label": "仪器兄弟：示波器测频幅薄弱问校准",
        "mastery": [
            {"topic_id": "示波器测量信号频率和幅值的方法", "levels": ["poor", "poor"], "days_since_last": 28},
        ],
        "error_repeated": ["示波器测量信号频率和幅值的方法"],
        "query": "示波器的校准怎么做？校准信号怎么接、怎么调？",
        "matched_topic": "示波器的校准方法",
        "teacher_rule": "问哪块补哪块，同一台仪器的兄弟功能不算先修，不提",
        "gold": {"stitch": False},
    },
    # ===== 六、多缺口竞争 + 边界 =====
    {
        "id": "multi_gap_stale_wins",
        "label": "多缺口：先提快忘掉的（正练着的不急）",
        "mastery": [
            {"topic_id": "文氏电路的频率特性", "levels": ["poor", "poor"], "days_since_last": 30},
            {"topic_id": "rc网络的频率特性", "levels": ["poor", "poor", "poor"], "days_since_last": 1},
        ],
        # 文氏频率特性 dsl=30：窗内只剩一条 poor，复发口径（≥2）下不进名单
        "error_repeated": ["rc网络的频率特性"],
        "query": "文氏电路组成的正弦波振荡器实验要怎么搭？起振条件是什么？",
        "matched_topic": "文氏电路组成正弦波振荡器",
        "teacher_rule": "两个先修都薄弱，先提快忘掉的那个，正在练的不急",
        "gold": {"stitch": True, "evidence_topic": "文氏电路的频率特性"},
    },
    {
        "id": "multi_gap_weaker_wins",
        "label": "多缺口：都久没练时提更薄弱的",
        "mastery": [
            {"topic_id": "文氏电路的频率特性", "levels": ["poor", "poor"], "days_since_last": 30},
            {"topic_id": "rc网络的频率特性", "levels": ["poor", "poor", "poor"], "days_since_last": 30},
        ],
        # 两条都 dsl=30：窗内各剩一条 poor，都凑不满复发 ≥2，名单为空
        "error_repeated": [],
        "query": "文氏电路组成的正弦波振荡器实验要怎么搭？起振条件是什么？",
        "matched_topic": "文氏电路组成正弦波振荡器",
        "teacher_rule": "都久没练，提更薄弱的那个",
        "gold": {"stitch": True, "evidence_topic": "rc网络的频率特性"},
    },
    {
        "id": "neg_answer_is_teaching",
        "label": "回答即教学：问的就是自己薄弱的主题本身（旧账已出复发窗）",
        "mastery": [
            {"topic_id": "rc微分电路", "levels": ["poor", "poor"], "days_since_last": 45},
        ],
        "error_repeated": [],
        "query": "RC微分电路的原理是什么？输出和输入是什么关系？",
        "matched_topic": "rc微分电路",
        "teacher_rule": "这轮回答本身就在教这块，不用再叠一层预警",
        "gold": {"stitch": False},
    },
    # ===== 七、时钟三态（同档案不同今天，遗忘/冷却由管道按 now 实算）=====
    # 同一套期望档案：电阻串并联电路 2×poor、最后接触 base−1 天、问叠加定理；
    # clock_day28 两例只把探测钟加 27 天（day28 例年龄 = 1+27 = 28 天）。
    {
        "id": "clock_day1_silent",
        "label": "时钟三态：刚练过一天（保鲜期内，沉默）",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor"], "days_since_last": 1},
        ],
        "error_repeated": ["电阻串并联电路"],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "昨天刚练过，还在保鲜期，先别打扰",
        "gold": {"stitch": False},
    },
    {
        "id": "clock_day28_stitch",
        "label": "时钟三态：同一档案四周后（生疏了，该提）",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor"], "days_since_last": 1},
        ],
        "error_repeated": ["电阻串并联电路"],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "同一个档案，四周后还卡在叠加定理，地基早该提了",
        "gold": {"stitch": True, "evidence_topic": "电阻串并联电路"},
        "clock_days": 27,
    },
    {
        "id": "clock_day28_cooldown",
        "label": "时钟三态：四周后且一小时前刚提醒过（冷却，门控沉默是设计争议）",
        "mastery": [
            {"topic_id": "电阻串并联电路", "levels": ["poor", "poor"], "days_since_last": 1},
        ],
        "error_repeated": ["电阻串并联电路"],
        "query": "叠加定理怎么验证？多个电源的电路要怎么拆？",
        "matched_topic": "叠加定理的验证方法",
        "teacher_rule": "四周后且一小时前刚提醒过同一件事；门控按防打扰沉默（争议预定漏提醒，同 cooldown 两例口径）",
        "gold": {"stitch": True, "evidence_topic": "电阻串并联电路"},
        "clock_days": 27,
        "cooldown_hours": 1,
    },
]
