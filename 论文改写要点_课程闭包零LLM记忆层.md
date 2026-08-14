# 论文改写要点——课程闭包零 LLM 记忆层（阶段 5f 交付）

> 本文件是 `论文_实验技术与管理投稿版.md` 的改写指令清单（按 plan 阶段 5f：只给要点，正文自写）。
> 所有行号对准 2026-08-14 版本（447 行）。代码已全部落地于 worktree-wobbly-hugging-cook，实测数字见文末。

## 1. 修交叉引用（第 193 行）

现在写「学情拼接门控（详见 §3.8）」——**§3.8 是"技能与外部工具的渐进式披露"，指错了**。
改法：门控公式统一收进新增的 **§3.7.2**（放在 §3.7.1 课程结构层之后），193 行改为「详见 §3.7.2」。
不要为门控另起两个创新标题——§3.7.1（结构层）+ §3.7.2（决策层）正好构成"坐标 + 决策"一对。

## 2. 新增 §3.7.2 主体：VoI 三因子门控（替换现有 PRISM 单阈值表述）

现 42 行贡献声明与正文的门控口径是 PRISM 式 `τ = C_FA/(C_FA + p_need·C_FN)` 单阈值。主框架换成：

- **三因子分解**（VoI，ACL 2026 long 1987 的 clarify-or-commit 框架）：
  - Query Ambiguity → 置信度 c：问句→主题最近邻相似度经 (sim−θ)/(1−θ) 归一（θ=0.55）
  - Task Risk → p_need × severity：p_need 含遗忘项（见第 3 条）与班级难度先验（risk = sigmoid(b−theta)，b 由日频全班重折、按来源权重吃判分/对话/词表全部观测）；severity 是教师标注的主题安全关键度（默认 1.0，通电操作类标高）
  - Cognitive Load → 动作成本：完整简报 0.30 / 一句轻提醒 0.12 / 沉默 0
- **三档决策**：value = p_need·severity·c；NetVoI(full)=value−0.30，NetVoI(light)=0.6·value−0.12，silent=0，取最大。
  等价边界（写进论文方便教师理解）：**value<0.20 沉默、0.20≤value<0.45 轻提醒、≥0.45 完整简报**。
- **cooldown**：同一证据主题 24h 内实际拼过则直接沉默（依据 AIED 2026 主动导师对照实验 N=81：主动组交互升但 perceived control 降，其主动方用轻量个性化 + cooldown 防过载）。
- **诚实声明（必须写）**：VoI 原文是 parameter-free（三因子由 LLM 推断效用），本文**不是**——门控须零 LLM 且可单测，故用**确定性效用表**（少量显式常量，可扫描）。表述为「三因子分解 + 确定性效用」，不得自称 parameter-free。
- **PRISM 降级为消融臂**：单边界等价于 p_need≥0.434，正好落在轻提醒带（0.20–0.45）内部——这是干净的消融叙事：单阈值要么过度打扰（轻提醒也被当 full 拼出）要么全沉默，三档把中间带让给一句轻提醒。

## 3. §3.7 读时衰减段（第 197 行）——方向修正

现在写「有效风险取风险值乘以 exp(−λ·age)，λ=0.01」。**这在门控里方向是反的**（时间越久越不拼 = 埋掉最需要预警的点），且与新实现不符。改为：

- 门控用遗忘概率 **p_forget = 1 − exp(−t/S)**（MemoryBank 口径，S=strength 初值 5 天、教师肯定/强化时 +1，spacing effect）。C1 前置缺口的 p_need = risk_used × p_forget——刚观测的薄弱点不打扰（正学着），久未触碰才预警。C2 错题复发不乘遗忘（复发是错题信号非时间信号）。
- λ=0.01 的 exp(−λ·age) **仅保留给掌握度读出排序**（get_mastery_context），不再进门控。
- 按 `相关论文阅读清单.md` 的修正结论：**明确不再把"读时衰减"当贡献写**（TASA 已做读时衰减），只作工程细节陈述。

## 4. §3.7 L3 段（第 189 行附近）——mem0 写入方式已重写

现文「写入先进入按用户与会话分桶的缓冲队列…」（Redis buffer）已废（flush cron 停用，巩固统一走 episodes 表 outbox），且「按 session 近似 SeCom」的说法要改。改写口径：

- 写入路径：每轮落 episodic 原始层（episodes 表，永不删除的 outbox），后台巩固任务按**累积重要度阈值**触发。
- 巩固时 mem0 按**两槽**写入（TASA 的 persona/event 双 bank 思路，借"分存分取"不借其 LLM Generator）：
  - **persona 槽**：跨概念稳定事实（学习偏好等），一条一记，`infer=False` 逐字入库
  - **event 槽**：**一个主题连续段一条**（SeCom 的 segment 粒度结论：turn 太碎、session 主题混、summary 丢细节），段边界=课程主题变更（闭集最近邻判定），文本规则压成一行（`主题·N轮·信号`），不接 LLMLing-2、不做 LLM 摘要
- 巩固全程 `infer=False` 零 LLM 抽取；检索按两槽分别过滤（persona/event），注入分「## 学习画像 / ## 相关学习事件」两段
- [27] SeCom 引用保留，但删掉"session 分组是其零成本近似"的旧说法（源码注释已同步清理）

## 5. §3.5/§3.7 写侧——补两条引用 + 一个论证

写侧闭集化的论证结构（新增段，建议放 §3.7.1 末尾或 §3.7 掌握度估计前）：

- **Zero-Mem (arXiv:2607.29377)**：记忆构建/组织/检索全程不调 LLM 可行，代价是不做抽象（保留原始 trace）。本文不照搬其两视图检索——抽象已由**建库期**课程主题图提供（教师可审计的配置层，全班共享、每学期一次，摊到每 episode 边际成本为零），运行时生成式抽取是纯浪费（且每次抽出的 id 不稳定，正是写读两侧 id 断链的成因）。
- **SAGE (arXiv:2605.30711)**：贵调用只留"不确定带"。本文判据换成**行为证据**（教师纠正/自述困惑/重复提问/负向反馈，关键词规则）而非 embedding 密度——要抽的是错误模式不是新事实，密度不是好判据。仅强错误信号的轮次才调一次 LLM 抽错题，且只取 error_patterns。
- 落点句：写侧从 O(episodes) 的 LLM 开放抽取变为 O(1) 的闭集最近邻 + 规则信号，LLM 调用只保留在错题不确定带。

## 6. 贡献声明（第 42 行）——按清单改写

**删掉**「首次把主动记忆做成代价敏感门控」类表述（PASK 已占）。改为三条真实差分：

1. **候选集被离线课程前置闭包锁死**（对照 PASK 的开放检索）：门控只考虑"问句命中主题的先修闭包 ∩ 该生薄弱"，全库薄弱点禁止入闸——不拼无关学情由结构保证，而非靠模型自觉。
2. **决策为零 LLM 可审计规则**（对照 IntentFlow 训练分类器）：三因子全部来自已有数据（最近邻相似度/掌握度表/教师标注），纯函数可单测、可向教师解释、跨课零训练。
3. **代价因实验安全后果天然非对称**：漏报（下次仍错，实验中可能损坏仪器/触电）重于误报（答非所问）——这是本项目相对既有工作（代价均为认知代价）唯一真实的代价结构差分，论文靠它立住。

## 7. §5 实验章——四臂表 + 抽取三臂

- **stitch 四臂表**（balanced accuracy，PASK 口径）：none / unconditional / PRISM gate / VoI gate。另报 VoI 三档分布（silent/light/full）。
- **抽取三臂**：开放 LLM / catalog 选择题 / 零 LLM 闭集——主题一致率 + LLM token/回合（闭集恒 0）+ embedding 次数。
- **明确写**：无课堂准实验（无真人班级对照），全部为离线评测；Ask Early/Late 时机曲线需下游学习结果指标，列为 limitation 与后续工作，不硬凑。

## 8. 实测数字（可直接引用，2026-08-14 worktree 实测）

| 评测 | 结果 |
|---|---|
| stitch e2e 四臂（9 例 gold） | baseline_none **0.500** / unconditional **0.500** / gate(PRISM) **1.000** / gate_voi **1.000**（tp=4 fn=0 fp=0 tn=5）；VoI 三档分布 silent=4 light=0 full=4 |
| 抽取闭集臂（8 例主题一致率） | **7/8 = 0.875**（真实 text-embedding-v3） |
| 单测 | 新增 test_proactive(3) + test_topic_match(7) + test_consolidation_topic_match(3) + test_promote_two_slots(10) + test_forgetting_gate(6) + test_cohort_shrink(3) + test_voi_gate(11) + test_eval_memory_runners(2)；全量 1164 passed |

注：closed_set 0.875 的那 1 个 miss 与 open_llm/catalog 两臂数字需在有真实 LLM key 的环境补跑：
`python -m scripts.eval_memory.run_extract_compare --arms closed_set,open_llm,catalog`
`python -m scripts.eval_memory.run_stitch_e2e`

## 9. 明确不写（防过度声明）

- 不写「parameter-free」；不写「读时衰减是贡献」；不写「按 session 近似 SeCom」
- 不写「下学期换真人」（plan 明确：论文不写这个承诺）
- 不把 cooldown 说成从 AIED 论文"提出"——只是"其实证依据"
