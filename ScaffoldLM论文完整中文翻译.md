# ScaffoldLM：面向教学型 LLM 导师的规划引导辅导与评估驱动记忆

**作者：** Zechen Li¹²³, Qiannan Zhu¹²³†, Mei Wang¹²³, Jia Li¹²³, Hua Huang¹²³  
¹ 北京师范大学人工智能学院  
² 北京师范大学人工智能教育北京市重点实验室  
³ 智能技术与教育应用教育部工程研究中心  
**通讯作者：** zhuqiannan@bnu.edu.cn  

**出处：** ACL 2026 长文（第 7165–7188 页）  
**原文 PDF：** https://aclanthology.org/2026.acl-long.325.pdf  
**代码：** https://github.com/BNU-ERC-ITEA/ScaffoldLM  

---

## 摘要

为大型语言模型（LLM）配备教学辅导能力对教育具有重大前景。现有方法模拟导师行为或偏好，并用于 prompt 或微调 LLM 进行对话辅导。然而，此类方法往往难以 sustained 高质量教学对话——既缺少 explicit 分步脚手架，又难以 adapt 于学习者 evolving 的认知状态。为此，我们提出 **ScaffoldLM**：一种面向多轮数学对话辅导的**规划引导框架**，配备**评估驱动记忆**。ScaffoldLM 首先从解题步骤生成**分步教学计划**，作为 explicit 脚手架的稳定 backbone。辅导过程中，评估驱动控制环更新 tutoring memory：推断学习者认知状态、评估当前步骤目标是否达成、自适应选择辅导动作。计划、步骤级进度、推断的学习者状态与对话历史均保存在 memory 中以支持连贯多轮引导。多轮数学辅导 benchmark 实验表明，ScaffoldLM 显著优于强 baseline 的教学质量。

---

## 1 引言

LLM 在自然语言理解与复杂逻辑推理上取得 remarkable 成功，在数学解题方面潜力尤其突出（Ahn 等，2024；OpenAI 等，2024；Li 等，2025）。随着核心能力成熟，兴趣 growing 于将 LLM 从 mere 解题者演进为**个性化导师**（Zhu 等，2025）。有效导师不简单 reveal 答案，而 adherence 教学原则如**脚手架（scaffolding）**，引导学习者独立构建解（Dan 等，2023；Black & Wiliam，1998）。核心挑战在于使 LLM **平衡数学 rigor 与教学 sensitivity**，确保指导既逻辑正确又 adapt 于学习者 evolving 认知状态。

robust 教学辅导仍是重大挑战。早期方法依赖 prompt engineering，将教师 persona 编码进指令（Zhang 等，2024；Kargupta 等，2024），但 notoriously brittle， minor prompt 变化下 collapse 或 long 对话中难以 maintain 一致教学逻辑（Jurenka 等，2024）。近期工作用教育对话上的**监督微调（SFT）**——如 SocraticLM 设计“院长–教师–学生”多智能体系统构建苏格拉底式多轮教学对话用于微调（Liu 等，2024）。Others explore **强化学习（RL）**，如 TutorRL（Dinucu-Jianu 等，2025）基于教学 reward 优化响应。Despite 进展，当前 LLM 导师 struggle 两个 fundamental 局限：

**(1) 缺乏 explicit 教学规划。** 数学教育研究强调脚手架需要 structured、分步计划以 bridge 学习者能力与解之间的差距（Bakker 等，2015）。现有模型 incrementally 生成响应，无 explicit、可验证的分步计划定义 intermediate targets，导致 inconsistent 指导、fragmented 推理或 premature 答案泄露。

**(2) 认知状态自适应弱。** 有效教学需要**形成性评估（formative assessment）**，诊断学习者状态（如困惑、错误概念）以调整支持（Black & Wiliam，1998）。现有方法缺乏 explicit 机制跨轮跟踪步骤级进度与学习者状态，导致 generic 反馈无法 address 具体学习 gap。

为此我们提出 **ScaffoldLM**——面向多轮数学对话辅导的**规划引导框架**。ScaffoldLM 首先将复杂问题 decompose 为**分步教学计划**：有序 intermediate 引导问题及其参考答案序列，为 explicit 脚手架提供 stable backbone。辅导中框架执行**动态控制环**：explicit 评估学习者最新响应以在评估驱动 memory 中更新计划进度，生成 state-aligned 反馈—— either address 当前步骤 immediate 错误概念，或在掌握后 transition 至下一子问题。

为 instantiate ScaffoldLM，我们引入基于**双智能体模拟**的可扩展自动化数据合成 pipeline。非 unstructured chat，我们通过**一致性 enforced 交互环**生成数据：首先模型生成教学计划；每轮学习者智能体扮演 sampled 认知状态，导师智能体评估输入并生成响应。为确保可靠 state-action 监督，仅当导师评估与学习者 intended 状态匹配时保留该轮；否则学习者 resample 话语。 resulting 轨迹序列化为监督数据集，链接问题上下文、计划步骤与学习者状态至辅导动作。

### 主要贡献

(1) **ScaffoldLM 框架**：提出规划引导框架，将辅导 anchored 于分步教学计划，为交互提供 rigorous 逻辑 backbone。  
(2) **评估驱动机制**：引入动态 memory 与控制环，explicit 评估学习者状态并跟踪步骤级进度，enable precise、cognition-aligned 自适应。  
(3) **一致性 enforced 数据合成**：开发双智能体模拟 pipeline，enforced 导师评估与学习者状态间 consistency filter，构建高质量苏格拉底辅导数据用于 instruction tuning。

---

## 2 相关工作

传统 tutorial dialogue 系统如 AutoTutor 依赖 hand-crafted 脚本与 rule-based 对话管理器提供提示、 leading 问题与 corrective 反馈（Graesser 等，2004）。LLM 显著降低构建对话导师成本，但 off-the-shelf 模型常 default 直接给答案而非 sustained 教学交互（Tack & Piech，2022；Macina 等，2023）。现有方法大致分三类：

**Prompt-based steering**：在 prompt 中编码教师角色与教学约束（Wang 等，2025；Sonkar 等，2023；Kargupta 等，2024），但 long-horizon 辅导中 brittle（策略 drift、 subtle 答案泄露）。StratL 通过 intent-conditioned prompt 附加 steering LLM 导师 follow expert 定义的多轮计划（转移图），在 Productive Failure 策略下高中数学演示（Puech 等，2025）。

**监督微调（SFT）**：模仿 human 或 synthetic 高质量教学对话（Macina 等，2023；Chevalier 等，2024；Dan 等，2023；Liu 等，2024 等），但 supervision 常 weakly grounded 于 reference-checked 分步计划，学习者状态 rarely 逐轮 controlled 与 validated。SocraticLM 通过 Dean–Teacher–Student pipeline 构建 SocratiTeach 数据并微调生成多轮苏格拉底对话（Liu 等，2024）。

**Post-training alignment**：RL 或 preference optimization 进一步优化辅导行为（Dinucu-Jianu 等，2025；Scarlatos 等，2024）。TutorRL 提出 online 多轮 RL 框架 explicit 管理教学约束与学习者解题成功 trade-off（Dinucu-Jianu 等，2025），但 reward 信号 typically sparse 且 confounded，优化 sensitive 且有时 encourage shortcut 行为。

Human-annotated 资源如 MathDial、NewtBot 提供 realistic 教师语言与学习者行为但规模有限（Macina 等，2023；Lieb & Goel，2024）。Many works 用 LLM synthesize TutorChat、EduChat、SocraticLM、SocraticMath 等语料（Chevalier 等，2024；Dan 等，2023；Liu 等，2024；Ding 等，2024），但 pipeline 常依赖 prompt 质量，缺乏 reference-checked 分步 backbone 与 controlled 学习者状态转移，导致 unstable 脚手架粒度与 solution leakage 风险。

**与上述方法 contrast**，我们将辅导 anchored 于 explicit 分步计划，并在数据合成中 enforced 教学轨迹与学习者状态的 strict consistency。

---

## 3 教学原则

有效教学 foster 深度概念理解，鼓励 independent 推理而非 passive 接收（Ding 等，2024；Black & Wiliam，1998）。聚焦 1:1 数学辅导，我们考察脚手架与苏格拉底 questioning 如何 engage 学习者 active 解题（Gaffney & Rodgers，2018；Chi & Wylie，2014；Freeman 等，2014）。借鉴 constructive tutoring 与形成性评估研究（Black & Wiliam，1998；Lepper & Woolverton，2002），采用两条核心准则：

**分步脚手架指导（Stepwise Scaffolding Guidance）**  
辅导应 follow explicit 教学计划，将解 decompose 为 intermediate 目标。非 reveal 答案，导师用 minimal prompts 与 locally grounded 问题 sequential 引导学习者，确保 observable 进度与 targeted 支持（Wood 等，1976；Bakker 等，2015）。

**认知对齐苏格拉底辅导（Cognition-aligned Socratic Tutoring）**  
干预必须 adapt 于学习者认知状态。导师应 assess 理解以 apply 与当前进度 aligned 的苏格拉底动作（如 probing questions），促进 active 思考而不造成依赖（Ding 等，2024；Sfard，2001；Lepper & Woolverton，2002）。

---

## 4 ScaffoldLM

我们提出 ScaffoldLM——面向多轮数学对话辅导的**规划引导框架**。ScaffoldLM 首先生成分步教学计划作为 stable backbone，将问题 decompose 为步骤级 intermediate 目标。辅导中维护**评估驱动 memory**，保留完整计划与对话 context，同时 conditioning 生成于当前 active 步骤，enable coherent 脚手架与 reliable 步骤 transition 直至所有步骤完成。

### 4.1 任务形式化

将辅导建模为学习者与 LLM 导师间的多轮交互。给定问题 Q，会话形成轮次序列 {(L₁,T₁), …, (L_N,T_N)}，L₁ = Q。第 i 轮，导师 conditioning 工作 memory M^{i−1} 与学习者话语 L_i 产生下一导师响应：

**T_i = LLM_θ(M^{i−1}, L_i)**

M^{i−1}  summarize 辅导 context：教学计划、当前 active 步骤、跟踪的步骤级进度与学习者状态、对话历史。目标非 direct 答案生成，而是产生**分步指导**，advance 学习者通过 intermediate 目标，同时 adapt 干预于学习者当前认知状态。

### 4.2 分步教学规划

**分步脚手架指导**原则认为 structured、逐步脚手架 enable 学习者 independent 推理并 progressive 到达正确解。复杂数学解题 involve latent 多阶段推理；无 explicit 结构任务 obscure 下一步。将复杂任务 chunk 为分步 targets 使 intermediate 目标 explicit，通过 manageable checkpoints 引导 independent 推理。分步结构亦提供 clear 学习进度证据，困难 arise 时 enable precise 诊断。

形式化：设 Q 为数学问题。ScaffoldLM 首先将 Q decompose 为 N 个逻辑步骤的分步解 rationale R(Q)。对 R(Q) 中每步 t，模型生成 guiding 子问题 q_t 作为 intermediate 目标，及对应答案 a_t。这些对 {(q_t, a_t)} 构成问题的**教学计划** P(Q)：

**P(Q) = [(q_t, a_t)]_{t=1}^N**

计划在步骤 N 终止，a_N 取为 Q 的最终答案。每对 (q_t, a_t) 定义 specific local 学习目标。辅导设置中，步骤答案 a_t 存于 assessment 与 grounding；导师 constrained 避免在 prerequisite 步骤完成前 reveal future-step 答案（尤其 a_N）。

### 4.3 评估驱动辅导

**认知对齐苏格拉底辅导**原则：有效辅导应 adapt 指导于学习者认知状态。ScaffoldLM 通过两个 synergistic 组件实现：**状态对齐生成器（State-Aligned Generator）**与**评估驱动记忆（Assessment-Driven Memory）**。二者经**四步 recurrent 环**协调：生成器执行 **Assess** 与 **Act** 解释学习者状态并产生响应；memory 管理 **Track** 与 **Record** 维护计划进度与交互历史。

第 k 轮，memory M^{k−1} encapsulate 三关键组件：
- 教学计划 P(Q)
- 进度向量 p^{k−1} ∈ {0,1}^N，指示每步完成状态
- 对话历史 H^{k−1} = [(L₁,s₁,T₁), …, (L_{k−1}, s_{k−1}, T_{k−1})]，s ∈ S 为推断的学习者状态

记 M^{k−1} = (P(Q), p^{k−1}, H^{k−1})，t 为当前 active 步骤索引。交互环四操作：

**Assess（评估）**——基于响应 L_k 与当前 memory 推断学习者状态 s_k：  
**s_k = Assess(L_k, M^{k−1})**  
模型从 M^{k−1} 检索当前目标 (q_t, a_t) 评估 L_k 正确性。

**Act（行动）**——基于 s_k 选择 scaffolding 策略生成导师响应 T_k：  
**T_k = Act(L_k, s_k, M^{k−1})**  
例如 s_k 为 Correct 则 affirm 并 proceed；为 Incorrect 则对当前 q_t 提供 targeted hints。

**Track（跟踪）**——基于评估更新进度向量 p^k。若 s_k 表明当前步骤 resolved（Correct 或 Comprehension）：  
**p^k[t] ← 1**  
否则向量不变。

**Record（记录）**——将当前轮 archive 进对话历史：  
**H^k ← H^{k−1} ∥ (L_k, s_k, T_k)**

**表 1：学习者状态到四操作的映射**

| 状态 | Assess | Act | Track | Record |
|------|--------|-----|-------|--------|
| Start | 初始化会话 | 分析并 pose q₁ | p⁰← | H 初始化 |
| Correct | 验证与 a_t 对齐 | 确认并 transition 至 q_{t+1} | p[t]←1 | append H |
| Comprehension | 表明理解 | acknowledge 并 transition | p[t]←1 | append H |
| Incorrect | 检测错误或 mismatch | scaffold/引导 self-correction | p[t] 不变 | append H |
| Confusion | 识别缺乏理解 | 提供 hint/简化 | p[t] 不变 | append H |
| Question | 识别澄清 query | 回答并 steer back | p[t] 不变 | append H |
| Irrelevant | 检测 off-topic | refocus 注意力 | p[t] 不变 | append H |
| End | 验证最终解 | 结束会话 | p[N]←1 | finalize H |

**会话流程摘要：** 学习者提出 Q 时，ScaffoldLM 首先构造 P(Q)。会话以 p⁰=0、H⁰=∅ 初始化。每轮识别当前步骤 t，assess 学习者，选策略生成指导，更新 p 中完成状态，在 H 中 record 交互。p 全为 1 时会话结束。

---

## 5 数据构建与训练

### 5.1 苏格拉底数据合成

**一致性 enforced 双智能体模拟**  
通过 orchestrating 学习者智能体与导师智能体间 dynamic 过程 synthesize realistic 交互，二者 grounded 于 Q 与 P(Q)。会话以学习者 pose 问题 (L₁=Q) 开始，导师初始化 memory s₁=Start 并 pose 初始子问题 q₁。后续轮 k≥2，学习者智能体 sample target 认知状态 s_k（排除 Start/End）并基于当前步骤答案 a_t 生成响应 L_k。为确保数据 validity，生成导师监督前 enforce **consistency check**：导师智能体验证 L_k 是否与 sampled s_k 一致。若失败，拒绝 L_k，导师返回 brief rationale 作为 feedback，学习者智能体在相同 s_k 下 regenerate L_k。仅 consistency verified 后导师生成最终训练 targets：Rationale（详述评估证据与 intended scaffolding action）、identified s_k、实际 T_k。循环直至 p 完成，设 target s=End，导师生成 concluding summary。

**过滤与序列化**  
会话级 filtering 丢弃过长或 logically incoherent 轨迹。valid 轨迹 τ flatten 为单轮训练样本：输入为当前 tutoring memory（含计划、进度、历史）与学习者话语；target 输出为“analysis-first”结构：rationale、identified state、final response。得到：

**D_tutor = {((M^{k−1}, L_k), (Rationale_k, s_k, T_k))}**

**教学计划生成**  
从原始数学题集合 D_raw = {(Q, A_ref)} 开始。LLM planner 先产生详细 rationale R(Q)，再提取分步计划 P(Q)。过滤 pipeline：(1) 步骤数 N ∈ [2,7]；(2) 导出最终答案 a_N 须匹配 A_ref；(3) 外部 LLM judge 验证步骤转移逻辑 validity。高质量样本形成规划数据集：

**D_plan = {(Q, P(Q))}**

### 5.2 指令微调

**D_train = D_plan ∪ D_tutor**

为两种数据类型设计 distinct system instructions，使模型在会话开始 autonomous 生成教学计划（D_plan），随后 engage adaptive 多轮交互（D_tutor）。混合 D_plan 与 D_tutor 训练 jointly 支持计划生成与多轮辅导。D_plan 中 reasoning-intensive 监督 empirically 有助于 maintain 一般数学解题性能。

---

## 6 实验

### 6.1 数据集与 Baseline

**数据集构建**  
实验基于 **BigMath**（Albalak 等，2025）。聚焦单答案问题，难度分层采样得 10,000 训练题与 300 held-out 评估题。规划模型与导师智能体用 **DeepSeek V3**，学习者智能体用 **Doubao-Seed-1.6-Flash**。过滤后：5,635 规划样本，49,815 单轮辅导样本（来自 5,844 会话，平均 8.5 轮）。

**Baselines**  
**ScaffoldLM-7B**：在 Qwen2.5-7B-Instruct 上微调混合数据集。对比：(1) **通用 LLM**：Qwen2.5-7B/72B-Instruct、DeepSeek-V3、GPT-4o（prompt 辅导）；(2) **专用辅导 LLM**：SocraticLM、TutorRL-7B、InnoSpark-turbo、EduChat-7B。

### 6.2 评估

三方面评估：

**(1) 教学能力（Pedagogical Ability）**  
用 GPT-4o-mini 作为 automated judge 评估六维度（与 expert 教师 79.6% exact-match 一致，见附录 D.3）。前五维（答案准确率、分步脚手架、主题 adherence、问题质量、指导质量）在 300 held-out BigMath 上与学习者模拟器多轮对话后评分。第六维**自适应反馈**在 human verified 静态数据集上评估（210 样本，Correct/Incorrect/Question 各 70）。

**(2) 答案泄露分析（Solution Leakage）**  
遵循 TutorRL 协议报告 **Δ Solve Rate (%)**（对话后学习者解题率提升）与 **Leaked Solution (%)**（导师直接 reveal 解的比例）。

**(3) 通用推理（General Reasoning）**  
在 MATH500、OlympiadBench、TheoremQA 等 out-of-domain 数据集报告结果，确保辅导 specialization 不 compromise 一般推理。

---

## 7 结果

### 7.1 主结果

**表 2：ScaffoldEval 辅导表现（节选）**

| 模型 | 答案 Acc. | 分步 Scaff. | 主题 Adh. | 问题 Qual. | 指导 Qual. | 自适应反馈 (Correct/Incorrect/Question) |
|------|-----------|-------------|-----------|------------|------------|----------------------------------------|
| Qwen2.5-7B-Instruct | 0.682 | 0.912 | 0.801 | 0.818 | 0.314 | 0.682 / 0.747 / 0.410 |
| GPT-4o | 0.763 | 0.931 | 0.813 | 0.782 | 0.485 | 0.831 / 0.817 / 0.713 |
| SocraticLM | 0.562 | 0.652 | 0.810 | 0.682 | 0.332 | 0.710 / 0.882 / 0.304 |
| TutorRL-7B | 0.682 | 0.953 | 0.968 | 0.789 | 0.281 | 0.714 / 0.757 / 0.486 |
| **ScaffoldLM-7B (Ours)** | **0.785** | **0.988** | **0.999** | **0.830** | **0.525** | **0.841 / 0.892 / 0.729** |

ScaffoldLM 在 6 项中 5 项最佳。源于双模块框架：**分步教学计划**确保 structured 推理，**Tutoring Memory** 优化实时指导。explicit grounding 于 pre-computed 计划与实时 state tracking，超越 RL-aligned 导师与更大通用模型。

**动态评估赋能 robust 错误纠正：** 表 2 右侧按学习者状态分解自适应反馈。多数 baseline 处理 explicit 问题好但 fail 识别错误——72B 模型 Incorrect 仅 0.613。ScaffoldLM-7B Incorrect 最高 **0.729**，优于 DeepSeek-V3 与 GPT-4o。归因于 Tutoring Memory 内 **Assessment 机制**——explicit 将学习者响应与计划 target 比较以 infer 认知状态。

### 7.2 消融研究

**表 3：核心组件消融**

| 设置 | 答案 Acc. | 分步 Scaff. | 主题 Adh. | 问题 Qual. | 指导 Qual. |
|------|-----------|-------------|-----------|------------|------------|
| ScaffoldLM-7B | 0.785 | 0.988 | 0.999 | 0.830 | 0.525 |
| w/o Plan | 0.753 | 0.965 | 0.987 | 0.812 | 0.468 |
| w/o Assessment | 0.750 | 0.972 | 0.978 | 0.803 | 0.425 |
| w/o Both | 0.723 | 0.925 | 0.981 | 0.795 | 0.392 |

- **Planning** 确保 structured 指导；移除显著降低指导质量与答案准确率。  
- **Assessment** enable 动态自适应；static 计划 insufficient，须 active 跟踪学习者状态决定 proceed 或 corrective hints。

### 7.3 答案泄露分析

**图 2** 绘制 Δ Solve Rate 对 Leaked Solutions。ScaffoldLM-7B 达 **30.6%** solve rate 仅 **8.8%** leakage，显著优于 TutorRL baseline (λ=0.5: 30.9% solve, 25.1% leak)。structured 框架提供 intermediate 指导 targets，assist 学习者而不 reveal 最终答案——即使未 explicit 优化 reward trade-off，incorporating 教学结构仍 attain 更 favorable 平衡。

### 7.4 通用推理能力

**表 4：推理 benchmark（准确率 %）**

| 模型 | Math-500 | OlympiadBench | TheoremQA |
|------|----------|---------------|-----------|
| Qwen2.5-7B-Instruct | 77.2 | 39.9 | 47.5 |
| ScaffoldLM (Ours) | 76.0 (−1.2) | 40.4 (+0.5) | 47.2 (−0.3) |

ScaffoldLM-7B 与 backbone 性能 comparable，仅 marginal 差异，表明 scaffolded 微调成功 instilled 教学行为而不 compromise  fundamental 数学推理。

---

## 8 结论

本文提出 ScaffoldLM——面向多轮数学辅导的规划引导框架，facilitate 分步脚手架指导与 cognition-aligned 苏格拉底辅导。ScaffoldLM 首先生成分步教学计划作为 stable 交互 backbone，随后用 assessment 与动态 tutoring memory 跨轮跟踪进度、interpret 学习者响应并选择 appropriate scaffolding actions。经 plan-aligned 辅导合成监督数据训练，ScaffoldLM 在实验中 demonstrate 高于 baseline 的整体 tutoring 质量，highlight **plan-aware 指导**与 **memory-aware 自适应**在数学辅导中的价值。

---

## 局限性

(1) 模型 primarily 在 logic-intensive 领域（数学）synthetic 数据上训练；synthetic 性质可能无法 fully capture 真实学生交互 unpredictability，框架对 open-ended 或 subjective 学科的 applicability 待探索。  
(2) 当前框架 solely 依赖模型 internal 表示进行推理与计算，无 external tools；未来 aim integrate 代码解释器等 explicit 工具验证 intermediate 步骤。

---

## 伦理声明

**数据集许可：** BigMath (Apache 2.0)；MATH、OlympiadBench、TheoremQA (MIT)；TutorRL 测试集 (CC-BY-4.0)。  
**Human 参与：** 多名 trained annotator 与两名中学教师；按 institutional 规范 compensated；未收集 PII。

---

## 附录摘要

### A 案例研究（图 5）

完整多轮示例展示 predefined 计划步骤如何 derive 每道导师问题，系统基于学生响应 dynamic adapt 反馈——学生犯错时提供 corrective 反馈与 scaffolding 再 proceed。

**示例问题：** 求方程「某数 × 4 × 2 − 72 = 9938」中的初始数。  
**计划步骤：** (1) 4×2=? (36) → (2) 36×2=? (72) → (3) 如何 isolate 初始数 (加 72 到 9938) → (4) 9938+72=? (10010)

### B 实现细节

基于 ms-swift；Qwen2.5-7B-Instruct full-parameter SFT 2 epochs，bfloat16，max seq 12,288 tokens，Flash Attention 2；8× NVIDIA A800 80GB，DeepSpeed ZeRO-3；global batch 64，lr 1×10⁻⁵ cosine decay。

### C 数据集细节

- BigMath 分层采样：80% 高难度 (solve rate < 0.6)，20% 低难度  
- 规划阶段：7,353 valid plans → verifier 接受 5,635 用于训练 planning  
- 辅导模拟：学习者状态采样 P = (0.5 Correct, 0.2 Incorrect, 0.1 Question, 0.1 Comprehension, 0.05 Confusion, 0.05 Irrelevant)  
- 最终：5,844 对话，49,815 轮，平均 8.5 轮

### D 评估细节

**ScaffoldEval 六维度（表 5 摘要）：**
- **答案准确率**：导师最终推导与参考答案语义等价  
- **分步脚手架**：多轮交互无 premature 答案泄露  
- **主题 adherence**：从 irrelevant 回复 steer 回问题  
- **问题质量**：3=高认知需求 probing，2=中等，1=低，0=无问题  
- **指导质量**：相对上一轮提供新有效 pedagogical 内容  
- **自适应反馈**：Understanding + Feedback 分数

GPT-4o-mini 作为 judge 与 teacher-gold 一致率 79.6%。

**表 7 答案泄露完整结果（节选）：** ScaffoldLM-7B Δ Solve 30.6%，Leak 8.8%，Ped-RM 4.7/4.7；优于多数 TutorRL 配置与通用大模型。

### E Prompt 模板（图 6–14）

附录含完整英文 prompt，涵盖：
- **图 6**：规划 prompt——将数学题 decompose 为 3–7 子问题  
- **图 7**：Verifier prompt——检查 final-answer 一致性、步骤对齐、粒度、期望答案正确性  
- **图 8–9**：双智能体合成中 Tutor/Learner system prompt  
- **图 10–11**：学习者 Correct 响应与 Tutor consistency check + scaffolding transition  
- **图 12**：对话级 filtering（premature leakage、逻辑不一致、退化文本）  
- **图 13**：多轮评估用标准化苏格拉底 tutoring system prompt  
- **图 14**：每轮后 termination check（[end] / [continue]）

**表 8–9**：规划与辅导阶段训练样本格式模板（含 [Analysis and Decision]、[Learner State]、[Reply] 结构）。

---

## 参考文献

（完整列表见原文 PDF，主要包括 Liu 等 2024 SocraticLM；Dinucu-Jianu 等 2025 TutorRL；Dan 等 2023 EduChat；Black & Wiliam 1998 形成性评估；Bakker 等 2015 数学脚手架；Albalak 等 2025 BigMath 等。）

---

*本译文基于 ACL Anthology 正式出版 PDF 全文翻译，技术术语保留必要英文并附中文说明。Prompt 附录因篇幅以摘要呈现，完整英文原文见 https://aclanthology.org/2026.acl-long.325.pdf 第 7179–7188 页。*
