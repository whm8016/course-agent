# IntelliCode：基于集中式学习者建模的多智能体 LLM 辅导系统

**作者：** Jones David¹, Shreya Ghosh²  
¹ VIT-AP 大学计算机科学与工程学校，印度  
² 印度理工学院布巴内斯瓦尔分校电气与计算机科学学院，印度  
**通讯：** jones.22bce8135@vitapstudent.ac.in, shreya@iitbbs.ac.in  

**出处：** EACL 2026 系统演示论文（第 129–138 页）  
**原文 PDF：** https://aclanthology.org/2026.eacl-demo.10.pdf  
**arXiv：** https://arxiv.org/abs/2512.18669  

---

## 摘要

基于 LLM 的辅导系统通常是单轮助手，缺乏对学习者知识的持久表示，难以提供有原则、透明且长期的教学支持。我们提出 IntelliCode——一种围绕**集中式、版本化的学习者状态**构建的多智能体 LLM 辅导系统，该状态整合掌握度估计、 misconceptions（错误概念）、复习计划与参与度信号。StateGraph 编排器协调六个专用智能体：技能评估、学习者画像、分级提示、课程选择、间隔重复与参与度监控；每个智能体在**单写者策略**下作为共享状态上的纯变换运行。该架构支持可审计的掌握度更新、 proficiency-aware（ proficiency 感知）提示、依赖感知的课程自适应与安全对齐的提示。演示展示端到端辅导流程：学习者尝试一道 DSA 题，卡住时获得概念性提示，提交修正后的解答，并立即看到掌握度更新与个性化复习间隔。我们在模拟学习者上报告验证结果，显示状态更新稳定、分级提示提升任务成功率，以及多样化的课程内容覆盖。

**在线系统：** https://intellicode.redomic.in  
**视频演示：** https://youtu.be/oO8bZfeleOU  

---

## 1 引言

大语言模型（LLM）迅速拓展了自动化辅导的可能性，但多数现有系统本质上仍是**反应式**的：每个查询被孤立处理，几乎不延续或感知学习者 evolving 的知识（Wu 等，2025）。相比之下，人类导师维护丰富的、持久的学生理解模型，以支持针对性反馈、课程脚手架与长期学习轨迹（VanLEHN，2011）。这一差距限制了当前 LLM 辅导的教学可靠性——它们常给出不一致的提示、忽视概念间依赖，且无法应对系统性错误概念。

GenMentor（Wang 等，2025）与 SocraticLM（Liu 等，2024）等近期框架展示了多智能体编排与结构化对话在稳定辅导行为方面的潜力。然而，这些系统通常依赖**短暂或隐式记忆**，缺乏跨智能体共享的、显式且可审计的学习者模型。同时，从贝叶斯知识追踪（BKT）（Corbett & Anderson，1995）、深度知识追踪（DKT）（Piech 等，2015）到记忆网络（Zhang 等，2017）与 PFA（Pavlik Jr 等，2009），数十年学习者建模工作都强调准确掌握度估计对个性化的重要性。但很少有基于 LLM 的辅导系统将此类形式化模型与生成式推理集成到持久、可更新的状态中。

我们的目标是弥合这一差距。我们提出 IntelliCode——围绕**集中式、版本化学习者状态**构建的多智能体 LLM 辅导系统，该状态作为所有教学决策的**单一事实来源**。与先前系统不同，IntelliCode 通过 StateGraph 编排器强制执行**单写者策略**，确保每次掌握度更新、提示干预或课程选择都来自对学习者模型的一致、形式化验证的变换。该设计减轻漂移、防止智能体输出冲突，并实现透明的多轮个性化。

IntelliCode 整合成熟的教学原则——掌握度估计、分级提示、依赖感知课程规划与间隔重复——与现代 LLM 能力。学习者状态编码带不确定性的掌握向量、错误概念、复习计划与行为信号。六个专用智能体（技能评估、学习者画像、教学反馈、内容策展、进度综合与参与度编排）均作为该共享状态上的纯函数运行。例如，若学习者在递归题中反复遗漏 base case，画像智能体记录错误概念，反馈智能体调整提示级别，策展智能体相应调整后续任务。

### 贡献

本工作贡献如下：（1）集中式、版本化学习者状态与单写者编排机制，实现一致、可审计的多轮辅导；（2）一套教学智能体，将掌握度估计、分级提示、课程规划与间隔重复实现为共享状态上的纯变换；（3）功能完整的端到端 LLM 辅导系统，通过模拟学习者研究展示稳定交互、可解释决策与稳健内容覆盖。

图 1 突出驱动自适应策略的统一学习者状态。通过将智能体行为 grounded 于该结构化表示，同时利用 LLM 推理处理提示与代码分析等高方差任务，IntelliCode 为无记忆对话式辅导提供了透明且教学一致的替代方案。

---

## 2 系统架构

我们将自适应个性化教育建模为**部分可观测马尔可夫决策过程（POMDP）**。在每个时间步，学习者状态 S_t 维护掌握向量、基于 SM-2 的复习计划、参与度指标与元认知记忆。观测 O_t 反映提交、错误、提示请求等噪声行为信号；动作 A_t 对应内容推荐、分级提示、计划调整等有教学意义的干预。奖励函数 R_t 在掌握度增益与过度提示使用、低效解题时间的惩罚之间权衡（细节见附录 B）。该形式化支持 principled 自适应，并与演示中的模块化多智能体设计相契合。

**表 1：六个教学智能体的角色与职责**

| 智能体 | 职责 |
|--------|------|
| 教学反馈（Pedagogical Feedback） | 提供 proficiency 感知、五级 graduated hinting，不泄露完整解答 |
| 内容策展（Content Curator） | 基于掌握度、依赖关系与 40/50/10 课程策略选择个性化题目 |
| 参与度编排（Engagement Orchestrator） | 监控动机、节奏与脱离信号，发出支持性 nudge |
| 技能评估（Skill Assessment） | 混合评估：测试用例执行 + 语义代码审查 |
| 学习者画像（Learner Profiler） | 估计掌握度增量、识别错误概念、推断行为趋势 |
| 进度综合（Progress Synthesizer） | 用增强 SM-2 机制安排复习，含上下文感知调整 |

### 2.1 编排器概览

IntelliCode 核心是 **StateGraph 编排器**——唯一允许写入持久学习者记录的组件。它维护学习者状态的同步内存副本，协调六个教学智能体间的所有交互。当事件发生时，编排器将其路由至相应智能体，聚合输出，验证提议的状态变更，然后作为**原子更新**提交。与 GenMentor（Wang 等，2025）的协调策略类似，该机制防止冲突写入、强制执行安全与 schema 约束，并确保系统在多轮、长期辅导会话中行为可预测。编排器 thus 为 principled 学习者建模提供可靠性与可审计性。

### 2.2 触发类型与路由

编排器响应有教学意义的事件并分派给相关智能体。这些触发器 operationalize 系统中演示的完整工作流：

- **on_submission**：代码或答案提交触发技能评估，随后学习者画像与教学反馈。
- **on_hint_request**：学习者求助触发教学反馈智能体， informed 于当前 proficiency 与提示历史。
- **on_session_check**：每日签到触发内容策展与参与度编排。
- **on_daily_generation**：系统通过内容策展生成当日个性化题集。
- **on_review_due**：SM-2 复习到期时调用进度综合与内容策展。

这些触发器使 IntelliCode 能自适应响应学习者 evolving 的行为，在会话间保持教学连续性。

### 2.3 系统智能体概览

每个智能体作为共享学习者状态上的纯变换运行，产生编排器验证并整合的结构化输出。六个智能体共同支持评估、个性化、节奏、提示与复习调度。表 1 总结其职责，图 2 展示编排器如何 mediate 通信。该设计确保所有教学决策 grounded 于一致、可审计的学习者状态，并在整个辅导轨迹中保持可追溯。

整体数据流见图 2。StateGraph 编排器 mediate 智能体与持久学习者状态间的所有通信，确保教学决策在会话间可追溯且一致。

---

## 3 学习者状态与智能体自适应

IntelliCode 维护**集中式、版本化学习者状态**， governing 所有教学决策。状态由历史活动初始化，并用受 BKT 启发的机制更新，纳入难度、新近性、提示使用与解题时间效应。间隔重复由增强 SM-2 调度器管理，基于回忆质量与交互历史计算个性化复习间隔。该共享表示 enable 一致、长期自适应，而非孤立单轮响应。

为说明更新过程，考虑学习者正确解一道递归题但请求多次提示且超过预期解题时间。BKT 启发更新赋予 modest 掌握度增益，因更依赖提示而 attenuated；进度综合器安排更早复习以强化 retention。内容策展器随后将递归 interpret 为学习者“成长区”并调整未来选题。

### 3.1 智能体行为与教学逻辑

六个智能体均作为学习者状态上的纯变换运行，产生编排器验证后作为原子更新提交的结构化输出。架构支持 fully generative 智能体，但为可复现性，学习者画像与内容策展使用确定性逻辑，而提示、代码分析等高方差组件 leverage LLM。

#### 学习者画像

画像智能体是系统诊断 backbone，识别掌握度增量、错误概念及疲劳、速度下降等行为趋势。它消费正确性、主题标签、错误模式、任务时间、提示使用与当前掌握度图。例如，若学习者反复在递归题中遗漏 base case，画像记录与终止条件相关的错误概念，后续指导提示与内容选择。

#### 技能评估

该智能体通过执行测试用例并进行语义代码审查进行混合评估。测试失败时仅 surface 错误；通过时提供时间复杂度、空间使用、可读性、边界覆盖等改进建议。例如，merge sort 实现成功后，可能建议减少辅助内存以提升空间效率。

#### 教学反馈

受 Socratic Playground（Zhang 等，2024）启发，教学反馈智能体采用**五级 graduated hinting** 协议：

1. **元认知**： prompt 反思（“你尝试了什么？发生了什么？”）
2. **概念**： surface 关键思想（“本题依赖识别递推关系。”）
3. **策略**： suggest 方法（“考虑分解输入并递归求解两半。”）
4. **结构**： highlight 缺失逻辑（“你的解缺少空输入的 base case。”）
5. ** targeted**：指向感兴趣区域（“检查第 14 行附近条件；终止可能无法保证。”）

提示 specificity 随 proficiency 估计 p̂ 缩放。初学者获简单类比与单步 cue，中级获模式导向指导，高级获 concise nudge 与边界 emphasis。同一递归 bug，初学者可能被提示“把递归想成下梯子”，高级者可能被 prompt“检查终止条件是否可达”。

#### 内容策展

策展器用依赖感知的 **40/50/10 策略** operationalize 学习者状态：

**selection = 0.4 × due_reviews + 0.5 × growth_zone + 0.1 × challenge**

成长区项对应掌握度 0.3–0.7；挑战项针对低于 0.3 的技能。策展器 enforce 先修依赖、避免 k 天内重复、确保主题多样性。例如，递归掌握中等但 DP 弱的学习者可能收到：（i）递归复习题，（ii）中等难度 DP 子问题，（iii）轻量 DP 挑战题。

#### 进度综合

进度综合器用 SM-2（Wozniak，1990）与遗忘曲线理论（Ebbinghaus，1913）， augmented 上下文特征（Settles & Meeder，2016；Reddy 等，2016）govern 间隔重复。 heavily 使用提示时复习间隔缩短；快速自信解题时扩展；预测 recall 在到期日附近下降时收紧。若图遍历概念 recall 概率低于阈值，即使学习者近期未交互也会提前复习。

#### 参与度编排

参与度编排器监控动机信号。连续失败后发出支持性 prompt，不活跃期后鼓励 re-engagement，失败 streak 累积时 suggest 更简单变体。所有干预 rate-limited 且措辞 non-judgmental。例如，树题多次失败后系统可能 suggest：“是否想在重试前先 revisit 较易的‘二叉树基础’练习？”

 collectively，这些智能体构成 IntelliCode 自适应引擎，将每个提示、选题与复习决策 grounded 于统一、可审计的学习者状态。

---

## 系统演示

IntelliCode 平台后端用 FastAPI，前端 React，多智能体编排 LangGraph。持久、图结构学习者模型存于 ArangoDB，跨会话跟踪掌握度、错误概念与复习计划。

演示展示完整自适应辅导环。学习者从课程路线图开始，由内容策展器用 40/50/10 策略分配 DSA 题。若卡住，教学反馈产生 proficiency 对齐提示——例如 Level 2 概念 cue：“本题需要识别递推模式。”学习者 incorporate 提示后提交正确解。提交触发协调智能体行为序列：技能评估验证正确性并提供语义反馈；学习者画像更新递归掌握度估计；进度综合器安排两天后 spaced-repetition 复习，反映提示使用与解题时间 profile。界面允许查看掌握度轨迹、 upcoming 复习与历史交互，使自适应过程透明。该端到端交互 exemplify IntelliCode 如何将实时评估、分级提示、课程自适应与间隔重复整合为 coherent、状态驱动教学 cycle。

---

## 4 评估协议

我们从离线、在线与公平性维度评估 IntelliCode，展示学习者建模、内容自适应与多智能体交互的可靠性。

### 离线指标

离线分析 focus 验证学习者模型 fidelity。用 Brier 分数与 ECE 测量**掌握度校准**——预测掌握度 m̂_t 与后续正确性 y_{t+1} 的相关。**内容策略**跟踪主题覆盖与多样性，30 天 horizon 目标至少 90% 覆盖。**调度质量**评估 recall 预测器准确率（AUROC ≥ 0.75）并验证 SM-2 复习到期 adherence。

### 在线指标

跟踪 live 交互表现。**学习增益**通过 held-out 评估前后掌握度变化估计。**参与度**通过 streak retention、不活跃 gap、自愿练习率监控。**系统效率**用中位端到端延迟评估，目标低于 500 ms。**安全**通过提示接受率（目标 ≥ 70%）及验证系统从不泄露完整解评估。

### 公平性分析与消融

为确保不同 profile 间公平支持，比较 proficiency 十分位间的学习增益、提示级别与节奏行为，目标四分位距在中位数 15% 内。消融研究隔离内容策展、教学反馈、上下文感知 SM-2 调度器等关键组件贡献。

**图 4–6** 分别展示：按初始技能水平的平均掌握度增益；有无提示利用的成功率；模拟期间覆盖的前 10 主题。

### 模拟学习者验证

为评估多智能体架构 responsiveness 与稳定性，我们用 agent-based 学习者 persona（Wu 等，2025）进行 preliminary 模拟，方法借鉴 generative agent societies（Park 等，2023）。虽不能替代 human 研究评估教育 efficacy，但提供系统对 diverse 认知 profile 与交互模式 coherent 响应的 initial 验证。

### 4.1 响应性与提示有效性

十条模拟学习轨迹中，IntelliCode 动态调整任务难度与提示行为，平均掌握度增益 5.04%（图 4）。graduated hinting 也 robust：请求提示的任务成功率 89.1%，整体 baseline 52.4%（图 5）。确认教学反馈智能体有效 intervening——提供概念指导而不泄露解——同时保持架构可靠性。

### 4.2 内容覆盖

内容策展器在主题间保持强多样性（图 6），展示 40/50/10 策略避免主题 starvation 同时 respect 先修关系。 extended 会话中系统保持技能领域 balanced 覆盖，验证编排器协调长期学习 arc 与课程 progression 的能力。

---

## 5 结论

本文呈现 IntelliCode——围绕持久、可审计学习者状态构建的 principled 多智能体 LLM 自适应教育框架。通过整合形式化掌握度更新规则、proficiency 感知 graduated hinting、依赖与公平感知课程自适应及安全对齐提示，系统提供透明一致的多轮辅导能力。模拟展示稳定架构行为、有效提示干预与 robust 内容覆盖，highlight 方法的技术 viability。我们 envision IntelliCode 作为下一代教育系统 foundation， blend 现代 LLM 推理与学习者建模、 instructional design 与认知科学 established 原则。

---

## 局限性

IntelliCode 虽展示 promising 架构与教学能力，仍有局限。First，掌握度估计依赖 BKT/DKT 启发 proxy，需 sufficient 交互规模与 careful 校准；cold-start 学习者需 conservative 先验，早期会话 personalization 可能降低。Second，LLM 驱动组件因 model drift、偶发 refusal 与成本引入 variability；guardrails 与 validation schema mitigate 但无法完全消除。Finally， rigorous 公平评估需 diverse 代表性数据集，仍 vulnerable 于 selection bias、行为信号噪声、LLM drift、数据泄漏与 survivorship bias。这些 underscore 未来大规模 longitudinal human 研究需求。

---

## 伦理考量

LLM 智能体在教育场景部署需 careful 关注准确性、依赖与隐私。虽 IntelliCode 整合技能评估等 verifier 验证代码逻辑，教学反馈等生成组件仍 susceptible 于 hallucination 或 plausible 但错误解释。因此系统设计为** supplemental tutor** 而非正式 instruction 替代，建议在 human 教育者指导下使用以 monitor 潜在偏差。

为 mitigate 学习者对 AI  assistance 过度依赖，我们 implement strict graduated hinting 协议。但 acknowledge 长期依赖 automated scaffolding 可能影响 unassisted 解题能力。设计 prioritize 元认知 prompting 而非直接泄露解以 foster genuine 技能 acquisition。

数据隐私方面，所有学习者交互以 strict minimization 处理。PII 在智能体 ingestion 前 redacted，学习者画像 operate 于 anonymized 掌握度整数而非 raw 用户 profile。Finally，虽课程策略 incorporate 公平约束确保 equitable 主题覆盖，cold-start 校准所用底层数据集可能 inherently 反映 historical bias，需 ongoing 监控 diverse  demographic 群体学习结果。

---

## 附录 A 资源与可用性

为支持可复现与 further 研究，我们在 MIT 许可下开放平台组件与模拟数据：

- **Live Demo：** https://intellicode.redomic.in
- **Simulation Framework：** https://github.com/Redomic/intellicode_student_sim

---

## 附录 B 数学形式化细节

自适应个性化教育形式化为 POMDP：

**POMDP = (S, A, O, T, R, γ, b₀)**

### B.1 状态空间

时刻 t 学习者状态：

**S_t = {m_t, r_t, e_t, p_t, M_t, v_t}**

其中：
- **m_t**：掌握向量，m_{t,i} ∈ [0,1]，主题 i ∈ T
- **r_t**：复习计划，项含 (q_id, topics, d_due, interval, EF, n_reviews)
- **e_t**：参与状态，streak、last-seen 时间戳、近期活动窗口
- **p_t**：偏好，技能水平、模态、时间预算、opt-outs
- **M_t**：长期记忆，趋势、错误概念、洞察的结构化文本节
- **v_t**：版本、时间戳（审计用）

每主题 uncertainty u_{t,i} 编码为 Beta 参数 (α_{t,i}, β_{t,i})。

### B.2 观测空间

观测为状态的 partial、噪声信号：

**O_t ∈ {submission, hint_request, session_start, due_review}**

每次提交观测：

**o_t = (q_id, y, τ, h_cnt, errors, t_solve)**

y ∈ {0,1}（过/败），τ 时间戳，h_cnt 提示数，errors 语义信号，t_solve 任务时间。

### B.3 动作空间

编排器（经智能体）选择动作：

**A_t ∈ {recommend_item, hint(l), adjust_schedule, intervene, feedback(d)}**

l ∈ {1,2,3,4,5} 为提示级别，d 为反馈细节级别。

### B.4 奖励函数

奖励 proxy 学习进展同时惩罚 inefficiency：

**R_t = w_m Δm_t + w_r 1[review_success] − w_h h_cnt − w_t max(0, t_solve − μ_t)**

含公平（主题覆盖）与参与度 regularizer。

---

## 附录 C 学习者状态更新

### C.1 状态初始化

由历史提交计算初始掌握度，用 recency 加权指数移动平均：

**m_{t,i}^{(0)} = 0.6 × success_rate_i + 0.4 × recent_success_rate_i + N(0, σ₀²)**

Beta 参数初始 (α₀, β₀) = (1,1)（无信息先验），复习队列为空，ease factor EF₀ = 2.5。

### C.2 掌握度更新规则

对 tagged 主题 Q 上结果 y ∈ {0,1}：

成功时：**m_{t,i} ← min(1, m_{t,i} + α w_d w_r (1 − m_{t,i}))**  
失败时：**m_{t,i} ← max(0, m_{t,i} − β w_d^{-1} w_r m_{t,i})**

其中 w_d ∈ {0.8, 1.0, 1.2} 映射 Easy/Medium/Hard；w_r = exp(−Δt/τ_upd) 随 recency 衰减；提示/时间惩罚与 momentum smoothing 减少 jitter。

### C.3 Proficiency 综合

整体 proficiency：**p̂ = Σ_k w_k s_k**

s_k 含：主题掌握平均（0.40）、 expertise rank（0.25）、自报技能（0.20）、近期成功率（0.10）、streak 归一化（0.05）。

### C.4 间隔重复更新（SM-2）

质量分 q ∈ {0,…,5}：5=快速无提示，4=解出 minor delay，3=有提示解出，≤2=失败/遗忘。

**EF' = max(1.3, EF − 0.8 + 0.28q − 0.02q²)**

间隔 I₁=1，I₂=6，I_n = round(I_{n−1} · EF') 天。预测 recall：**R(Δt) = exp(−Δt/τ)**，τ ∝ EF'。

---

## 附录 D 可复现细节

1. Seeded random splits；固定 topic-graph 快照  
2. 冻结 prompt 版本与 role 文本  
3. 记录超参：α, β, w_d, τ_upd, λ，proficiency 权重  
4. 智能体输出的 validation schema（JSON spec）  
5. 行为信号预处理与 masking 规则  
6. 消融代码与离线评估脚本  

---

## 参考文献

（原文引用列表见 PDF；主要包括 Corbett & Anderson 1995 BKT；Piech 等 2015 DKT；Liu 等 2024 SocraticLM；Wang 等 2025 GenMentor；Wu 等 2025 模拟学生；Park 等 2023 Generative Agents；Zhang 等 2024 Socratic Playground 等。）
