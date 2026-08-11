# ReAct：在大语言模型中协同推理与行动

**ReAct: Synergizing Reasoning and Acting in Language Models**

> 发表于 ICLR 2023  
> arXiv:2210.03629v3 [cs.CL] 2023年3月10日  
> 项目主页与代码：https://react-lm.github.io/

**作者**  
Shunyu Yao*¹, Jeffrey Zhao², Dian Yu², Nan Du², Izhak Shafran², Karthik Narasimhan¹, Yuan Cao²

¹ 普林斯顿大学计算机科学系  
² Google Research, Brain 团队

---

## 摘要

尽管大语言模型（LLM）在语言理解与交互式决策等任务上表现突出，其**推理**能力（如思维链提示）与**行动**能力（如动作计划生成）此前大多被分开研究。本文探索让 LLM **交替生成推理轨迹与任务相关动作**，使二者产生更大协同：推理轨迹帮助模型诱导、跟踪、更新行动计划并处理异常；行动则让模型与外部知识库或环境交互以获取额外信息。我们将该方法命名为 **ReAct**，并在多种语言与决策任务上验证其有效性：在问答（HotpotQA）与事实验证（Fever）上，ReAct 通过与简单 Wikipedia API 交互，缓解思维链中常见的幻觉与错误传播，生成更可解释的人类式解题轨迹；在 ALFWorld 与 WebShop 两个交互决策基准上，ReAct 以仅 1–2 个上下文示例提示，分别比模仿学习与强化学习方法绝对成功率提高 **34%** 与 **10%**。

---

## 图 1：四种 prompting 方法对比（HotpotQA + ALFWorld）

![图1 第1页](react_paper_images/figure_page_1.png)

**图 1 说明（中文）**

- **(1) HotpotQA**：对比 (a) 标准提示、(b) 思维链 CoT（仅推理）、(c) 仅行动 Act-only、(d) ReAct（推理+行动）。
- **(2) ALFWorld**：对比 (a) 仅行动 vs (b) ReAct。
- 轨迹中：`Thought/Act` 为模型生成，`Obs` 为环境反馈。提示中省略了上下文示例，仅展示模型实际生成轨迹。

**HotpotQA 示例问题（图1右上）**  
> 除 Apple Remote 外，还有什么设备能控制 Apple Remote 原本设计交互的程序？

| 方法 | 关键行为 | 结果 |
|------|----------|------|
| 标准 | 直接答 iPod | ❌ 错误 |
| CoT | 逐步想：Apple Remote 控制 Apple TV；Apple TV 可由 iPhone/iPad/iPod Touch 控制 | ✅ 正确但可能幻觉 |
| Act-only | 搜索 Apple Remote → Front Row → Front Row software | ❌ 未整合信息得出答案 |
| **ReAct** | 思考→搜索→观察→再思考→搜索 Front Row software→得出 keyboard function keys | ✅ 正确且可核查 |

**ALFWorld 示例任务**  
> 把胡椒粉 shaker 放到 drawer 上。

ReAct 会先 **Think** 分解子目标（找 shaker 可能位置：cabinet 1–6、countertop 1–3…），再依次 `go to`、`take`、`put`；Act-only 则在 sinkbasin 反复 `take` 失败。

---

## 1 引言

人类智能的一个独特之处，是能将**面向任务的行动**与**语言推理**（内心独白）无缝结合。这种“行动—推理”紧密协同，使人类能在未知情境或信息不确定时快速学习并稳健决策。

近期工作分别探索了 LLM 的推理与交互决策，但二者结合不足：

- **思维链（CoT）**：多步推理能力强，但是静态黑盒，不接地外部世界 → 易幻觉、错误传播（见图 1-1b）。
- **语言模型作策略（WebGPT、SayCan 等）**：能生成动作/计划，但很少用语言做高层目标推理或工作记忆维护。

**ReAct** 让 LLM **交替生成语言推理轨迹与动作**，实现：
- **推理以行动（reason to act）**：动态创建、维护、调整高层计划；
- **行动以推理（act to reason）**：与 Wikipedia 等环境交互，把外部信息纳入推理。

### 主要贡献

1. 提出 ReAct：协同推理与行动的通用 prompting 范式；
2. 在 HotpotQA、Fever、ALFWorld、WebShop 上系统实验，少样本下优于单独推理或单独行动；
3. 消融分析：推理任务中行动的价值、交互任务中推理的价值；
4. 分析 prompting 局限，并做初步微调实验。

---

## 2 ReAct：协同推理 + 行动

### 2.1 形式化

智能体在时刻 t 接收观察 `o_t`，在上下文 `c_t` 下选动作 `a_t`。

ReAct 将动作空间扩展为 **Â = A ∪ L**，其中 **L** 为语言空间。语言动作 `â_t ∈ L`（称为 **thought / 推理轨迹**）**不改变外部环境**，无观察反馈；其作用是推理当前上下文并更新 `c_{t+1} = (c_t, â_t)`，支持后续推理或行动。

有用 thought 的类型包括：
- 分解目标、制定计划；
- 注入常识；
- 从观察中提取关键信息；
- 跟踪进度、切换计划；
- 处理异常、调整计划。

### 2.2 实现

- 使用冻结大模型 **PaLM-540B**（附录 A.1 有 GPT-3 结果）；
- **少样本 in-context 示例**：人工轨迹含 action、thought、observation；
- **知识密集型任务**（HotpotQA/Fever）：**密集** thought，严格 **Thought → Action → Observation** 交替；
- **决策任务**（ALFWorld/WebShop）：thought **稀疏**，由模型自行决定何时思考。

### 2.3 ReAct 特点

| 特点 | 说明 |
|------|------|
| A) 直观易设计 | 标注者只需在动作旁写自然语言思考 |
| B) 通用灵活 | 适配 QA、事实验证、文本游戏、网页导航等 |
| C) 性能稳健 | 1–6 个示例即可泛化，跨域一致优于纯推理/纯行动 |
| D) 人类对齐 | 轨迹可解释；可通过编辑 thought 实时纠正行为（图 5） |

---

## 3 知识密集型推理任务

### 3.1 实验设置

**数据集**
- **HotpotQA**：多跳问答，需跨 ≥2 篇 Wikipedia；
- **FEVER**：事实验证，标签 SUPPORTS / REFUTES / NOT ENOUGH INFO。

**设定**：仅给问题/claim，不给支持段落；模型靠内部知识或通过环境检索。

**Wikipedia API 动作空间**
1. `search[entity]`：返回实体 Wiki 页前 5 句，或相似实体建议；
2. `lookup[string]`：页内查找含 string 的下一句（模拟 Ctrl+F）；
3. `finish[answer]`：提交答案。

### 3.2 方法

**ReAct 提示**：HotpotQA 6 例、Fever 3 例（更多示例无益）。

**基线**
- **Standard**：去掉 thought/action/observation；
- **CoT**：仅推理；
- **CoT-SC**：21 条 CoT 自洽投票（temperature=0.7）；
- **Act**：去掉 thought。

**ReAct + CoT-SC 组合**
- **ReAct → CoT-SC**：ReAct 步数用尽仍未答 → 回退 CoT-SC（HotpotQA 7 步，Fever 5 步）；
- **CoT-SC → ReAct**：CoT-SC 多数票不足 n/2 → 回退 ReAct。

**微调**：用 ReAct 生成的 3000 条正确轨迹微调 PaLM-8B/62B。

### 3.3 结果

#### 表 1：PaLM-540B 在 HotpotQA / Fever 上的 prompting 结果

| 方法 | HotpotQA (EM) | Fever (Acc) |
|------|:-------------:|:-----------:|
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC | 33.4 | 60.4 |
| Act | 25.7 | 58.9 |
| **ReAct** | **27.4** | **60.9** |
| CoT-SC → ReAct | 34.2 | **64.6** |
| ReAct → CoT-SC | **35.1** | 62.0 |
| 监督 SOTA | 67.5 | 89.5 |

**要点**
- ReAct > Act：推理指导行动，尤其最终答案综合；
- ReAct vs CoT：Fever 上 ReAct 更好（需精确检索）；HotpotQA 略低（CoT 推理结构更灵活）；
- **最佳组合**：HotpotQA 用 ReAct→CoT-SC；Fever 用 CoT-SC→ReAct。

#### 图 2：CoT-SC 样本数 vs 性能

![图2](react_paper_images/figure_page_5.png)

ReAct+CoT-SC 用 **3–5** 个 CoT-SC 样本即可接近 21 样本 CoT-SC 性能。

#### 表 2：HotpotQA 成功/失败模式（人工标注 200 例）

| 类型 | 定义 | ReAct | CoT |
|------|------|:-----:|:---:|
| **成功-真阳性** | 推理与事实均正确 | 94% | 86% |
| **成功-假阳性** | 推理或事实幻觉 | 6% | 14% |
| **失败-推理错误** | 含重复步骤无法恢复 | 47% | 16% |
| **失败-检索错误** | 搜索空或无信息 | 23% | - |
| **失败-幻觉** | 幻觉推理/事实 | 0% | 56% |
| **失败-标签歧义** | 预测合理但与标注不完全匹配 | 29% | 28% |

**关键观察**
- CoT 幻觉严重（失败中 56%）；ReAct 更接地、可核查；
- ReAct 结构约束导致推理错误率更高（47% vs 16%），含重复 thought/action 循环；
- ReAct 检索失败占 23%，会打断推理链。

#### 图 3：HotpotQA 规模效应（prompt vs finetune）

![图3](react_paper_images/figure_page_6.png)

- **Prompting**：ReAct 在 8B/62B 上最难（需同时学推理+行动）；
- **Finetune 3000 例后**：ReAct 成为最佳；PaLM-8B 微调 ReAct > 全部 62B prompting；62B 微调 ReAct > 全部 540B prompting。

#### 图 4：过时标签示例（ReAct 获最新答案）

![图4](react_paper_images/figure_page_14.png)

**问题**：Circue du Soleil 秀 Mystère 所在酒店有多少房间？  
**标注**：2,664（已过时）  
**ReAct 检索**：Treasure Island Hotel and Casino 有 **2,884** 房间 + 220 套房 → 答案 **3,104** ✅

---

## 4 决策任务

### 4.1 ALFWorld

文本家庭环境，6 类任务（如 examine paper under desklamp）。专家策略可 >50 步。

- ReAct 提示：每任务类型 3 条轨迹，含稀疏 thought（分解目标、跟踪子目标、常识定位物品）；
- 对比 **BUTLER**（10⁵ 条专家轨迹模仿学习）；
- 134 个未见游戏，6 组 prompt 排列做鲁棒性测试。

#### 表 3：ALFWorld 各任务成功率（%）

| 方法 | Pick | Clean | Heat | Cool | Look | Pick2 | **All** |
|------|:----:|:-----:|:----:|:----:|:----:|:-----:|:-------:|
| Act (best/6) | 88 | 42 | 74 | 67 | 72 | 41 | 45 |
| ReAct (avg) | 65 | 39 | 83 | 76 | 55 | 24 | 57 |
| **ReAct (best/6)** | **92** | **58** | **96** | **86** | **78** | **41** | **71** |
| ReAct-IM (best/6) | 62 | 68 | 87 | 57 | 39 | 33 | 53 |
| BUTLER (best/8) | 46 | 39 | 74 | 100 | 22 | 24 | 37 |

ReAct 最佳试验 **71%**，Act **45%**，BUTLER **37%**；相对 Act 平均提升 **62%**。

### 4.2 WebShop

118 万真实商品、1.2 万购物指令。评估 **Score**（属性覆盖率）与 **SR**（完全满足率）。

#### 表 4：WebShop 结果

| 方法 | Score | SR |
|------|:-----:|:--:|
| Act | 62.3 | 30.1 |
| **ReAct** | **66.6** | **40.0** |
| IL | 59.9 | 29.1 |
| IL+RL | 62.4 | 28.7 |
| Human | 82.1 | 59.6 |

ReAct SR 比先前最佳高 **10** 个百分点。

### 4.3 内部推理 vs 外部反馈（ReAct vs Inner Monologue）

**Inner Monologue (IM)**：thought 主要是环境状态与当前子目标的外部反馈复述。

**ReAct-IM 消融**：用 IM 式密集外部反馈 thought → ALFWorld 总体 **53%** vs ReAct **71%**。

ReAct 优势：能判断子目标何时完成、下一子目标是什么、用常识推断物品位置。

#### 图 5：ALFWorld 人机协同 thought 编辑

![图5](react_paper_images/figure_page_15.png)

**任务**：把两把钥匙放进保险箱。

- **(a) 原始 ReAct**：Act 17 幻觉“第二把钥匙在 drawer 4”（实际只有 watch）→ 失败；
- **(b) 人工编辑 thought**：删除 Act 17 幻觉句；Act 23 加入提示“更可能在 dresser/garbagecan/safe…” → 成功。

人类只需改 2 句 thought，无需改数十个动作。

---

## 5 相关工作

### 语言模型推理
- **CoT**（Wei et al., 2022）及 least-to-most、zero-shot-CoT、自洽（CoT-SC）等；
- **Selection-Inference**、**STaR**、**Faithful reasoning**、**Scratchpad** 等多步架构。

ReAct 区别：不仅孤立固定推理，还把**动作与观察**纳入输入流，支持交互决策。

### 语言模型决策
- **WebGPT**：浏览器交互，依赖昂贵人类反馈 RL，无显式推理；
- **BlenderBot / Sparrow / SimpleTOD**：对话 API 决策，需大量标注；
- **SayCan / Inner Monologue**：机器人规划；IM 的“内心独白”更像外部反馈而非灵活推理。

ReAct：用自然语言描述推理过程，**成本更低**的 policy 学习。

---

## 6 结论

ReAct 以简单方式协同 LLM 的推理与行动，在多跳 QA、事实核查、交互决策上取得更好性能且轨迹可解释。

**局限**：大动作空间需更多演示，易超 in-context 长度；微调初探有前景，但仍需更多高质量标注。

**未来**：多任务扩展、与 RL 结合、更好解码（beam search 缓解重复循环）。

---

## 附录 A：额外结果

### A.1 GPT-3 实验（表 5）

| 任务 | PaLM-540B | GPT-3 (text-davinci-002) |
|------|:---------:|:------------------------:|
| HotpotQA EM | 29.4 | **30.8** |
| ALFWorld SR% | 70.9 | **78.4** |

ReAct prompting 跨模型有效。

### A.2 ReAct 获取 HotpotQA 最新知识

见 **图 4**（上文）。

### A.3 ALFWorld 人机协同

见 **图 5**（上文）。

---

## 附录 B：实验细节

### B.1 HotpotQA 微调
- batch size 64；
- PaLM-8B：ReAct/Act 4000 步，Standard/CoT 2000 步；
- PaLM-62B：ReAct/Act 4000 步，Standard/CoT 1000 步；
- Standard/CoT 微调过久会退化。

### B.2 ReAct-IM
IM 式 thought 仅含：(1) 分解当前目标；(2) 当前待完成子目标。  
缺少：子目标完成判断、下一子目标规划、常识定位。

---

## 附录 C：提示词（节选翻译）

### C.1 HotpotQA — ReAct 示例 1

**问题**：Colorado orogeny 东段延伸区域的海拔范围是多少？

```
Thought 1: 我需要搜索 Colorado orogeny，找到东段延伸区域，再查该区域海拔范围。
Action 1: Search[Colorado orogeny]
Observation 1: Colorado orogeny 是科罗拉多及周边造山运动…
Thought 2: 未提及东段，需 Lookup[eastern sector]。
Action 2: Lookup[eastern sector]
Observation 2: 东段延伸至 High Plains，称 Central Plains orogeny。
Thought 3: 东段延伸至 High Plains，需搜索 High Plains 的海拔。
Action 3: Search[High Plains]
Observation 3: High Plains 可指两个不同区域…
Thought 4: 应搜索 High Plains (United States)。
Action 4: Search[High Plains (United States)]
Observation 4: 从东到西海拔约 1,800 到 7,000 ft。
Thought 5: 答案是 1,800 to 7,000 ft。
Action 5: Finish[1,800 to 7,000 ft]
```

完整 6 例见原文附录 C.1（英文）。

### C.2 FEVER — ReAct 示例

**Claim**：Nikolaj Coster-Waldau 曾与 Fox Broadcasting Company 合作。

```
Thought 1: 搜索 Nikolaj Coster-Waldau，确认是否与 Fox 合作。
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: …曾出演 Fox 电视电影 Virtuality (2009)…
Thought 2: 因出演 Fox 作品，应与 Fox Broadcasting Company 有合作。
Action 2: Finish[SUPPORTS]
```

### C.3 WebShop（表 6）

指令：*3 ounce bright citrus deodorant for sensitive skin, price < $50*

**ReAct 额外步骤**：
```
Action: think[B078GWRC1J 和 B078GTKVXY 是 bright citrus 且 <50 美元，先查 B078GWRC1J]
Action: think[该商品有 'bright citrus' 和 '3 ounce (pack of 1)' 选项，可以购买]
```

![WebShop 提示对比 第22页](react_paper_images/figure_page_22.png)

### C.4 ALFWorld — ReAct vs Act vs ReAct-IM

**任务**：put a clean lettuce in diningtable

| 方法 | 关键差异 |
|------|----------|
| **Act** | 直接 go/open/take/clean/put，仅在末尾一条 think |
| **ReAct** | 每步 think：分解任务→推测 lettuce 位置→找到后 take→去 sink 洗→放回 |
| **ReAct-IM** | think 重复“First I need to find a lettuce”，缺少进度跟踪与常识 |

完整 prompt 见原文 Table 7–9。

---

## 附录 D：轨迹示例（节选）

### D.1 FEVER

**例 1951（标注 REFUTES，CoT 错答 SUPPORTS）**  
Claim: Soyuz was part of the American space program.

- **ReAct**：搜 Soyuz → 再搜 American space program → 未提及 Soyuz → **NOT ENOUGH INFO** ✅（保守）
- **CoT**：Soyuz 是俄罗斯飞船，NASA 与俄罗斯合作 ISS → **SUPPORTS** ❌（幻觉）

### D.2 ALFWorld — pick_clean_then_place knife

**ReAct**：系统搜索 cabinet/drawer/countertop → 在 countertop 2 找到 knife → sink 清洗 → 放到 countertop 1 ✅

**Act**：找到刀后未先去 sink，直接 `clean knife with sinkbasin` → Nothing happens → 陷入重复循环 ❌

**ReAct-IM**：缺少“已拿到刀，下一步去 sink”的 thought，行为类似 Act 失败。

---

## 附录 E：失败模式示例

见原文附录 E.1（HotpotQA ReAct/CoT 各类成功失败案例）。

---

## 可复现性

- 主实验基于 **PaLM-540B**（非开源）；
- 附录 C 含全部 prompt；
- 附录 A.1 含 **GPT-3** 实验；
- GPT-3 ReAct 代码：https://react-lm.github.io/

---

## 伦理声明

ReAct 轨迹更可解释、可诊断、可控。但 LLM 接外部动作空间（网页、物理环境）存在检索不当信息或危险行为风险。本实验限制在 Wikipedia/WebShop 研究基准，无真实购买或 Wiki 编辑。

---

## 参考文献（原文 42 篇，节选）

1. Wei et al. (2022) — Chain-of-Thought Prompting  
2. Wang et al. (2022a) — Self-Consistency  
3. Nakano et al. (2021) — WebGPT  
4. Ahn et al. (2022) — SayCan  
5. Huang et al. (2022b) — Inner Monologue  
6. Yang et al. (2018) — HotpotQA  
7. Thorne et al. (2018) — FEVER  
8. Shridhar et al. (2020b) — ALFWorld  
9. Yao et al. (2022) — WebShop  
10. Zelikman et al. (2022) — STaR  

完整列表见 PDF 第 10–12 页。

---

## 全论文页面图片索引

所有 33 页高清渲染图位于：`react_paper_images/pages/`

| 页码 | 内容 |
|:----:|------|
| page_01.png | 标题、摘要、**图1** |
| page_02.png | 引言续 |
| page_03.png | **第2节 ReAct 方法** |
| page_04–05.png | 第3节、**表1、图2** |
| page_06.png | **表2、图3** |
| page_07–08.png | 第4节、**表3–4** |
| page_09.png | 第5–6节、结论 |
| page_10–12.png | 参考文献 |
| page_13–15.png | 附录 A、**图4–5** |
| page_16–21.png | 附录 C 提示词 |
| page_22.png | WebShop **表6** |
| page_23–24.png | ALFWorld **表7–9** |
| page_25–33.png | 附录 D 轨迹 |

---

## 核心公式与伪代码（中文）

```
循环直到 finish 或达到最大步数:
    1. Thought_t  ← LLM(上下文, "Thought:")
    2. Action_t   ← LLM(上下文 + Thought_t, "Action:")
    3. Observation_t ← Environment(Action_t)   # 若 Action ∈ A
    4. 将 (Thought_t, Action_t, Observation_t) 追加到上下文
```

**与 CoT 区别**：CoT 只有 Thought→Answer；ReAct 插入可改变外部状态的 Action 与真实 Observation。

**与 Act-only 区别**：Act 无 Thought，难以分解目标、纠正检索、综合最终答案。

---

*本译文根据 ICLR 2023 官方 PDF 完整翻译，图表来自原论文页面渲染。仅供学习研究使用。*
