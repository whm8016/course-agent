# 数据库表设计宪法 + 31 表体检 + 收尾路线图（已核销）

> 本文是 `course-agent` 后端数据库 schema 的**设计规范（"宪法"）**与**现状体检**。
> 所有新表、迁移、重构都应以本文 7 条原则为准绳。2026-08-08 审计产出的 P0–P4 收尾路线图**已全部治理完成**（2026-09-16 复核核销，状态见 Part 3）。
> 模型定义单文件：`backend/core/db/database.py`；迁移：`backend/alembic/versions/`（001→037，缺 007）。

---

## 0. TL;DR

- 32 张表 / 10 个域，对标 Moodle（468 表）属**健康范围**——表多本身不是病。
- 分层方向（事件→rollup→展示、FK/no-FK 故意分裂）已是**业界中上水准**。
- 2026-08-08 体检发现的 3 类病（缺 `course_id` / 双真相源 / 死 rollup）经 P0–P4 整改**已全部治理**；当前体检 **27 ✅ / 4 ⚠️ / 0 🔴**，无结构级违规。
- 2026-08 后新增 4 表（`course_faq` / `course_topic` / `course_topic_edge` / `mastery_observation`）已入册体检，全部合规。
- 剩余 4 个 ⚠️ 均为"可改进/有理由保留"（见 Part 2 要点列），不是欠账；新表仍必须 7 原则全合规。

---

## 1. 三维调研依据

| 维度 | 关键结论 | 出处 |
|---|---|---|
| **范式 vs 反范式** | 范式保一致性+写好（OLTP），反范式保读快（OLAP/报表）。按**访问模式分区**决定，非二选一。 | [ByteByteGo](https://blog.bytebytego.com/p/database-schema-design-simplified) · [CodiLime](https://codilime.com/blog/normalization-vs-denormalization-in-databases/) |
| **事件溯源 / Event+State** | 纯 ES=只追加事件表为真相源；**Event+State 双表（CQRS 读模型）**更适合生产——事件表存历史、状态表存当前快照。 | [Azure 架构中心](https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing) · [SE: 事件表+触发器写状态表](https://softwareengineering.stackexchange.com/questions/431186/event-sourcing-inside-a-rdbms) |
| **多租户隔离** | 三档：库-per-租户 / schema-per-租户 / 共享表+行级。本项目用最便宜那档（`course_id` 裸字符串行级隔离），隔离靠**查询纪律**。 | [Azure SQL SaaS 模式](https://learn.microsoft.com/en-us/azure/azure-sql/database/saas-tenancy-app-design-patterns) |
| **表演进** | **Expand/Contract**：先扩(新旧并存双写)→迁数据→再缩(删旧)。拆/合表走过渡期，不一刀切。 | [Ambler 重构数据库](https://scottambler.github.io/refactoring-databases/) · [Expand/Contract](https://newsletter.systemdesignclassroom.com/p/refactoring-databases-is-a-different-animal) |

**论文**：[mem0, arXiv:2504.19413](https://arxiv.org/html/2504.19413v1) — 生产级长期记忆（动态抽取→合并→检索，向量库+图库双存）。本项目 L3 记忆层（episodic / mastery / mem0）直接对标。

**顶级项目**：Moodle（468 表，[ER 图](https://www.dbdiagrams.com/mysql/online-er-diagram-moodle/)）、Canvas LMS（[schema](https://databasesample.com/database/canvas-lms)、[Red Gate LMS 建模](https://www.red-gate.com/blog/database-design-management-system/)）——LMS 表多只要分层清晰就不算病。

---

## Part 1 — 表设计宪法（7 条原则）

每条 = **规则 · 为什么 · 正例 · 反例（历史违规，均标注治理落点）**。新表必须全部合规。

### 原则 1：分层归属（一概念一真相源）

每张表归属且仅归属一层：

| 层 | 特征 | 本项目表 |
|---|---|---|
| ① 原始事件 | 只追加，内容永不 UPDATE | `messages` `learning_events` `memory_episodes` `llm_usage_records` `mastery_observation` `turn_errors` |
| ② 派生读模型 | cron 重算（delete-then-reinsert），从不 read-modify-write | `course_daily_rollup` `student_course_rollup` `llm_usage_daily` `course_faq` |
| ③ 可变状态 | 频繁 UPDATE 的当前事实 | `sessions` `kb_builds` `knowledge_mastery` `research_checkpoints` |
| ④ 引用/配置 | 读多写少 | `knowledge_bases` `enrollments` `course_topic` `course_topic_edge` `user_*` 配置表 |

**规则**：一个业务概念只能有**一个权威存储**，其余只能是它的投影/缓存（且必须能从权威源重算）。

**为什么**：双真相源必然漂移；查"KB 状态"不知信谁、查"用户图谱"两个地方不一致。

**正例**：`kb_builds` 是 KB 构建状态的唯一真相源（`UNIQUE(kb_id, backend)`），KB 行的展示状态由 `aggregate_build_status(kb.builds)` 现算投影。
**历史反例（已治理）**：`knowledge_bases` 行上曾残留死状态列与 `kb_builds` 重复 → 迁移 028 删列收口；`users.knowledge_graph` 曾与 `knowledge_mastery` 双写 → 改判保留为展示层 + OCC（迁移 031，见附录决策 2）。

### 原则 2：读写分离（CQRS：事件 → rollup 读模型）

**规则**：原始层只 INSERT、内容永不 UPDATE；rollup 层全派生（delete-then-reinsert），从不 read-modify-write。**读路径只读 rollup，不现算**。

**为什么**：现算=每请求 JOIN 风暴；rollup=O(1) 读 + 后台批量算。

**正例**：`llm_usage_records`(原始) → `llm_usage_daily`(派生)，已被 teacher/admin 端点消费；`learning_events`(原始) → `course_daily_rollup` / `student_course_rollup`(派生) → analytics 端点消费，行缺失/过期降级现算兜底。
**历史反例（已治理）**：两张 rollup 曾 cron 在算却无读路径（P0 已接通）；`learning_events` 曾缺 `verb=answered`，答题口径退回 JOIN `NotebookEntry ⋈ Session`（已由 `api/question_notebook.py` 提交作答即写接通）。

### 原则 3：多租户就地隔离（行级 `course_id`，写时落盘）

**规则**：共享表 + 行级隔离档下，**凡是课程级可查的行，必须在写入时就把 `course_id` 落盘**，使查询无需 JOIN 即可确定租主。

**为什么**：靠 JOIN Session 反查 `course_id` 已致一次跨租户 bug（`teacher.py` 旧实现仅按 user_id 全局聚合，把他课程的答题也算进本课程）。

**正例**：`knowledge_mastery`（`UNIQUE(user_id, course_id, kp_id)`）、`messages`、`notebook_entries` 均带写时落盘的 `course_id` + 复合索引。
**历史反例（已治理/已拍板）**：`messages` / `notebook_entries` 缺 `course_id` → 迁移 026 补列+回填+索引；`bot_notifications` 拍板不加（仅按 `user_id` 查询，YAGNI，附录决策 3）。

> **注**：不新建 `courses` 表（course 与 `knowledge_bases` 1:1 已工作；给 13 张表的 `course_id` 改真 FK 是巨大 Expand/Contract 却只换边际收益）。治跨租户病的正确姿势是补 `course_id` 列，而非建父表。

### 原则 4：FK 纪律（故意分裂，写进 docstring）

**规则**：业务表用真 FK + `ondelete=CASCADE`（数据随父死亡）；分析/计费/检查点表用**裸字符串无 FK**（历史须在用户/课程删除后存活）。**每个选择都要在模型 docstring 写明理由**。

**为什么**：账单 `llm_usage_records` 若 CASCADE，删用户=删账单，计费数据不可回溯。

**正例（已合规）**：`llm_usage_records` / `llm_usage_daily` / `research_checkpoints` 无 FK，docstring 写明"须在用户删除后存活"。`messages`/`enrollments`/`notebook_*` 等业务表用 CASCADE。本项目此项**已达标**，仅需制度化。

### 原则 5：并发一致性（可变列要 OCC 或逐行 append）

**规则**：凡在并发下被整体 rewrite 的可变列，必须有 **OCC 版本号**或改成**逐行 append**。裸 read-then-全量 rewrite JSON = 丢更新 bug。

**为什么**：两个 `consolidate_memory` job 同用户并发，会各自 read→rewrite→互相覆盖。

**正例**：`sessions.summary_version`（OCC：`UPDATE ... WHERE summary_version = old`）防多 worker 摘要丢更新；`users.graph_version`（迁移 031，条件 UPDATE + 冲突重试）防图谱整列 rewrite 丢更新；`knowledge_mastery` 逐行 append + 读时衰减。
**历史反例（已治理）**：`users.knowledge_graph` / `error_graph` 无版本号、由 `graph_memory.save_graphs` 整列 rewrite → 迁移 031 加 OCC（治理方式从原案"删列"改判为"OCC 保留"，附录决策 2）。

### 原则 6：演进纪律（Expand/Contract，不留孤儿，迁移唯一权威）

**规则**：
1. schema 变更走 **Expand → Migrate → Contract**，"保留不删"必须挂一个 Contract 工单。
2. 生产建表只经 Alembic；create_all 仅作 dev/test 基座（拍板保留，附录决策 4）。
3. 每个迁移有 `revision`/`down_revision` 链，文件名与 revision 一一对应。

**为什么**：双轨曾致 rev 014 补建 7 张早已"活在 create_all 里"的表、每个迁移裹 `_table_exists/_column_exists` 幂等守卫；"保留不删"欠下 6 个孤儿记忆列的债。

**历史反例（已治理/已拍板）**：`users` 6 个孤儿记忆列 → 4 个已删（迁移 029），余 2 个图谱列改判保留+OCC（031）；`create_all` 双轨 → **拍板保留**（生产路径跳过 create_all，漂移由 greenfield/head 链迁移守卫测试兜底，附录决策 4）。

### 原则 7：约定一致（ID / 时间戳 / 命名）

固化现有约定，新表照办：

| 项 | 约定 |
|---|---|
| 主键 | `String(32)` 短 UUID（`_short_uuid`），少数流水表用自增 `Integer` |
| 时间戳 | `Float`（Unix 秒），`default=time.time`；**不用** datetime |
| 保留字 | `metadata` → Python 属性 `metadata_` |
| 租主键 | `course_id: String(64)` 裸字符串（**非** FK，见原则 3/4） |
| `user_id` | `String(32)`，对齐 `users.id` |
| 复合索引 | 前缀放高基数 + 常过滤列（如 `(course_id, updated_at)`、`(user_id, course_id, created_at)`） |
| 1:1 配置表 | `user_id` 唯一 FK + `nullable=False` |

**正例**：现状高度一致，本项**已达标**。

---

## Part 2 — 31 表体检打分

图例：`✅合规` / `⚠️可改进（或有理由保留）` / `🔴违反原则`

| 域 | 表 | 判定 | 要点 |
|---|---|---|---|
| Auth | `users` | ⚠️ | 4 死记忆列已删（迁移 029）、图谱列 OCC 已修（031）；余 `knowledge_graph`/`error_graph` 保留为**仪表盘展示层**（每生对话现编图），与 `knowledge_mastery`（掌握状态）、`course_topic_edge`（全班课程结构）三方分工已澄清 |
| | `user_social_bindings` | ✅ | 标准 OAuth 绑定 |
| | `user_mcp_enrollments` | ✅ | 1:1 配置 |
| | `user_search_configs` | ✅ | 1:1 配置 |
| | `user_llm_providers` | ⚠️ | 1:1 配置合规；`fast_model` 是死字段 |
| | `teacher_invites` | ✅ | 一次性码流程 |
| | `teacher_applications` | ✅ | 审批流程，索引合理 |
| Chat | `sessions` | ⚠️ | OCC 摘要做对；`updated_at` 每条消息 UPDATE（有界，留意索引 churn） |
| | `messages` | ✅ | `course_id` 已补（迁移 026，写时落盘+复合索引）；037 加 `turn_id` 唯一索引（服务端权威回合日志幂等） |
| RAG | `knowledge_bases` | ✅ | 7 个死状态列已删（迁移 028）；构建状态唯一真相源=`kb_builds`，展示态由 `aggregate_build_status` 现算投影 |
| | `kb_builds` | ✅ | 真相源，`UNIQUE(kb_id,backend)` 合理 |
| | `kb_files` | ✅ | |
| Quiz | `notebook_entries` | ✅ | `course_id` 已补（迁移 026）；036 加判分列（grade/grade_source/feedback/score） |
| | `notebook_categories` | ✅ | |
| | `notebook_entry_categories` | ✅ | M2M join |
| Academic | `enrollments` | ✅ | 租户 JOIN 枢纽（符合预期） |
| | `course_schedules` | ✅ | 课表 |
| | `grades` | ✅ | upsert 键合理 |
| Memory | `memory_episodes` | ✅ | 原始+outbox 双职，append-only |
| | `knowledge_mastery` | ✅ | 逐行 append + 读时衰减，P3/P5 范例 |
| | `mastery_observation` | ✅ | 新增（033）：append-only 观测日志，字段契约对齐 pyBKT tutor log；状态由 `fold_observations` 确定性折叠进 `knowledge_mastery`——换估计器只重跑折叠不重跑 LLM；`evidence_quote` 判定依据可审计 |
| | `course_topic` | ✅ | 新增（032）：课程结构层，读多写少，建库时一次性离线抽取；`uq(course_id, topic_id)`；`topic_id` 对齐 `knowledge_mastery.kp_id`；难度侧 `difficulty_b`/`n_item` 日频全班重折、写路径不在线更新 |
| | `course_topic_edge` | ✅ | 新增（032）：先修边唯一事实源，`uq(course_id, src, dst, relation)`；`evidence` 讲义原句（子串校验过）可审计，闭包推断边无 evidence；回溯按 dst 查有索引 |
| Analytics | `learning_events` | ⚠️ | `asked`/`answered` 已接通（answered 由 question_notebook 提交作答即写，rollup 在吃）；`feedback` 待 Phase 4 |
| | `course_daily_rollup` | ✅ | 读路径已接（teacher 概览 7 天趋势），行缺失/过期 >6h 降级现算（P0 收口） |
| | `student_course_rollup` | ✅ | 读路径已接（学生统计 2-5 项），stale 降级（P0 收口） |
| | `course_faq` | ✅ | 新增（027）：L1 读模型，ARQ cron 语义聚类删后重算（幂等）；embedding 存 JSON 非 pgvector——SQLite 测试可跑、Python 算 cosine（避开 `func.left` 同款 PG 专有坑） |
| Billing | `llm_usage_records` | ✅ | append-only + 无 FK，P4 范例 |
| | `llm_usage_daily` | ✅ | 被 teacher/admin 消费；Redis 配额失联由 `reconcile_quota_from_db` 从本表聚合回填兜底（P4.2） |
| Research | `research_checkpoints` | ✅ | 无 FK 故意，合理 |
| Bot | `bot_notifications` | ✅ | 每日 04:53 清理 N 天前**已读**通知（`purge_old_notifications`）；`course_id` 拍板不加（仅按 `user_id` 查询，YAGNI，附录决策 3） |

**统计**：**27 ✅ / 4 ⚠️ / 0 🔴**（⚠️=users 展示层保留 / user_llm_providers 死字段 / sessions updated_at churn / learning_events feedback 未落）。2026-08-08 首检逐表实为 16✅/4⚠️/7🔴（当时统计行误作 18/4/9，本次修正为逐表可加总口径）。

---

## Part 3 — 收尾路线图（P0–P4 全部核销，2026-09-16 复核）

原路线图按 ROI/风险排序分 5 阶段，各阶段独立交付。以下为各阶段的**病 → 实际落地（含改判）→ 验证**存档。

### P0 ｜ 接通死 rollup + 补全事件层 —— ✅ 已完成

- **病**：`course_daily_rollup` / `student_course_rollup` 每小时 DELETE+重算却无人读；analytics 端点每请求现算（6 条串行查询）；`learning_events` 缺 `verb=answered`。
- **落地**：analytics 概览/学生统计读 rollup（`api/teacher.py`），行缺失/过期 >6h **降级现算**兜底；`verb=answered` 由 `api/question_notebook.py` 提交作答即写，rollup 答题口径不再依赖 `NotebookEntry ⋈ Session`。
- **验证**：`tests/test_teacher_analytics.py`（含"rollup 读路径接通（P0 收口）"回归）。

### P1 ｜ 租主硬化：补 `course_id` —— ✅ 已完成（2/3 表，第 3 表拍板不做）

- **病**：`messages` / `notebook_entries` / `bot_notifications` 缺 `course_id`，课程级查询必 JOIN Session，已致跨租户 bug。
- **落地**：前两表迁移 026 补列（NOT NULL default ""）+ 复合索引 + 关联子查询回填（PG/SQLite 通用）；写时落盘（`memory.py add_message` / `notebook_store.upsert` 反查 Session.course_id）；5 处查询去 JOIN。`bot_notifications` 拍板不加（附录决策 3）。
- **验证**：41 项相关测试绿。⚠️ 测试 fixture 直造 NotebookEntry 须补 `course_id` 否则查不到（已沉淀为测试约定）。

### P2 ｜ KB 状态单一真相源 —— ✅ 已完成

- **病**：`knowledge_bases` 残留 7 死状态列（status/progress*/chunks_*/token_estimate/error_msg）与 `kb_builds` 重复。
- **落地**：迁移 028 DROP 7 列；KB 真相源收口 `kb_builds`；`_ensure_column` 死条目与 seed 传参同步清理（漏删=dev/prod schema 分叉，真实漂移 bug 已修）。
- **验证**：`test_kb_builds` / `test_migration_greenfield` 绿（greenfield 有"无死列"守卫断言）。

### P3 ｜ 记忆单一真相源 + users 瘦身 + 并发 —— ✅ 已完成（含一次改判）

- **病**：①图谱双写 `users.knowledge_graph`（无 OCC 整列 rewrite）↔ `knowledge_mastery`；②`users` 6 个记忆列多为孤儿（语义记忆已交 mem0）。
- **落地**：迁移 029 删 4 死列（summary/profile/scope/preferences_memory）。图谱列**原案"删列"被推翻**：前端 GraphPage/仪表盘/学生统计直接渲染该列，删列=拆功能，用户拍板"修 bug 别删"→ 迁移 031 加 `graph_version` OCC（条件 UPDATE+冲突重试 3 次，modify 幂等）+ 新节点打 `course_id` 标签 + 教师视图按课程过滤。随后 `course_topic_edge`（032）落地，"全班课程结构（可审计唯一事实源）"与"每生展示图"分工正式澄清。
- **验证**：仪表盘图谱渲染正常；OCC 冲突重试有测试。
- **教训**：①删列前 grep 零读者零写者（含 seed 构造传参）②"前端渲染原始 JSON 列"是删列硬阻塞③并行加迁移会撞号，加迁移先 `ls versions/`。

### P4 ｜ 建表纪律 + 配额回填 —— ✅ 已完成（create_all 双轨拍板保留）

- **病**：①双轨建表（`init_db()` create_all + Alembic）；②配额真相只在 Redis，Redis 丢=静默重置；③迁移文件卫生（缺 revision 007、`005_` 文件重名）。
- **落地**：配额回填 `reconcile_quota_from_db`（`core/quota/cost_quota.py`，`main.py` 启动调用）从 `llm_usage_daily` 聚合 SETNX 回填 Redis 配额键；迁移文件改名补注释。kill create_all **实证不可行**（002 create_unique_constraint SQLite 不支持；create_all 是 900+ 测试基座）→ **拍板保留双轨**：生产路径跳过 create_all（`init_db` 守卫），漂移由 greenfield/head 链迁移守卫测试兜底（附录决策 4）。
- **验证**：`test_cost_quota.py`（Redis 失联回填）；`test_migration_greenfield.py`（空库 `alembic upgrade head` 全表齐）。

---

## 附录：四个已拍板决策

1. **不新建 `courses` 表**——course 与 `knowledge_bases` 1:1 已工作，建父表要改 13 张表的 `course_id` 为 FK，ROI 不划算；治跨租户病用"补 `course_id` 列"（`knowledge_mastery` 范式）。
2. **图谱列保留 + OCC（改判）**——原案"P3 删列而非加 OCC"于 2026-08 下旬推翻：前端 GraphPage/仪表盘/学生统计直接渲染 `users.knowledge_graph`，删列=拆功能；用户拍板"修 bug 别删"，走迁移 031 `graph_version` OCC。随后 `course_topic_edge` 落地，全班课程结构与每生展示图分工澄清，"双真相源"降级为"展示层 + 状态层"。
3. **`bot_notifications` 不加 `course_id`**——首检体检表 over-flag：该表只按 `user_id` 查询，加列是 YAGNI。留存清理已另行落地（每日 04:53 `purge_old_notifications`，只清已读）。
4. **`create_all` 双轨保留**——kill 实证不可行（SQLite 不支持 002 的 create_unique_constraint；create_all 是 900+ 测试基座）。收口方式：生产路径跳过 create_all + greenfield/head 链迁移守卫测试兜漂移，而非强行单轨。

---

*2026-08-08 审计产出（三维调研 + 27 表体检 + P0–P4 路线图）；2026-09-16 复核刷新：P0–P4 全部核销（Part 3 存档状态）、4 张新表入册（`course_faq`/`course_topic`/`course_topic_edge`/`mastery_observation`）、体检重打分 27✅/4⚠️/0🔴、迁移对齐 001→037。**2026-09-19 第 32 表入册：`turn_errors`（迁移 038，错误处理与可观测性重整）——失败 turn 的 append-only 运维流水，与 messages（学生叙事流）两概念两真相源；无 FK 裸字符串（原则 4，用户/课程删除后存活）；消费方 `GET /admin/errors/recent?ref=`（学生报障精确查）。*
