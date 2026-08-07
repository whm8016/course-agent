# 数据库表设计宪法 + 27 表体检 + 收尾路线图

> 本文是 `course-agent` 后端数据库 schema 的**设计规范（"宪法"）**与**现状体检**。
> 所有新表、迁移、重构都应以本文 7 条原则为准绳；已有的违规项见 [Part 3 收尾路线图](#part-3--收尾路线图expandcontract分阶段可中断)，按 P0→P4 排期治理。
> 模型定义单文件：`backend/core/db/database.py`；迁移：`backend/alembic/versions/`（001→025）。

---

## 0. TL;DR

- 27 张表 / 10 个域，对标 Moodle（468 表）属**健康范围**——表多本身不是病。
- 分层方向（事件→rollup→展示、FK/no-FK 故意分裂）已是**业界中上水准**。
- 体检：**18 ✅ / 4 ⚠️ / 9 处 🔴**，问题高度集中在 3 个同类病：
  1. **缺 `course_id`**（Message / NotebookEntry / BotNotification）→ 课程级查询必 JOIN Session，已致一次跨租户 bug。
  2. **双真相源**（KB 状态在 knowledge_bases 死列 + kb_builds；图谱在 users.knowledge_graph + knowledge_mastery）。
  3. **死 rollup**（course_daily_rollup / student_course_rollup 每小时在算却没人读）。
- 不是结构错误，是**少数残留没收尾**。路线图用 Expand/Contract 分 5 阶段、可中断地治。

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

每条 = **规则 · 为什么 · 正例 · 反例（现状违规）**。新表必须全部合规。

### 原则 1：分层归属（一概念一真相源）

每张表归属且仅归属一层：

| 层 | 特征 | 本项目表 |
|---|---|---|
| ① 原始事件 | 只追加，内容永不 UPDATE | `messages` `learning_events` `memory_episodes` `llm_usage_records` |
| ② 派生读模型 | cron 重算（delete-then-reinsert），从不 read-modify-write | `course_daily_rollup` `student_course_rollup` `llm_usage_daily` |
| ③ 可变状态 | 频繁 UPDATE 的当前事实 | `sessions` `kb_builds` `knowledge_mastery` `research_checkpoints` |
| ④ 引用/配置 | 读多写少 | `knowledge_bases` `enrollments` `user_*` 配置表 |

**规则**：一个业务概念只能有**一个权威存储**，其余只能是它的投影/缓存（且必须能从权威源重算）。

**为什么**：双真相源必然漂移；查"KB 状态"不知信谁、查"用户图谱"两个地方不一致。

**正例**：`kb_builds` 是 KB 构建状态的唯一真相源（`UNIQUE(kb_id, backend)`），KB 行的展示状态由 `aggregate_build_status(kb.builds)` 现算投影。
**反例（违 P1）**：`knowledge_bases` 行上残留的死状态列（`status/progress*/chunks_*/token_estimate`）与 `kb_builds` 重复 → 见 [P2](#p2--kb-状态单一真相源中高风险纯收尾)。`users.knowledge_graph` 与 `knowledge_mastery` 双写 → 见 [P3](#p3--记忆单一真相源--users-瘦身--并发高风险最大块)。

### 原则 2：读写分离（CQRS：事件 → rollup 读模型）

**规则**：原始层只 INSERT、内容永不 UPDATE；rollup 层全派生（delete-then-reinsert），从不 read-modify-write。**读路径只读 rollup，不现算**。

**为什么**：现算=每请求 JOIN 风暴；rollup=O(1) 读 + 后台批量算。`analytics_student_stats` 现状一次 6 条串行查询 + 重 Python 处理。

**正例**：`llm_usage_records`(原始) → `llm_usage_daily`(派生)，且 `llm_usage_daily` **已被** teacher/admin 端点消费。
**反例（违 P2）**：`course_daily_rollup` / `student_course_rollup` cron 每小时算，却**无读路径**消费——纯浪费 → 见 [P0](#p0--接通死-rollup--补全事件层高-roi低风险)。`learning_events` 只写 `verb=answered` 缺失，导致 rollup 答题口径退回去 JOIN `NotebookEntry ⋈ Session`。

### 原则 3：多租户就地隔离（行级 `course_id`，写时落盘）

**规则**：共享表 + 行级隔离档下，**凡是课程级可查的行，必须在写入时就把 `course_id` 落盘**，使查询无需 JOIN 即可确定租主。

**为什么**：靠 JOIN Session 反查 `course_id` 已致一次跨租户 bug（`teacher.py:779` 注释："旧实现仅按 user_id 全局聚合，会把他课程的答题也算进本课程"）；rollup 也因此 JOIN 沉重。

**正例**：`knowledge_mastery` 带 `course_id`（`UNIQUE(user_id, course_id, kp_id)`），当初正是为修 `users.knowledge_graph` 跨课污染而加。
**反例（违 P3）**：`messages` / `notebook_entries` / `bot_notifications` 缺 `course_id` → 见 [P1](#p1--租主硬化给-3-表补-course_id中风险)。

> **注**：不新建 `courses` 表（course 与 `knowledge_bases` 1:1 已工作；给 13 张表的 `course_id` 改真 FK 是巨大 Expand/Contract 却只换边际收益）。治跨租户病的正确姿势是补 `course_id` 列，而非建父表。

### 原则 4：FK 纪律（故意分裂，写进 docstring）

**规则**：业务表用真 FK + `ondelete=CASCADE`（数据随父死亡）；分析/计费/检查点表用**裸字符串无 FK**（历史须在用户/课程删除后存活）。**每个选择都要在模型 docstring 写明理由**。

**为什么**：账单 `llm_usage_records` 若 CASCADE，删用户=删账单，计费数据不可回溯。

**正例（已合规）**：`llm_usage_records` / `llm_usage_daily` / `research_checkpoints` 无 FK，docstring 写明"须在用户删除后存活"。`messages`/`enrollments`/`notebook_*` 等业务表用 CASCADE。本项目此项**已达标**，仅需制度化。

### 原则 5：并发一致性（可变列要 OCC 或逐行 append）

**规则**：凡在并发下被整体 rewrite 的可变列，必须有 **OCC 版本号**或改成**逐行 append**。裸 read-then-全量 rewrite JSON = 丢更新 bug。

**为什么**：两个 `consolidate_memory` job 同用户并发，会各自 read→rewrite→互相覆盖。

**正例**：`sessions.summary_version`（OCC：`UPDATE ... WHERE summary_version = old`）防多 worker 摘要丢更新；`knowledge_mastery` 逐行 append + 读时衰减。
**反例（违 P5）**：`users.knowledge_graph` / `error_graph` 无版本号，由 `graph_memory.save_graphs` 整列 rewrite。→ 见 [P3](#p3--记忆单一真相源--users-瘦身--并发高风险最大块)（治理方式是删列迁到 `knowledge_mastery`，而非给将删的列加版本号）。

### 原则 6：演进纪律（Expand/Contract，不留孤儿，迁移唯一权威）

**规则**：
1. schema 变更走 **Expand → Migrate → Contract**，"保留不删"必须挂一个 Contract 工单。
2. **所有建表只经 Alembic**，禁止 `create_all` + `_ensure_column` 双轨。
3. 每个迁移有 `revision`/`down_revision` 链，文件名与 revision 一一对应。

**为什么**：双轨导致 rev 014 补建 7 张早已"活在 create_all 里"的表、每个迁移裹 `_table_exists/_column_exists` 幂等守卫；6 个孤儿记忆列是"保留不删"欠的债；缺 revision `007` + `005_` 文件重名是卫生信号。

**反例（违 P6）**：`init_db()` 的 `create_all`+`_ensure_column`（dev/test 路径）与生产 Alembic 双轨；`users` 6 个孤儿列。→ 见 [P3](#p3--记忆单一真相源--users-瘦身--并发高风险最大块) / [P4](#p4--建表纪律--配额回填基础设施卫生)。

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

## Part 2 — 27 表体检打分

图例：`✅合规` / `⚠️可改进` / `🔴违反原则`

| 域 | 表 | 判定 | 要点（违反哪条） |
|---|---|---|---|
| Auth | `users` | 🔴 | 超载（身份 + 6 记忆列）；`knowledge_graph` 无 OCC 整列 rewrite（违 P1/P5） |
| | `user_social_bindings` | ✅ | 标准 OAuth 绑定 |
| | `user_mcp_enrollments` | ✅ | 1:1 配置 |
| | `user_search_configs` | ✅ | 1:1 配置 |
| | `user_llm_providers` | ⚠️ | 1:1 配置合规；`fast_model` 是死字段 |
| | `teacher_invites` | ✅ | 一次性码流程 |
| | `teacher_applications` | ✅ | 审批流程，索引合理 |
| Chat | `sessions` | ⚠️ | OCC 摘要做对；`updated_at` 每条消息 UPDATE（有界，留意索引 churn） |
| | `messages` | 🔴 | 缺 `course_id`，课程级查询必 JOIN Session（违 P3） |
| RAG | `knowledge_bases` | 🔴 | 残留死状态列 `status/progress*/chunks_*/token_estimate`（违 P1） |
| | `kb_builds` | ✅ | 真相源，`UNIQUE(kb_id,backend)` 合理 |
| | `kb_files` | ✅ | |
| Quiz | `notebook_entries` | 🔴 | 缺 `course_id`（违 P3，已致跨课 bug） |
| | `notebook_categories` | ✅ | |
| | `notebook_entry_categories` | ✅ | M2M join |
| Academic | `enrollments` | ✅ | 租户 JOIN 枢纽（符合预期） |
| | `course_schedules` | ✅ | 课表 |
| | `grades` | ✅ | upsert 键合理 |
| Memory | `memory_episodes` | ✅ | 原始+outbox 双职，append-only |
| | `knowledge_mastery` | ✅ | 逐行 append + 读时衰减，P3/P5 范例 |
| Analytics | `learning_events` | ⚠️ | 只写 `asked`，`answered`/`feedback` 未产（违 P2 完整性） |
| | `course_daily_rollup` | 🔴 | cron 在算却无读路径（违 P2，纯浪费） |
| | `student_course_rollup` | 🔴 | 同上 |
| Billing | `llm_usage_records` | ✅ | append-only + 无 FK，P4 范例 |
| | `llm_usage_daily` | ⚠️ | 被 teacher/admin 消费，合规；配额真相只在 Redis（见 P4） |
| Research | `research_checkpoints` | ✅ | 无 FK 故意，合理 |
| Bot | `bot_notifications` | 🔴 | 缺 `course_id` + 无留存清理 cron（违 P3） |

**统计**：18 ✅ / 4 ⚠️ / 9 处 🔴（3 处 = 缺 `course_id` 同病；2 处 = 双真相源同病；2 处 = 死 rollup 同病）。**问题高度集中，非系统性。**

---

## Part 3 — 收尾路线图（Expand/Contract，分阶段可中断）

按 **ROI / 风险** 排序，每阶段独立可交付、可中断。每阶段格式：**病 → 修 → 文件 → 验证**。

### P0 ｜ 接通死 rollup + 补全事件层（高 ROI，低风险）

- **病**：`course_daily_rollup` / `student_course_rollup` 每小时 DELETE+重算却无人读；仪表盘 `analytics_overview` / `analytics_student_stats`（6 条串行查询）/ `analytics_student_detail` 每请求现算。
- **修**：
  1. 把三个 analytics 端点改为读 rollup 表；rollup 行缺失/过期时**降级现算**（兜底，保证正确性）。
  2. quiz 答题路径开始写 `learning_events(verb=answered)`，让 rollup 的答题口径不再依赖 `NotebookEntry ⋈ Session`。
- **文件**：`backend/api/teacher.py`（analytics_*）、`backend/core/analytics/learning_rollup.py`、`backend/main.py`（写 answered 事件）。
- **迁移**：无需。
- **验证**：仪表盘请求数 6→1（读 rollup）；rollup stale 降级路径有测试；学情测试绿。

### P1 ｜ 租主硬化：给 3 表补 `course_id`（中风险）

- **病**：`messages` / `notebook_entries` / `bot_notifications` 缺 `course_id`，课程级查询必 JOIN Session，已致跨租户 bug。
- **修（Expand/Contract）**：
  1. **Expand**：加可空 `course_id String(64)`（迁移 026）+ 复合索引 `(course_id, ...)`。
  2. **Migrate**：回填 `UPDATE t SET course_id = s.course_id FROM sessions s WHERE t.session_id = s.id`（`bot_notifications` 按 user 维度回填）。
  3. **Contract**：验证覆盖率 100% 后改 `NOT NULL`。
  4. 新写入在创建行时落 `course_id`（`add_message` / `notebook_store` / notification cron）。
  5. collapse analytics / rollup 里的 Session JOIN。
- **文件**：`database.py`（3 模型）、迁移 026、`core/memory/memory.py`(`add_message`)、`core/db/notebook_store.py`、`services/cron/executor.py`、`api/teacher.py` + `core/analytics/learning_rollup.py`。
- **验证**：`SELECT count(*) WHERE course_id IS NULL` = 0；跨租户回归测试保绿；EXPLAIN 显示去 JOIN。

### P2 ｜ KB 状态单一真相源（中高风险，纯收尾）

- **病**：`knowledge_bases` 残留死列 `status / progress / progress_msg / chunks_done / chunks_total / token_estimate / error_msg`，与 `kb_builds` 重复。
- **修（Contract）**：
  1. grep 全仓确认**无读者**依赖 KB 行这些列（`aggregate_build_status` 读的是 `kb.builds`，非 `kb.status`；前端 `_kb_to_dict`/`_kb_to_course` 同口径）。
  2. 迁移 027 DROP 这些列。
- **文件**：`database.py`(`KnowledgeBase`)、迁移 027、`api/admin.py` / 前端 `KbDetailPanel` 若有引用。
- **验证**：`test_kb_builds` / `test_migration_greenfield` 绿；建库/检索全流程冒烟。

### P3 ｜ 记忆单一真相源 + users 瘦身 + 并发（高风险，最大块）

- **病**：①图谱双写 `users.knowledge_graph`(无 OCC 整列 rewrite) ↔ `knowledge_mastery`(做对)；②`users` 6 个记忆列（`summary_memory/profile_memory/scope_memory/preferences_memory/knowledge_graph/error_graph`）多为孤儿（语义记忆已交 mem0 `memories` 表）。
- **修（Expand/Contract，建议拆 2 个子 PR）**：
  - **图谱（子 PR-a）**：教师仪表盘读路径（`teacher.py:734`）从 `users.knowledge_graph` 切到 `knowledge_mastery`（或其投影）→ 停止双写 → Contract 删 `knowledge_graph`/`error_graph`。**删列即消除 OCC bug**（比给将删的列加版本号更优）。
  - **L2 文本列（子 PR-b）**：逐列 grep 读者——`summary_memory`/`profile_memory` 若已被 mem0 取代则 Contract 删；仍在用的保留或迁独立表。`scope_memory`/`preferences_memory` 同理判定。
- **文件**：`database.py`(`User`)、`core/memory/graph_memory.py`、`api/teacher.py`、`core/memory/mem0_client.py`、迁移 028。
- **验证**：仪表盘图谱渲染正常（数据源切换）；mem0 检索冒烟；`test_rag_*` / 记忆测试绿。
- **⚠️ 风险**：最高的一块，必须逐列确认读者后再删。

### P4 ｜ 建表纪律 + 配额回填（基础设施卫生）

- **病**：①双轨建表（`init_db()` create_all + Alembic）；②配额真相只在 Redis，Redis 丢=静默重置；③迁移文件卫生（缺 revision 007、`005_` 文件重名）。
- **修**：
  1. `init_db()` 只做 seed，schema 全交 Alembic；测试用经 Alembic 迁移的 DB（或受控 create_all fixture）。
  2. 启动时 + 定期从 `llm_usage_daily` 回填 Redis `ca:costquota:*`，Redis 失联不静默重置。
  3. `005_` 重命名补注释说明 007 缺口（不可改 history）。
- **文件**：`core/db/database.py`(`init_db`)、`tests/conftest.py`、`core/observability/cost_quota.py`、迁移目录。
- **验证**：空库 `alembic upgrade head` → 27 表齐（greenfield 测试）；Redis 清空后配额从 DB 回填有测试。

---

## 附录：两个已拍板决策

1. **不新建 `courses` 表**——course 与 `knowledge_bases` 1:1 已工作，建父表要改 13 张表的 `course_id` 为 FK，ROI 不划算；治跨租户病用"补 `course_id` 列"（`knowledge_mastery` 范式）。
2. **P3 删列而非加 OCC**——`users.knowledge_graph` 是过渡双写死列，加版本号是给将删之物打补丁；删列即根治。

---

*本文由 2026-08-08 的 schema 审计产出（3 个并行 Explore：表清单 / 迁移 churn / 访问模式 + 三维调研）。体检判定已逐行核对 `database.py` line 78–247 + 529–733。*
