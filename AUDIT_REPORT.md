# 后端系统审查报告

**审查日期**: 2026-07-07
**审查范围**: `backend/` 全部模块（9 个批次，200+ 文件逐文件深挖）
**审查方法**: 逐函数推演调用时机 + 共享状态读写顺序，不是表面扫描

---

## 汇总统计

| 严重度 | 数量 | 说明 |
|--------|------|------|
| **CRITICAL** | 3 | 生产环境会导致数据丢失/功能完全失效 |
| **HIGH** | 22 | 安全漏洞、并发 bug、会导致用户可感知故障 |
| **MEDIUM** | 63 | 架构隐患、文档与代码不一致、边界条件 |
| **LOW** | 71 | 工程实践改进、死代码、小优化 |
| **总计** | **159** | |

| 批次 | CRITICAL | HIGH | MEDIUM | LOW | 合计 |
|------|----------|------|--------|-----|------|
| 1. 并发/多worker | 2 | 2 | 4 | 4 | 12 |
| 2. Agent调度核心 | 0 | 0 | 2 | 10 | 12 |
| 3. 四大能力管线 | 0 | 1 | 5 | 12 | 18 |
| 4. LLM Provider层 | 0 | 2 | 8 | 5 | 15 |
| 5. RAG/知识库 | 0 | 2 | 11 | 9 | 22 |
| 6. Bot/IM子系统 | 0 | 3 | 7 | 3 | 13 |
| 7. DB/持久化+鉴权 | 1 | 3 | 8 | 12 | 24 |
| 8. API路由层 | 0 | 5 | 11 | 8 | 25 (*) |
| 9. Memory/Skills/MCP | 0 | 4 | 7 | 8 | 19 (*) |

(*) Batch 8 含 1 条 Info 级未计入; Batch 9 原报告统计为 15 但逐项复核为 19。

---

## CRITICAL（必须立即修复）

### C-1. Shutdown 与 on_gain 回调竞态 — 可导致单例服务在关停 worker 上重启

- **模块**: leader.py + main.py
- **位置**: `backend/core/leader.py:140-154`, `backend/main.py:117-148`
- **触发条件**: Worker 刚赢得选举、`on_gain`(start_singleton_services) 正在执行 → Gunicorn SIGTERM → `shutdown_leader` 运行 `on_lose` 停单例 → `on_gain` 完成后又把 `_singletons_started=True` → Cron/Bot/MCP 在即将死亡的 worker 上重启
- **修复方向**: 加 `_shutting_down` 标志；`start_singleton_services` 入口和出口检查；`shutdown_leader` 同时 cancel campaign task

### C-2. Alembic 迁移链不完整 — 生产 greenfield 部署缺核心表

- **模块**: alembic/versions/
- **位置**: `backend/alembic/versions/001-013`
- **触发条件**: 全新生产 DB 只跑 `alembic upgrade head`（生产不走 `create_all`）→ 缺 `knowledge_bases`、`kb_files`、`notebook_entries`、`notebook_categories` 等核心表
- **修复方向**: 从 `Base.metadata` 生成完整 baseline migration；或补齐所有缺失表/列的增量迁移

### C-3. shutdown_leader 中 `_singletons_started` 无防护（C-1 同根因）

- **位置**: `backend/main.py:148`
- **说明**: 与 C-1 同一个竞态的另一面 — `_singletons_started = True` 无条件设置

---

## HIGH（应优先修复）

### 安全类

| # | 问题 | 位置 | 触发条件 |
|---|------|------|----------|
| H-1 | **Turn IDOR**: `answer_now` / `subscribe_turn` 无用户归属校验 | `api/chat.py` | 已认证用户传他人 turn_id |
| H-2 | **LightRAG 路径/课程绕过**: lightrag index API 无 course_id 归属检查 | `api/lightrag.py` | 传入他人 course_id |
| H-3 | **exam_mimic path traversal**: `paper_path` 未校验 | `api/question.py` | 传入 `../../etc/passwd` |
| H-4 | **任意文件读**: client `file_path` 穿透到 `inject_image_parts` | `attachment.py:37`, `multimodal.py:166-168` | 发送 `{"file_path":"/etc/passwd"}` |
| H-5 | **Notebook IDOR**: `delete_category` 先删 junction 再校验归属 | `notebook_store.py:235-244` | 传入他人 category_id |
| H-6 | **课程访问缓存未失效**: `course_access_invalidate()` 定义了但从未调用 | `cache.py:164-172` | 教师移除学生后 5 分钟窗口期内学生仍可访问 |

### 并发/数据完整性类

| # | 问题 | 位置 | 触发条件 |
|---|------|------|----------|
| H-7 | **Flush manager 先删后持久化**: Redis key 在 Mem0/PG 写入前被删除 | `flush_manager.py:202-206` | Mem0 或 PG 在 flush 期间不可用 → 缓冲轮次永久丢失 |
| H-8 | **Flush manager SCAN 不分页**: 单次 SCAN 漏 key | `flush_manager.py:178,229` | 超过 ~200 个待 flush key |
| H-9 | **Flush manager TTL 数据丢失**: 600s TTL 可在 worker 宕机期间静默过期 | `flush_manager.py:32,87-89` | ARQ worker 宕机 > 10 分钟 |
| H-10 | **LightRAG 实例池 use-after-evict**: 锁仅在 `_get_instance` 期间持有 | `instance_pool.py:113-152` | 并发多课程请求触发 evict 正在使用的实例 |
| H-11 | **Circuit breaker HALF_OPEN 死锁**: `success_threshold=2` > `half_open_max_calls=1` | `reliability.py:115-120,130-141` | 熔断后恢复探测成功一次 → 永久卡在 HALF_OPEN |

### Bot/多 Worker 类

| # | 问题 | 位置 | 触发条件 |
|---|------|------|----------|
| H-12 | **Follower worker 可通过 API 启动 bot**: 无 `is_leader()` 门控 | `manager.py:147-260`, `api/bot.py` | 多 worker + LB → 非 leader 处理 bot API 请求 |
| H-13 | **Bot split-brain**: 两个 worker 各自持有同一 bot 实例 | `manager.py:150-152,227-231` | H-12 + auto_start on leader |
| H-14 | **IM 消息去重仅内存**: leader failover 后丢失 | `qq.py:81`, `feishu.py:103` | Leader 切换 → 平台重发 → 重复处理 |
| H-15 | **Agent loop 忽略 TRM ERROR 事件**: IM 用户得到无回复的沉默 | `agent/loop.py:305-308` | LLM/orchestrator 异常时 bot turn |

### LLM Provider 类

| # | 问题 | 位置 | 触发条件 |
|---|------|------|----------|
| H-16 | **Agent loop 共享全局 circuit breaker**: profile client 失败会 OPEN 全局熔断 | `llm.py:169,198-206` | 用户选了个坏 profile → 连累所有用户 |

### DB/Auth 类

| # | 问题 | 位置 | 触发条件 |
|---|------|------|----------|
| H-17 | **DB 连接池未按 worker 缩放 + settings 未接入** | `database.py:35-43` | 4 worker × (10+15) = 100 连接 |
| H-18 | **Admin 自动提权**: 注册用户名 == `ADMIN_USERNAME` 自动变 admin | `auth.py:72-73,80` | 默认 `admin` 用户名未被预占 |

---

## MEDIUM（建议修复，分批处理）

### 并发/状态管理

| # | 问题 | 批次 | 位置 |
|---|------|------|------|
| M-1 | Redis 中断 > TTL 时短暂双 leader | B1 | `leader.py:201-231` |
| M-2 | `shutdown_leader` 中 `asyncio.shield` 可导致 campaign 重启 | B1 | `leader.py:292-298` |
| M-3 | Worker fallback Redis 连接未 close | B1 | `worker.py:196-199,221-224` |
| M-4 | Agent loop 最后一轮仍返回 tool_calls 时 final_text 为空 | B2 | `loop.py:437-439,554-570` |
| M-5 | Shallow `replace()` + 并行 research 共享可变嵌套字段 | B2 | `context.py:20-65` |
| M-6 | Deep solve `conversation_history` 未隔离（历史泄入 solve） | B3 | `solve/pipeline.py:93-98` |
| M-7 | `SolveSession` pipeline 入口未重置（stale plan on sid collision） | B3 | `solve/session.py:83-92` |
| M-8 | Quiz 一题失败整批中断 | B3 | `question/pipeline.py:140-169` |
| M-9 | Quiz JSON schema 校验薄弱 | B3 | `question/pipeline.py:235-263` |
| M-10 | Research 并行 block 事件在共享 StreamBus 上交错 | B3 | `research/pipeline.py:388-400` |
| M-11 | Session summary 并发压缩无锁 | B9 | `main.py:94-95`, `session_summary.py:87-183` |
| M-12 | Graph memory flush 无 `db.commit()` | B9 | `flush_manager.py:142-156` |
| M-13 | Flush manager 并发 flush 竞态（无 per-key lock） | B9 | `flush_manager.py:186-206` |

### LLM Provider

| # | 问题 | 位置 |
|---|------|------|
| M-14 | Worker scaling 未应用到 configured circuit breaker | `reliability.py:84-91,311-318` |
| M-15 | LLM Semaphore 文档有、代码无 | `reliability.py` + `ARCHITECTURE.md:249` |
| M-16 | `BACKEND_WORKERS` env vs `settings.backend_workers` 不一致 | `reliability.py:82` vs `settings/base.py:464` |
| M-17 | Fallback 仅在 `chat_complete` 不在 agent loop | `llm.py:269-283` |
| M-18 | Fallback 绕过 retry/circuit | `llm.py:273-279` |
| M-19 | Profile `base_url` 空值不回退到 `.env` | `provider_factory.py:102-105` |
| M-20 | Catalog 读写 TOCTOU（无锁读 → 可解析到空 catalog） | `catalog.py:107-116` |
| M-21 | Vision 调用绕过 reliability 层 | `vision_describe.py:82-94` |

### RAG/知识库

| # | 问题 | 位置 |
|---|------|------|
| M-22 | `finalize_storages()` 异常被 bare except 吞掉 | `instance_pool.py:70-73` |
| M-23 | ARQ worker 不计入 LRU 缩放公式 | `instance_pool.py:26` |
| M-24 | `BACKEND_WORKERS` 须手动匹配 `-w` 数 | `settings/base.py:653-655` |
| M-25 | Embedding 无 `len(resp.data)==len(batch)` 校验 | `embedding_bridge.py:88-91` |
| M-26 | LightRAG 可用性检查不含 embedding 配置 | `llm_adapter.py:89-103` |
| M-27 | `.doc`/`.ppt` 上传允许但无 handler | `file_routing.py:191-194` |
| M-28 | Resume 会重复索引所有图片 | `ingestion.py:426-460` |
| M-29 | 图片先提交、文本失败 → 知识图谱只有图片 | `ingestion.py:426-477,525+` |
| M-30 | Abort 后 partial batch 未回滚 | `ingestion.py:603-611` |
| M-31 | RAG 缓存已初始化但未接入检索路径（死代码 + 误导日志） | `main.py:198`, `cache.py` |
| M-32 | 缓存无失效 hook（索引完成后不清缓存） | `cache.py:201-240` |
| M-33 | `_index_signatures` 内存态，重启后丢失 | `instance_pool.py:31,184-191` |

### Bot/IM

| # | 问题 | 位置 |
|---|------|------|
| M-34 | Bind code 进程内存不跨 worker | `binding.py:10-11` |
| M-35 | Bus 内消息无持久化（leader 死亡 → 丢消息） | `bus/queue.py:16-18` |
| M-36 | Feishu WS thread stop 不 join | `feishu.py:170-172` |
| M-37 | Cron job 不随 bot 删除而清理 | `manager.py:321-332` |
| M-38 | Cron 无分布式执行锁（leader crash mid-run → 重执行） | `cron/service.py:347-376` |
| M-39 | `SessionManager._cache` 无驱逐（内存泄漏） | `session/manager.py:71-86` |
| M-40 | In-flight dispatch task 不随 bot stop 取消 | `agent/loop.py:102-109` |
| M-41 | `NotificationService` 用首个匹配 bot 而非正确归属 | `notification.py:86-107` |

### DB/Auth

| # | 问题 | 位置 |
|---|------|------|
| M-42 | DB session 贯穿整个 SSE 流（长时间持有连接） | `database.py:607-615` |
| M-43 | Stale deny cache（先被拒 → 教师加入 → 5 分钟仍被拒） | `cache.py:137,152-159` |
| M-44 | 破坏性迁移 008 无回滚 | `008_drop_v3_memory_tables.py:21-30` |
| M-45 | `user_llm_provider` 加密失败静默存空 key | `user_llm_provider.py:129-134` |
| M-46 | DB pool settings 定义了但 engine 没用 | `settings/base.py:228-229` vs `database.py:39-40` |

### Skills/MCP

| # | 问题 | 位置 |
|---|------|------|
| M-47 | Course skill 写入无 course 授权检查 | `api/skill_knowledge.py:73-86` |
| M-48 | `always:true` skill 注入面（恶意课程 skill 作者） | `skill_service.py:283-286` |
| M-49 | MCP 连接 task 退出后适配器仍注册（stale adapter） | `manager.py:323-331` |

### API 路由层（来自 Batch 8 的 11 条 MEDIUM，以下为主要代表）

| # | 问题 | 位置 |
|---|------|------|
| M-50+ | 多个端点缺少多租户 course_id 校验、upload 文件名清洗不完整、expensive 端点无 rate limit 等 | `api/*.py` 多处 |

---

## 按修复优先级排序的 Top 20 建议

| 优先级 | 问题编号 | 一句话描述 | 工作量 |
|--------|---------|-----------|--------|
| 1 | H-1,H-4,H-5 | 修复 IDOR + 任意文件读 (安全) | 小 |
| 2 | H-3,H-2 | Path traversal + lightrag 课程绕过 | 小 |
| 3 | C-1,C-3 | Shutdown/on_gain 竞态（加 `_shutting_down` flag） | 小 |
| 4 | H-6,M-43 | 课程访问缓存失效（调用 `course_access_invalidate`） | 小 |
| 5 | H-11 | Circuit breaker HALF_OPEN 死锁（调 success_threshold） | 极小 |
| 6 | H-7,H-8,H-9 | Flush manager 数据丢失三连（先删后写+SCAN+TTL） | 中 |
| 7 | H-16 | Profile client 隔离 circuit breaker | 小 |
| 8 | H-12,H-13 | Bot API leader 门控 | 小 |
| 9 | H-17,M-46 | DB 连接池接入 settings + 按 worker 缩放 | 小 |
| 10 | H-10 | LightRAG 实例池引用计数/lease | 中 |
| 11 | C-2 | Alembic baseline migration | 中 |
| 12 | H-14 | IM 去重移到 Redis | 中 |
| 13 | H-15 | Bot agent loop 处理 ERROR 事件 | 小 |
| 14 | H-18 | Admin 自动提权去掉或加 bootstrap token | 小 |
| 15 | M-6,M-7 | Deep solve history 隔离 + session reset | 小 |
| 16 | M-12 | Graph memory flush 加 commit | 极小 |
| 17 | M-14,M-15 | Circuit breaker scaling + LLM semaphore | 中 |
| 18 | M-42 | SSE 端点短生命周期 DB session | 中 |
| 19 | M-47 | Skill API 加课程授权检查 | 小 |
| 20 | M-8,M-9 | Quiz 单题容错 + JSON 校验加强 | 中 |

---

## 做得好的部分（无需改动）

| 领域 | 评价 |
|------|------|
| Leader 选举状态机（正常路径） | CAS 续约 + 竞选者循环设计自洽，防脑裂四道防线齐全 |
| Agent loop 并发隔离 | 每个 turn 独立 context/bus/messages，concurrent chat 安全 |
| Tool dispatch 错误隔离 | 三层 try/except（executor/registry/dispatch），单工具失败不崩 loop |
| StreamBus replay 机制 | 断线重连不丢不重，subscriber cleanup 正确 |
| Rate limiter | 正确使用 Redis 共享，多 worker 全局生效 |
| 密码学 | bcrypt + HS256 JWT + Fernet 加密 + `SecretStr` |
| Prompt loader | `yaml.safe_load`，无注入，缺失优雅降级 |
| Invite code 生成 | `secrets.choice`，不可预测 |
| Skills path traversal 防护 | `..` 检查 + `is_relative_to` + 100k 读取上限 |
| Deferred tools 设计 | 可变 tool_schemas list + contextvar 注入，无并发写冲突 |
| LRU 缩放（RAG 实例池） | `capacity // workers` 已实现并有测试 |

---

## 审查方法论建议（回答你的"该用什么提示词"）

有效的 prompt 结构：

```
只审查 <具体一个模块/文件>。目标：找并发/时序 bug、边界条件 bug、资源泄漏、与文档描述不符的地方。要求：
1. 逐个函数推演调用时机和共享状态的读写顺序，不要只看表面逻辑对不对。
2. 每个问题必须给出：触发条件（什么时序/输入会触发）+ 文件:行号 + 影响。
3. 没发现严重问题就如实说"没发现"，不要为了有产出而编造无关痛痒的建议。
4. 只读不改，先给报告。
```

关键原则：
- **一次只喂一个模块**（而不是"检查整个项目"）
- **要求推演时序/状态机**（而不是"看代码好不好"）
- **要求具体触发条件**（而不是"建议加强错误处理"这种空话）
- **允许说"没问题"**（否则模型会编造问题凑数）
