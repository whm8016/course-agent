# 交付与上线手册

面向交付部署与验收的运维手册。架构细节见 `backend/docs/ARCHITECTURE.md`，表设计规范见 `backend/docs/SCHEMA_DESIGN_PRINCIPLES.md`。

## 一、首次部署（docker compose）

```bash
# ① 项目根目录：compose 变量（数据库密码等）
cp .env.example .env
# 编辑 .env：修改 POSTGRES_PASSWORD

# ② backend/：应用层配置
cd backend && cp .env.example .env
# 编辑 .env：LLM__API_KEY 必填；生产还需 SECURITY__JWT_SECRET /
# SECURITY__PROVIDER_ENCRYPTION_KEY（见下节 checklist）

# ③ 启动（生产不含 eval 服务，它在 docker-compose.eval.yml）
cd .. && docker compose up -d
```

验证：`docker compose ps` 全部 healthy；`curl http://localhost/api/health` 返回 `"status": "ok"`。

## 二、切生产环境 checklist（顺序敏感）

`ENVIRONMENT=production` 会触发 settings 的生产 fail-fast（`backend/settings/base.py` `_check_prod`），以下项缺一**拒绝启动**——这是有意设计，防止带弱配置上线。按序配好再切：

1. `SECURITY__JWT_SECRET`：强随机值（`openssl rand -hex 32`）
2. `SECURITY__ALLOWED_ORIGINS`：具体域名，不允许 `*`
3. `SECURITY__PROVIDER_ENCRYPTION_KEY`：Fernet 密钥（`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`，用于加密用户级 LLM Key）
4. 根目录 `.env` 的 `POSTGRES_PASSWORD`：强密码（弱密码 postgres 同样 fail-fast）。注意密码在 postgres 数据卷**首次创建**时生效；已有数据卷需 `docker compose down -v` 重建或库内 `ALTER USER`
5. 最后切 `ENVIRONMENT=production`

生产形态下的暴露面：backend 8002 只绑宿主机回环，外部流量一律走 frontend（nginx :80）反代；Swagger/redoc/openapi.json 端点关闭（`docs_url=None`）；`/metrics` 仅宿主机回环可达（远端采集走 ssh 隧道或反代加鉴权）。

## 三、内存预算参考（4GB 部署机）

| 服务 | 预算（估） | 说明 |
|---|---|---|
| postgres（pgvector+zhparser） | ~800MB | 词典 mmap 走 OS page cache，连接内存低 |
| backend（gunicorn ×4） | ~1.2GB | LLM/RAG 主进程 |
| worker（ARQ） | ~600MB | 索引/研究/定时任务 |
| mcp-gateway | 1GB（已 cap） | 第三方代码，唯一已设 mem_limit |
| frontend（nginx） | ~30MB | 静态反代 |
| redis | ~100MB | AOF+RDB |

合计贴近 4GB，故本批**未**对其余服务加 mem_limit——宁可不 cap，不要交付第一天 OOM。首次部署稳定后用 `docker stats` 实测再定 cap 值。

## 四、测试基线

```bash
cd backend
.venv/Scripts/python -m pytest --ignore=tests/test_stitch_cooldown_commit.py -q
.venv/Scripts/python -m pytest tests/test_stitch_cooldown_commit.py -q   # 该文件单独跑（两步法）
```

- 基线：**1435+ passed / 2 skipped / 1 xfailed**（SQLite in-memory，无需外部服务）
- **已知测试隔离问题**（非产品代码 bug）：`test_stitch_cooldown_commit.py` 单文件跑 5/5 全过（~1s），
  但混在全量里会挂——最小复现 `pytest tests/test_chat_happy.py tests/test_stitch_cooldown_commit.py`
  （2 用例被拖到 ~34s，更大组合下第 4 用例 `test_failed_turn_does_not_burn_cooldown` 无限等待）。
  根因方向：chat 系测试的 client fixture 起 app 后残留全局状态，拖慢/卡死 cooldown 文件
  turn 路径的 `subscribe_turn` 终止。建议交付后专项修（TurnRuntimeManager 测试隔离），
  交付期间用上面两步法口径。
- 前端：`cd frontend && npm test`（67 用例）

## 五、已知问题

1. **Bot 启停偶发 409**：backend 多 worker 选主，非 leader 对启停操作返回 409。前端已带 ~2 秒预算的自动重试（重试可落到其他 worker 命中 leader）；若重试耗尽仍失败，稍候再点一次即可。
2. **users.knowledge_graph 双真相源**：图谱快照列与 knowledge_mastery 并存是 documented 决策（前端渲染直接消费该列），由巩固流程双写 + `graph_version` OCC（条件 UPDATE + 冲突重试）止血，建议交付后专项收口。
3. **ELASTICSEARCH__\* 配置为预留开关**：compose 已移除 ES 服务（sparse 路走 PG tsvector），默认关闭；如需自托管 ES 混合检索，自行加回服务块并设 `ELASTICSEARCH__ENABLED=true`。
4. **git 历史含论文/内部审计旧版本**：相关文件已 `git rm --cached` 出库（不再被跟踪），但历史提交中仍有旧版本；彻底清除需 rewrite 历史（高风险，未做）。
5. **rerank 开关口径**：代码默认已开（`RERANK__ENABLED=true`，`.env.example` 同步）。本地开发若曾设 `false`，**不要照抄进客户环境**。
6. **课程主题图构建迁移 032**：`course_topic`/`course_topic_edge` 两表在模型存在早于生产迁移（create_all 自建 vs 032 补票），升级既有库时 032 幂等守卫会跳过已建对象，无需干预。

## 六、升级 / 回滚

- 升级：`git pull` → `docker compose up -d --build`（backend/worker 以源码挂载运行；`backend_venv` volume 仅在依赖变更时需删除重建）
- 数据库 schema：容器启动命令自带 `alembic upgrade head`，失败会重试
- 回滚：`git checkout <commit>` → `docker compose up -d`；数据卷不受影响
- 评测环境（可选）：`docker compose -f docker-compose.yml -f docker-compose.eval.yml up -d`
