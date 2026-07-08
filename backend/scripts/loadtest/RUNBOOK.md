# 上线前压测运行手册（RUNBOOK）

> 工业级 20 并发压测的完整执行流程。压测环境**对齐生产 docker-compose.yml 形态**：
> 1× backend(gunicorn -w4) + 1× worker(arq) + nginx(前置代理) + pgvector + redis + 监控。
> 配套文件（同目录）：
> `docker-compose.loadtest.yml` / `.env.loadtest` / `prometheus.yml` / `nginx.loadtest.conf`
> / `seed.py` / `locustfile.py` / `requirements.loadtest.txt`

## 前置

1. **Docker Desktop（Windows）已启动**，daemon 在跑（`docker ps` 能返回）。
2. **backend 代码已是最新**（含 `core/llm/loadtest_mock.py` 概率分流 + 观测 3 个 Gauge）。
3. **宿主装 Locust**（独立于 backend/venv）：
   ```bash
   pip install -r backend/scripts/loadtest/requirements.loadtest.txt
   ```
4. **DeepSeek 真 key**（仅真打模式需要；纯 mock 可跳过）。

---

## 0. 一键启动压测环境（全隔离，对齐生产）

```bash
cd backend/scripts/loadtest
docker compose -f docker-compose.loadtest.yml up -d --build
docker compose -f docker-compose.loadtest.yml ps          # 等 backend + worker 起来、backend healthy
```

- 首次 `--build` 要装 backend 依赖（uv sync），约 5–10 分钟；后续 build 走缓存秒级。
- **形态对齐生产 `docker-compose.yml`**：1 个 backend 容器（gunicorn -w4）+ 1 个 worker 容器（arq）+ nginx 前置。不再用"4 容器各 1 worker"（那是不存在的多 Pod 臆想形态）。
- backend healthy 后 nginx 才起（`depends_on: service_healthy`）；worker 等 backend healthy 后起。
- 端口：nginx `8000` / backend 宿主 `8102`（容器内 `8002`，gunicorn -w4）/ postgres `5433` / redis `6380` / prometheus `9090` / grafana `3000`。

> LLM 模式（`.env.loadtest`）：
> - 默认纯 mock（`LOAD_TEST_MOCK_LLM=1`，不烧钱）
> - 15% 真打：真 `LLM__API_KEY` + `LOAD_TEST_REAL_RATIO=0.15`
> - 全真打：`LOAD_TEST_MOCK_LLM=0`（烧 DeepSeek 额度、易撞上游限流，建议 5→10→20 阶梯）

## 1. 造数据（seed，缓存 token）

```bash
./backend/venv/Scripts/python.exe backend/scripts/loadtest/seed.py \
    --students 20 --courses 3 --out tokens.json
```

输出 `seeded 20 students + 3 courses → tokens.json`。教师升 admin 走 asyncpg 直连 DBA 通道。**Locust 不在压测期登录**（bcrypt CPU-bound 挡 greenlet）。

## 2. 跑 Locust

```bash
locust -f backend/scripts/loadtest/locustfile.py --host http://localhost:8000
```

浏览器开 http://localhost:8089 ：
- Number of users: **20**（≤ seed 的 student 数，1:1 避免撞 per-token 20/min 限流）
- Spawn rate: 2/s
- Host: http://localhost:8000

**阶梯加压**（每档稳态 5 min 收数据）：5 → 10 → 20。

headless（CI/无人值守）：
```bash
locust -f backend/scripts/loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u 20 -r 2 --run-time 5m
```

## 3. 观测（Google SRE 四黄金信号）

| 入口 | 看什么 |
|---|---|
| Locust Web UI（8089）| 实时 RPS / P50/P95 / 失败率，按 `chat_ttft` `chat_total` `quiz_ttft` `*_total` 分类 |
| Prometheus（9090）| PromQL 查 ca_* 业务指标 |
| Grafana（3000，admin/admin）| 加 Prometheus 数据源 + 自建看板 |
| 单 worker 指标 | `curl http://localhost:8102/metrics \| grep ca_` |

> ⚠️ **/metrics 只看得到 4 个 worker 之一**：项目用普通 prometheus_client（未接 `prometheus_multiprocess`），gunicorn 4 worker 各独立 REGISTRY，抓取只命中其一。**不影响 Locust 客户端测的总 RPS/P95/失败率**（那些才是 go/no-go 核心）。要看全量 worker 指标需接 multiprocess（生产代码改动，另议）。

**关键 PromQL**：

| 黄金信号 | PromQL |
|---|---|
| 延迟·TTFT P95 | `histogram_quantile(0.95, rate(ca_llm_first_token_seconds_bucket[1m]))` |
| 延迟·整 turn P95 | `histogram_quantile(0.95, rate(ca_turn_duration_seconds_bucket[1m]))` |
| 流量（turn/s）| `sum(rate(ca_turn_duration_seconds_count[1m]))` |
| 错误率（5xx）| `sum(rate(http_requests_total{status=~"5.."}[1m]))` |
| 饱和度·DB 连接池 | `ca_db_pool_checkedout`（抓到的那个 worker；应 < pool_size 的 80%）|
| 饱和度·实例池 | `ca_lightrag_in_use / ca_lightrag_instances` |
| leader 收敛 | `ca_leader_is_leader`（抓到的进程未必是 leader，**以日志为准**）|

## 4. 通过标准（go/no-go，20 并发稳态）

- [ ] P95 `ca_turn_duration` mock 模式 < 2s（真打看模型，P95 TTFT < 2s）
- [ ] 5xx 率 < 0.1%；429 率可控（限流阈值内）
- [ ] **silence 失败率 = 0**（H-15：所有连接都有 error/answer/done 收尾）
- [ ] `ca_db_pool_checkedout` 不饱和（< 容量 80%）
- [ ] **leader 恰好 1 个**（`docker compose ... logs backend \| grep -i leader`；metrics 抓不全，以日志为准）
- [ ] **浸泡 2h**：RSS（`process_resident_memory_bytes`）平稳不爬升、P95 不退化

## 5. 并发 bug 复验（H-7~H-15）

### H-11 熔断三态（CLOSED→OPEN→HALF_OPEN→CLOSED）
```bash
# 阶段 A：.env.loadtest 设 LOAD_TEST_FORCE_FAIL_RATIO=1.0，docker compose up -d，跑 30s
docker compose -f docker-compose.loadtest.yml logs backend | grep -i circuit   # 见 CLOSED→OPEN
# 阶段 B：改回 0，docker compose up -d，跑 30s
docker compose -f docker-compose.loadtest.yml logs backend | grep -i circuit   # 见 HALF_OPEN→CLOSED
```

### H-10 实例池 use-after-evict
- `.env.loadtest` 设 `LIGHTRAG__LRU_CAPACITY=4`，`seed.py --courses 5`
- locustfile 的 `run_research_ws` 把 `tools` 改 `["rag"]`
- 观测：`ca_lightrag_in_use` 接近 `ca_lightrag_instances`；日志见"evict 跳过：全部在用"；**无** use-after-evict / finalized storage 报错

### H-7/8/9 flush 数据丢失
```bash
# 压测期看 Redis buffer key 数（失败期间不应误降）；flush 由 worker 容器消费
docker compose -f docker-compose.loadtest.yml exec redis redis-cli --scan --pattern 'mem_flush:*' | wc -l
docker compose -f docker-compose.loadtest.yml logs worker | grep -i flush
```

## 6. 清理

```bash
docker compose -f docker-compose.loadtest.yml down -v   # -v 清 pg 数据卷，彻底还原
```

---

## 设计取舍（为什么是 1 容器 + worker + nginx）

- **对齐生产**：生产 `docker-compose.yml` 就是 1 backend(gunicorn -w4) + worker(arq) + frontend nginx。压测逐项复刻，压的就是要上线的东西。之前的"4 容器各 1 worker"是不存在的多 Pod 形态，已废弃。
- **必须有 worker**：KB 索引 / LLamaIndex 构建 / mem flush 由 arq worker 消费（`worker.py:269`），web 进程只投递（`main.py:280`）不消费。不加 worker 这些链路跑不了、不真实，H-7~H-9 也无法复验。
- **保留 nginx**：生产链路 nginx→backend，nginx `proxy_buffering off` 保证 SSE 首 token 实时透传、TTFT 真实。删 nginx 会漏测这层（生产 `frontend/nginx.conf` 的 buffering 已修为 off）。
- **metrics 看 1/4 是已知 trade-off**：未接 prometheus_multiprocess，不修（另议）；不影响 go/no-go 核心指标。

## 故障排查

| 现象 | 排查 |
|---|---|
| backend 一直 unhealthy | `docker compose logs backend`；常见 env 缺失（TESTING/LLM__API_KEY/DB__URL）|
| worker 起不来 / 不消费 | `docker compose logs worker`；确认 redis 通、`worker.WorkerSettings` 能 import |
| Locust 大量 `silence` 失败 | 查 nginx 是否 `proxy_buffering off`（否则 SSE 攒包）、backend log 有无异常静默关闭 |
| 大量 429 | per-token 20/min 限流（chat）；加 user 数或临时调高 `api/chat.py:49` 限流 |
| prometheus target down | `curl http://localhost:8102/metrics` 手动验；backend 未 healthy 时抓不到 |
| TTFT 异常高 | 先确认 nginx buffering off，再看 mock 的 `LOAD_TEST_MOCK_TTFT_MS` / 真 LLM 延迟 |
