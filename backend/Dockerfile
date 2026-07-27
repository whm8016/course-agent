# Pinned: docker mirror (e.g. xuanyuan) often breaks floating tags (3.12-slim) with "content size of zero"
FROM python:3.12.11-slim-bookworm

WORKDIR /app

# 安装系统依赖
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    libwmf-bin librsvg2-bin libcairo2-dev \
    gsfonts fonts-dejavu-core fonts-noto-cjk fontconfig && \
    fc-cache -f

# 安装 Node.js 22 LTS（NodeSource 官方源）
# 用途：供 stdio 类型的 MCP server（如 npx -y mcp-deepwiki@latest）在容器内运行。
# backend 与 worker 共用本镜像，故两者都能拉起 stdio MCP 子进程。
# 采用显式 keyring 写法而非 setup_22.x 脚本（脚本内容随版本易变、不可控）；
# 不用 Debian 仓库的 nodejs/npm——其 Node 18 已 EOL，且 npm/npx 版本偏旧、易与 MCP 包不兼容。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs

# 安装 uv（官方推荐方式：从 astral-sh 镜像复制二进制）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 复制依赖定义文件
COPY pyproject.toml uv.lock ./

# 同步依赖（--frozen 确保不修改 uv.lock，--no-dev 排除开发依赖）
# UV_INDEX_URL：国内 build 经 --build-arg 指向镜像源（如清华）加速 docling/torch 全家桶下载；
# 不传则用 PyPI 默认源，行为与原先一致。
ARG UV_INDEX_URL
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_INDEX_URL="${UV_INDEX_URL:-}" uv sync --frozen --no-dev --no-install-project

# 复制项目代码
COPY . .

EXPOSE 8002

# Prometheus 多进程模式：gunicorn -w 4 下，默认 in-process registry 让 /metrics 只反映接到请求的
# 那个 worker（≈1/4 流量）。设 PROMETHEUS_MULTIPROC_DIR 后，prometheus_client 把各 worker 的
# Counter/Histogram/Gauge 写入共享目录，/metrics 用 MultiProcessCollector 聚合全部 worker。
# /dev/shm 是 Docker 默认 tmpfs（内存盘），读写快且随容器销毁，适合存放这些临时聚合文件。
ENV PROMETHEUS_MULTIPROC_DIR=/dev/shm/prometheus_multiproc
CMD ["sh", "-c", "mkdir -p $PROMETHEUS_MULTIPROC_DIR && rm -rf $PROMETHEUS_MULTIPROC_DIR/* && exec uv run --no-dev gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8002 --timeout 300 --graceful-timeout 30"]