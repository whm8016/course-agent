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

# 安装 uv（官方推荐方式：从 astral-sh 镜像复制二进制）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 复制依赖定义文件
COPY pyproject.toml uv.lock ./

# 同步依赖（--frozen 确保不修改 uv.lock，--no-dev 排除开发依赖）
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 复制项目代码
COPY . .

EXPOSE 8002

CMD ["uv", "run", "--no-dev", "gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", \
     "--bind", "0.0.0.0:8002", \
     "--timeout", "300", \
     "--graceful-timeout", "30"]