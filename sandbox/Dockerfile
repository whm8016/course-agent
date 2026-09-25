# 与 backend 同钉 Python 版本：判分子进程环境与开发环境一致，升级成对改。
FROM python:3.12.11-slim-bookworm

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
ARG UV_INDEX_URL
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_INDEX_URL="${UV_INDEX_URL:-}" uv sync --frozen --no-dev --no-install-project

COPY app ./app

# 非 root：RLIMIT_NPROC 按 UID 计，root 下失效（backend/core/question/sandbox.py
# 风险披露②）。uds 目录先 chown——命名卷首次挂载时 Docker 拷贝镜像内目录的
# 属主/权限，65532 才能在卷里 bind socket。
RUN mkdir -p /run/sandbox && chown 65532:65532 /run/sandbox
USER 65532

ENV PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp

CMD [".venv/bin/python", "-m", "app.main"]
