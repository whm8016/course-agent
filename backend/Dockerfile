# Pinned: docker mirror (e.g. xuanyuan) often breaks floating tags (3.12-slim) with "content size of zero"
FROM python:3.12.11-slim-bookworm

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    libwmf-bin librsvg2-bin libcairo2-dev \
    gsfonts fonts-dejavu-core fonts-noto-cjk fontconfig && \
    fc-cache -f

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    -r requirements.txt

COPY . .

EXPOSE 8002

CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", \
     "--bind", "0.0.0.0:8002", \
     "--timeout", "300", \
     "--graceful-timeout", "30"]
