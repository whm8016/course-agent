# Pinned: docker mirror (e.g. xuanyuan) often breaks floating tags (3.12-slim) with "content size of zero"
FROM python:3.12.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002

CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", \
     "--bind", "0.0.0.0:8002", \
     "--timeout", "300", \
     "--graceful-timeout", "30"]
