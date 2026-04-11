# ---- Stage 1: Build frontend ----
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Backend + Nginx ----
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends nginx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Backend source
COPY backend/ ./backend/

# Frontend build output
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

# Nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Persistent data dirs
RUN mkdir -p /app/backend/data /app/backend/vectorstore /app/backend/uploads

EXPOSE 80

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
