#!/bin/sh
set -e

# Start backend (uvicorn)
cd /app/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start nginx (foreground)
nginx -g "daemon off;"
