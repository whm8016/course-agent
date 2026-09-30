#!/usr/bin/env sh
# 数据库备份：pg_dump 经 compose exec 导出（容器内 pg_dump 版本必配服务器，
# 免宿主机客户端版本不匹配）+ gzip + 按份数轮转。
#
# 用法（宿主机，cron 示例见 docs/DELIVERY.md 第七节）：
#   ./scripts/backup_db.sh                     # 默认 ./backups/，保留 14 份
#   BACKUP_DIR=/data/backups BACKUP_KEEP=30 ./scripts/backup_db.sh
#
# 恢复（先停 backend/worker 防写入竞争）：
#   docker compose stop backend worker
#   gunzip -c backups/course_agent_<stamp>.sql.gz | docker compose exec -T postgres psql -U postgres course_agent
#   docker compose start backend worker
set -eu

cd "$(dirname "$0")/.."  # 仓库根（docker-compose.yml 所在目录）

OUT_DIR="${BACKUP_DIR:-./backups}"
KEEP="${BACKUP_KEEP:-14}"
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
FILE="$OUT_DIR/course_agent_$STAMP.sql.gz"

docker compose exec -T postgres pg_dump -U postgres course_agent | gzip > "$FILE"

# 轮转：只保留最近 KEEP 份（本机快照；异机留存用 rsync/scp 另行拉走，防宿主机盘损）
ls -1t "$OUT_DIR"/course_agent_*.sql.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f

echo "backup -> $FILE ($(du -h "$FILE" | cut -f1))"
