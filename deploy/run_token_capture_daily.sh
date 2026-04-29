#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${KINGOGPT_APP_DIR:-/home/eruin/kingoGPT}"
CONTAINER_NAME="${KINGOGPT_CONTAINER_NAME:-kingogpt-agent}"
LOG_FILE="$APP_DIR/state/token_capture.log"

cd "$APP_DIR"
mkdir -p "$APP_DIR/state"

docker exec "$CONTAINER_NAME" python /app/kingogpt_token_capture.py \
  --cache-file /app/state/kingogpt_token_cache.json \
  --config-file /app/state/kingogpt_config.json \
  --profile-dir /app/state/kingogpt_chrome_profile \
  >> "$LOG_FILE" 2>&1
