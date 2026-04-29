#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-$HOME/kingoGPT}"
cd "$APP_DIR"
APP_DIR="$(pwd)"

mkdir -p state

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed on this server." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose is not available on this server." >&2
  exit 1
fi

docker compose up -d --build
docker compose ps
chmod +x deploy/run_token_capture_daily.sh

if command -v crontab >/dev/null 2>&1; then
  cron_file="$(mktemp)"
  if crontab -l >/tmp/kingogpt-current-cron 2>/dev/null; then
    grep -v "kingogpt_token_capture.py" /tmp/kingogpt-current-cron \
      | grep -v "run_token_capture_daily.sh" \
      | grep -v "KingoGPT daily token capture" \
      > "$cron_file" || true
  fi
  {
    echo ""
    echo "# KingoGPT daily token capture (04:10 KST / 19:10 UTC)"
    echo "10 19 * * * KINGOGPT_APP_DIR='$APP_DIR' bash '$APP_DIR/deploy/run_token_capture_daily.sh'"
  } >> "$cron_file"
  crontab -n "$cron_file" >/dev/null 2>&1 || true
  crontab "$cron_file"
  rm -f "$cron_file" /tmp/kingogpt-current-cron
else
  echo "crontab is not available; skipping daily token capture schedule." >&2
fi
