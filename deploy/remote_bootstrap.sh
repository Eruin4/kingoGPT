#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-$HOME/kingoGPT}"
cd "$APP_DIR"

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

