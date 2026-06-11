#!/usr/bin/env bash
# refresh_production_server.sh — daily production refresh ON the VPS.
#
# API deltas → retrain → rebuild DuckDB (under /srv/chgk-model/data via
# symlink) → hot-reload the running app.
#
# Usage:
#   ./scripts/refresh_production_server.sh
#   ./scripts/refresh_production_server.sh --full-postgres
#
set -euo pipefail

export PATH="/root/.local/bin:${PATH}"

REPO_ROOT="${REPO_ROOT:-/opt/chgk-model}"
RATING_DB_DIR="${RATING_DB_DIR:-/opt/rating-db}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
QUESTIONS_DB="${QUESTIONS_DB:-$REPO_ROOT/data/questions.db}"
WEB_STACK_DIR="${WEB_STACK_DIR:-/srv/chgk-model}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
CRON_LOG="$LOG_DIR/production-cron.log"

FULL_POSTGRES=0
for arg in "$@"; do
  case "$arg" in
    --full-postgres) FULL_POSTGRES=1 ;;
    *) echo "[refresh_production_server] unknown arg: $arg" >&2; exit 64 ;;
  esac
done

refresh_args=(--skip-postgres --api-only)
if [[ $FULL_POSTGRES -eq 1 ]]; then
  refresh_args=()
fi

{
  echo
  echo "[refresh_production_server] $(date -u +%FT%TZ) begin (args: ${refresh_args[*]:-<full>})"

  if [[ -d "$RATING_DB_DIR" ]]; then
    (cd "$RATING_DB_DIR" && docker compose up -d >/dev/null)
  fi

  export PYTHON QUESTIONS_DB WEB_URL=http://127.0.0.1
  if [[ -f "$WEB_STACK_DIR/.env" ]]; then
    # shellcheck disable=SC2155
    export ADMIN_TOKEN="$(grep '^ADMIN_TOKEN=' "$WEB_STACK_DIR/.env" | cut -d= -f2-)"
  fi

  "$REPO_ROOT/scripts/refresh_data.sh" "${refresh_args[@]}"

  echo "[refresh_production_server] restarting app container…"
  (cd "$WEB_STACK_DIR" && docker compose restart app)

  echo "[refresh_production_server] $(date -u +%FT%TZ) done"
} 2>&1 | tee -a "$CRON_LOG"
