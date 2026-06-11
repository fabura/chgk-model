#!/usr/bin/env bash
# refresh_production.sh — daily production update FROM a workstation:
# pull rating API deltas, retrain, rebuild DuckDB, rsync to the VPS.
#
# Prefer running ON the server instead: scripts/refresh_production_server.sh
# (see website/deploy/setup_server_refresh.sh).  Safe for cron: refresh_data.sh
# enforces a PID lock at logs/refresh.lock.
#
# Usage:
#   ./scripts/refresh_production.sh                  # API deltas only (default)
#   ./scripts/refresh_production.sh --full-postgres  # also restore latest R2 dump
#
# Env (passed through to child scripts):
#   REPO_ROOT, PYTHON, QUESTIONS_DB, REMOTE_HOST, REMOTE_USER, SSH_KEY, DEPLOY_DIR
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
CRON_LOG="$LOG_DIR/production-cron.log"

FULL_POSTGRES=0
for arg in "$@"; do
  case "$arg" in
    --full-postgres) FULL_POSTGRES=1 ;;
    *) echo "[refresh_production] unknown arg: $arg" >&2; exit 64 ;;
  esac
done

refresh_args=(--skip-postgres --api-only)
if [[ $FULL_POSTGRES -eq 1 ]]; then
  refresh_args=()
fi

{
  echo
  echo "[refresh_production] $(date -u +%FT%TZ) begin (args: ${refresh_args[*]:-<full>})"

  # Ensure rating-db postgres is up (idempotent).
  RATING_DB_DIR="${RATING_DB_DIR:-/Users/fbr/Projects/personal/rating-db}"
  if [[ -d "$RATING_DB_DIR" ]]; then
    (cd "$RATING_DB_DIR" && docker compose up -d >/dev/null)
  fi

  SKIP_RELOAD=1 "$REPO_ROOT/scripts/refresh_data.sh" "${refresh_args[@]}"
  "$REPO_ROOT/website/deploy/refresh-db.sh"

  echo "[refresh_production] $(date -u +%FT%TZ) done"
} 2>&1 | tee -a "$CRON_LOG"
