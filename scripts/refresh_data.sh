#!/usr/bin/env bash
# refresh_data.sh — full ChGK data refresh pipeline.
#
# Stages (any failure aborts; previous artefacts kept on disk):
#   1. refresh_postgres.sh             — pull yesterday's rating backup,
#                                        re-restore the running postgres
#   2. python -m rating --mode db ...  — pull observations into data.npz,
#                                        train sequential model, write
#                                        results/seq.npz (single CLI call:
#                                        the existing pipeline always
#                                        trains after pulling)
#   3. python -m website.build.build_db
#                                      — build website/data/chgk.duckdb.new
#   4. atomic mv .new → .duckdb        — switch over
#   5. POST /admin/reload-db           — hot-reload running uvicorn
#
# A single instance is enforced via a PID-file lock at logs/refresh.lock
# so cron can fire this without colliding with manual runs.
#
# Usage:
#   ./scripts/refresh_data.sh                 # full refresh
#   ./scripts/refresh_data.sh --skip-postgres # reuse current rating-db state
#   ./scripts/refresh_data.sh --skip-train    # reuse current data.npz + seq.npz
#   ./scripts/refresh_data.sh --skip-build    # don't rebuild DuckDB
#   SKIP_RELOAD=1 ./scripts/refresh_data.sh   # don't ping the website
#
# Env:
#   REPO_ROOT        — model repo root (default: this script's grandparent)
#   PYTHON           — python executable (default: $REPO_ROOT/.venv/bin/python)
#   QUESTIONS_DB     — path to chgk-embedings questions.db
#   WEB_URL          — base URL of the running website (default http://127.0.0.1:8765)
#   ADMIN_TOKEN      — auth token for /admin/reload-db (also read from
#                      $REPO_ROOT/website/.admin_token if present)
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
QUESTIONS_DB="${QUESTIONS_DB:-/Users/fbr/Projects/personal/chgk-embedings/data/questions.db}"
WEB_URL="${WEB_URL:-http://127.0.0.1:8765}"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOCK_FILE="$LOG_DIR/refresh.lock"
LOG_FILE="$LOG_DIR/refresh-$(date -u +%Y%m%dT%H%M%SZ).log"
LATEST_LINK="$LOG_DIR/refresh-latest.log"

# Portable single-instance guard via PID file (macOS lacks flock).
if [[ -f "$LOCK_FILE" ]]; then
  prev_pid=$(cat "$LOCK_FILE" 2>/dev/null || true)
  if [[ -n "$prev_pid" ]] && kill -0 "$prev_pid" 2>/dev/null; then
    echo "[refresh_data] another refresh is already running (pid $prev_pid, lock $LOCK_FILE)" >&2
    exit 75
  fi
  echo "[refresh_data] stale lock from pid $prev_pid found; reclaiming." >&2
fi
echo "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

ln -sfn "$LOG_FILE" "$LATEST_LINK"

SKIP_POSTGRES=0
SKIP_TRAIN=0
SKIP_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --skip-postgres) SKIP_POSTGRES=1 ;;
    --skip-train)    SKIP_TRAIN=1 ;;
    --skip-build)    SKIP_BUILD=1 ;;
    *) echo "[refresh_data] unknown arg: $arg" >&2; exit 64 ;;
  esac
done

CACHE="$REPO_ROOT/data.npz"
RESULTS="$REPO_ROOT/results/seq.npz"
DUCKDB="$REPO_ROOT/website/data/chgk.duckdb"
DUCKDB_NEW="$REPO_ROOT/website/data/chgk.duckdb.new"

step() { echo; echo "[refresh_data] $(date -u +%FT%TZ) === $* ==="; }

{
  step "Begin refresh (REPO_ROOT=$REPO_ROOT, log=$LOG_FILE)"
  echo "[refresh_data] PYTHON=$PYTHON"
  $PYTHON --version

  # ---------------------------------------------------------------- 1
  if [[ $SKIP_POSTGRES -eq 0 ]]; then
    step "Stage 1/3: refresh postgres rating backup"
    "$REPO_ROOT/scripts/refresh_postgres.sh"
  else
    step "Stage 1/3: SKIPPED (postgres already up-to-date)"
  fi

  # ---------------------------------------------------------------- 2
  # The CLI does pull-from-DB and train in a single call; --mode db
  # writes data.npz then runs run_sequential and exports seq.npz.
  if [[ $SKIP_TRAIN -eq 0 ]]; then
    step "Stage 2/3: pull data.npz from DB + train + export results/seq.npz"
    cd "$REPO_ROOT"
    cp -p "$CACHE" "$CACHE.bak" 2>/dev/null || true
    cp -p "$RESULTS" "$RESULTS.bak" 2>/dev/null || true
    "$PYTHON" -m rating --mode db --cache_file "$CACHE" \
      --results_npz "$RESULTS"
    test -s "$CACHE"
    test -s "$RESULTS"
    echo "[refresh_data] data.npz size: $(du -h "$CACHE" | cut -f1)"
    echo "[refresh_data] seq.npz size:  $(du -h "$RESULTS" | cut -f1)"
  else
    step "Stage 2/3: SKIPPED (using existing $CACHE + $RESULTS)"
  fi

  # ---------------------------------------------------------------- 3
  if [[ $SKIP_BUILD -eq 0 ]]; then
    step "Stage 3/3: rebuild website DuckDB"
    cd "$REPO_ROOT"
    rm -f "$DUCKDB_NEW"
    "$PYTHON" -m website.build.build_db \
      --cache "$CACHE" \
      --results "$RESULTS" \
      --questions-db "$QUESTIONS_DB" \
      --out "$DUCKDB_NEW"
    test -s "$DUCKDB_NEW"
    new_size=$(du -h "$DUCKDB_NEW" | cut -f1)
    echo "[refresh_data] new DuckDB size: $new_size"

    # Atomic swap. The running uvicorn keeps an fd to the old inode, so
    # there's no race here — it just keeps reading the previous version
    # until we POST /admin/reload-db below.
    if [[ -f "$DUCKDB" ]]; then
      mv -f "$DUCKDB" "$DUCKDB.old"
    fi
    mv -f "$DUCKDB_NEW" "$DUCKDB"
    rm -f "$DUCKDB.old" 2>/dev/null || true
    echo "[refresh_data] DuckDB swapped: $DUCKDB"
  else
    step "Stage 3/3: SKIPPED (DuckDB not rebuilt)"
  fi

  # ---------------------------------------------------------------- reload
  if [[ "${SKIP_RELOAD:-0}" == "1" ]]; then
    step "Reload: SKIPPED via SKIP_RELOAD=1"
  else
    step "Reload: ping running website at $WEB_URL/admin/reload-db"
    token="${ADMIN_TOKEN:-}"
    if [[ -z "$token" && -f "$REPO_ROOT/website/.admin_token" ]]; then
      token=$(tr -d '[:space:]' < "$REPO_ROOT/website/.admin_token")
    fi
    if [[ -z "$token" ]]; then
      echo "[refresh_data] WARN: no ADMIN_TOKEN — skipping hot reload."
      echo "                  set ADMIN_TOKEN env or write to website/.admin_token"
    else
      http_code=$(curl -sS -o /tmp/refresh_reload.out -w '%{http_code}' \
        -X POST -H "X-Admin-Token: $token" "$WEB_URL/admin/reload-db" || true)
      echo "[refresh_data] reload response: HTTP $http_code"
      cat /tmp/refresh_reload.out 2>/dev/null || true
      echo
      if [[ "$http_code" != "200" ]]; then
        echo "[refresh_data] WARN: reload did not return 200; restart uvicorn manually if needed."
      fi
    fi
  fi

  step "Done."
} 2>&1 | tee "$LOG_FILE"
