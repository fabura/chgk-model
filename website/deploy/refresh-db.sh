#!/usr/bin/env bash
# Push only the DuckDB file to the server and reload the running app.
#
# Use this when the model was retrained / rebuilt and the website needs
# fresh data, but the image itself didn't change.
#
# Usage:
#   ./website/deploy/refresh-db.sh
#
# Reads the same .deploy.env as deploy.sh (REMOTE_HOST, REMOTE_USER,
# SSH_KEY, DEPLOY_DIR).  Existing env vars win over the file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-${SCRIPT_DIR}/.deploy.env}"
if [[ -f "${DEPLOY_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a; source "${DEPLOY_ENV_FILE}"; set +a
fi

if [[ -z "${REMOTE_HOST:-}" ]]; then
  echo "ERROR: REMOTE_HOST is not set." >&2
  echo "  Create ${DEPLOY_ENV_FILE} (see .deploy.env.example) or export REMOTE_HOST." >&2
  exit 2
fi
REMOTE_USER="${REMOTE_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
DEPLOY_DIR="${DEPLOY_DIR:-/srv/chgk-model}"

DB_LOCAL="${REPO_ROOT}/website/data/chgk.duckdb"

if [[ ! -f "${DB_LOCAL}" ]]; then
  echo "ERROR: ${DB_LOCAL} not found." >&2
  exit 5
fi

SSH="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new ${REMOTE_USER}@${REMOTE_HOST}"
log() { printf '\033[1;36m[refresh-db]\033[0m %s\n' "$*"; }

# Stage to a sibling file then atomically rename, so the running app
# always sees a complete file (rsync --inplace would leave a partial
# file visible mid-transfer).
log "Rsyncing DB → ${REMOTE_HOST}:${DEPLOY_DIR}/data/chgk.duckdb.new …"
# Apple's openrsync (default on macOS) doesn't speak --info=progress2;
# fall back to the older --progress flag there.
PROGRESS_FLAG="--info=progress2"
if rsync --version 2>&1 | head -1 | grep -qi openrsync; then
  PROGRESS_FLAG="--progress"
fi
rsync -ah --partial ${PROGRESS_FLAG} \
  -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new" \
  "${DB_LOCAL}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${DEPLOY_DIR}/data/chgk.duckdb.new"

log "Atomic swap on remote…"
$SSH "set -e
  cd ${DEPLOY_DIR}/data
  if [ -f chgk.duckdb ]; then
    mv -f chgk.duckdb chgk.duckdb.prev
  fi
  mv -f chgk.duckdb.new chgk.duckdb
  ls -lh chgk.duckdb*"

# We use restart instead of /admin/reload-db: cleaner when there are
# multiple uvicorn workers (reload only refreshes one), and downtime is
# under 2 seconds.
log "Restarting app…"
$SSH "cd ${DEPLOY_DIR} && docker compose restart app"

log "Health:"
$SSH "curl -fsS -o /dev/null -w '  HTTP %{http_code} (%{time_total}s)\n' http://127.0.0.1/ || echo '  health check FAILED'"

log "Done."
