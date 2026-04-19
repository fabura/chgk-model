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
DB_ZST_LOCAL="${DB_LOCAL}.zst"
REMOTE_DB_ZST="${DEPLOY_DIR}/data/chgk.duckdb.zst.new"

if [[ ! -f "${DB_LOCAL}" ]]; then
  echo "ERROR: ${DB_LOCAL} not found." >&2
  exit 5
fi
if ! command -v zstd >/dev/null 2>&1; then
  echo "ERROR: zstd not installed locally (brew install zstd / apt install zstd)." >&2
  exit 6
fi

SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=20"
SSH="ssh ${SSH_OPTS} ${REMOTE_USER}@${REMOTE_HOST}"
log() { printf '\033[1;36m[refresh-db]\033[0m %s\n' "$*"; }

if ! $SSH "command -v unzstd >/dev/null 2>&1"; then
  echo "ERROR: zstd not installed on ${REMOTE_HOST}.  apt install -y zstd" >&2
  exit 6
fi

# Compress only when the raw .duckdb is newer than the cached .zst.
# zstd -19 cuts ~373 MB → ~125 MB on this DB; decompression is always
# fast (~500 MB/s) so the per-deploy CPU spend (~25 s on workstation,
# ~1 s on remote) easily pays for itself on slow uplinks.
if [[ ! -f "${DB_ZST_LOCAL}" || "${DB_LOCAL}" -nt "${DB_ZST_LOCAL}" ]]; then
  log "Compressing DB (zstd -19) → ${DB_ZST_LOCAL}…"
  zstd -19 -T0 -fk -o "${DB_ZST_LOCAL}" "${DB_LOCAL}"
fi

# Apple's openrsync (default on macOS) doesn't speak --info=progress2;
# fall back to the older --progress flag there.
PROGRESS_FLAG="--info=progress2"
if rsync --version 2>&1 | head -1 | grep -qi openrsync; then
  PROGRESS_FLAG="--progress"
fi

log "Rsyncing DB ($(du -h "${DB_LOCAL}" | cut -f1) raw → $(du -h "${DB_ZST_LOCAL}" | cut -f1) zst, resumable) …"
rsync -ah --inplace --partial ${PROGRESS_FLAG} \
  -e "ssh ${SSH_OPTS}" \
  "${DB_ZST_LOCAL}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DB_ZST}"

# Decompress to a sibling file then atomically rename, so the running
# app always sees a complete file.
log "Decompressing + atomic swap on remote…"
$SSH "set -e
  cd ${DEPLOY_DIR}/data
  unzstd -f -o chgk.duckdb.new chgk.duckdb.zst.new
  rm -f chgk.duckdb.zst.new
  if [ -f chgk.duckdb ]; then mv -f chgk.duckdb chgk.duckdb.prev; fi
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
