#!/usr/bin/env bash
# Build the chgk-model image locally, ship it + compose files to the
# server, and (re)start the stack.
#
# Usage:
#   ./website/deploy/deploy.sh                 # full deploy: image + compose files + DB if --db
#   ./website/deploy/deploy.sh --no-db         # default: skip DB upload (use refresh-db.sh for that)
#   ./website/deploy/deploy.sh --db            # also rsync the DuckDB file (slow, ~400 MB)
#   ./website/deploy/deploy.sh --image-only    # only rebuild & push the image, restart `app`
#
# Required: ssh access as $REMOTE_USER@$REMOTE_HOST with passwordless key.
#
# Configuration is read from website/deploy/.deploy.env (gitignored, see
# .deploy.env.example for the template).  Anything already exported in
# the environment wins over the file, so one-off overrides still work:
#   REMOTE_HOST=other.example.org ./deploy.sh

set -euo pipefail

# ---- locate repo root ----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- config -------------------------------------------------------------
# Load deployment settings (host, user, paths) from a non-committed env
# file so the server address is not baked into the repository.  Existing
# environment variables always win over the file (handy for one-off
# overrides like REMOTE_HOST=other.example.org ./deploy.sh).
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
IMAGE_NAME="${IMAGE_NAME:-chgk-model}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
# ---- derived paths ------------------------------------------------------
DB_LOCAL="${REPO_ROOT}/website/data/chgk.duckdb"
DB_ZST_LOCAL="${DB_LOCAL}.zst"
TAR_LOCAL="${REPO_ROOT}/website/data/${IMAGE_NAME}.tar"
TAR_ZST_LOCAL="${TAR_LOCAL}.zst"
REMOTE_TAR_ZST="${DEPLOY_DIR}/.image-stage/${IMAGE_NAME}.tar.zst"
REMOTE_DB_ZST="${DEPLOY_DIR}/data/chgk.duckdb.zst.new"

# ---- args ----------------------------------------------------------------
WITH_DB=0
IMAGE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --db)         WITH_DB=1 ;;
    --no-db)      WITH_DB=0 ;;
    --image-only) IMAGE_ONLY=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ServerAlive* keeps the SSH session alive when rsync goes idle on a
# slow uplink — without it the channel can drop after ~10 min and the
# whole transfer aborts (only --partial saves us).
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=20"
SSH="ssh ${SSH_OPTS} ${REMOTE_USER}@${REMOTE_HOST}"
SCP="scp ${SSH_OPTS}"
RSYNC="rsync -e 'ssh ${SSH_OPTS}'"

log() { printf '\033[1;36m[deploy]\033[0m %s\n' "$*"; }

# ---- 1. ensure remote layout --------------------------------------------
log "Ensuring remote layout at ${REMOTE_USER}@${REMOTE_HOST}:${DEPLOY_DIR}"
$SSH "mkdir -p ${DEPLOY_DIR}/data"

# ---- 2. ensure docker + zstd on remote ----------------------------------
log "Checking docker on remote…"
if ! $SSH "command -v docker >/dev/null 2>&1"; then
  echo "ERROR: docker is not installed on ${REMOTE_HOST}." >&2
  echo "Install with: curl -fsSL https://get.docker.com | sh" >&2
  exit 3
fi
log "Checking zstd on remote…"
if ! $SSH "command -v unzstd >/dev/null 2>&1"; then
  echo "ERROR: zstd is not installed on ${REMOTE_HOST}." >&2
  echo "Install with: apt update && apt install -y zstd" >&2
  exit 3
fi
if ! command -v zstd >/dev/null 2>&1; then
  echo "ERROR: zstd is not installed locally." >&2
  echo "Install with: brew install zstd  (macOS)  or  apt install zstd  (Linux)" >&2
  exit 3
fi

# ---- 3. build & ship image ----------------------------------------------
log "Building image ${IMAGE_NAME}:${IMAGE_TAG} (linux/amd64)…"
# --pull=false: don't refresh the base-image manifest from Docker Hub on
# every build.  When Docker Hub is rate-limited or slow this otherwise
# stalls the build for 15+ min on "load metadata for python:3.12-slim".
# Pass DEPLOY_PULL=1 if you actually want to refresh the base image.
DEPLOY_PULL="${DEPLOY_PULL:-0}"
PULL_FLAG="--pull=false"
if [[ "${DEPLOY_PULL}" == "1" ]]; then
  PULL_FLAG="--pull=true"
fi
docker buildx build \
  --platform linux/amd64 \
  ${PULL_FLAG} \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -f "${REPO_ROOT}/website/Dockerfile" \
  --load \
  "${REPO_ROOT}"

log "Saving image to ${TAR_LOCAL}…"
docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "${TAR_LOCAL}"
ls -lh "${TAR_LOCAL}"

# Re-compress only when the tar is newer than the cached .zst.
# zstd -19 saves ~30% bytes vs gzip on this image; decompression is
# always fast (~500 MB/s), so the upfront cost (~1 min on a workstation)
# is paid back many times over on slow uplinks.
if [[ ! -f "${TAR_ZST_LOCAL}" || "${TAR_LOCAL}" -nt "${TAR_ZST_LOCAL}" ]]; then
  log "Compressing image (zstd -19) → ${TAR_ZST_LOCAL}…"
  zstd -19 -T0 -fk -o "${TAR_ZST_LOCAL}" "${TAR_LOCAL}"
fi
ls -lh "${TAR_ZST_LOCAL}"

# --partial + --inplace: a killed transfer resumes from the byte where it
# stopped on the next deploy, instead of starting over.  Use --info=progress2
# when available (modern rsync 3.1+) and fall back to --progress for
# Apple's openrsync, which only supports the older flag.
PROGRESS_FLAG="--info=progress2"
if rsync --version 2>&1 | head -1 | grep -qi openrsync; then
  PROGRESS_FLAG="--progress"
fi

log "Rsyncing image to remote (resumable, with progress)…"
$SSH "mkdir -p $(dirname "${REMOTE_TAR_ZST}")"
eval $RSYNC -ah --inplace --partial ${PROGRESS_FLAG} \
  "${TAR_ZST_LOCAL}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TAR_ZST}"

log "Loading image on remote…"
$SSH "unzstd -c ${REMOTE_TAR_ZST} | docker load && rm -f ${REMOTE_TAR_ZST}"

# ---- 4. compose files (only when not --image-only) ----------------------
if [[ ${IMAGE_ONLY} -eq 0 ]]; then
  log "Uploading docker-compose.yml + nginx.conf + nginx-main.conf…"
  $SCP "${SCRIPT_DIR}/docker-compose.yml" "${SCRIPT_DIR}/nginx.conf" \
       "${SCRIPT_DIR}/nginx-main.conf" \
       "${REMOTE_USER}@${REMOTE_HOST}:${DEPLOY_DIR}/"

  log "Checking remote .env (must contain ADMIN_TOKEN)…"
  if ! $SSH "test -s ${DEPLOY_DIR}/.env && grep -q '^ADMIN_TOKEN=' ${DEPLOY_DIR}/.env"; then
    log "Remote .env missing or has no ADMIN_TOKEN — uploading template (EDIT IT before next deploy)."
    $SCP "${SCRIPT_DIR}/.env.example" "${REMOTE_USER}@${REMOTE_HOST}:${DEPLOY_DIR}/.env"
    $SSH "chmod 600 ${DEPLOY_DIR}/.env"
    echo "ACTION REQUIRED: ssh into the box and edit ${DEPLOY_DIR}/.env" >&2
    echo "  (set ADMIN_TOKEN to a real secret), then rerun ./deploy.sh" >&2
    exit 4
  fi
fi

# ---- 5. DB (only with --db) ---------------------------------------------
if [[ ${WITH_DB} -eq 1 ]]; then
  if [[ ! -f "${DB_LOCAL}" ]]; then
    echo "ERROR: ${DB_LOCAL} not found.  Build it first:" >&2
    echo "  python -m website.build.build_db --cache data.npz --results results/seq.npz --out website/data/chgk.duckdb" >&2
    exit 5
  fi
  if [[ ! -f "${DB_ZST_LOCAL}" || "${DB_LOCAL}" -nt "${DB_ZST_LOCAL}" ]]; then
    log "Compressing DuckDB (zstd -19) → ${DB_ZST_LOCAL}…"
    zstd -19 -T0 -fk -o "${DB_ZST_LOCAL}" "${DB_LOCAL}"
  fi
  log "Rsyncing DuckDB ($(du -h "${DB_LOCAL}" | cut -f1) raw → $(du -h "${DB_ZST_LOCAL}" | cut -f1) zst) to ${DEPLOY_DIR}/data/…"
  eval $RSYNC -ah --inplace --partial ${PROGRESS_FLAG} \
    "${DB_ZST_LOCAL}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DB_ZST}"
  log "Decompressing + atomic swap on remote…"
  $SSH "set -e
    cd ${DEPLOY_DIR}/data
    unzstd -f -o chgk.duckdb.new chgk.duckdb.zst.new
    rm -f chgk.duckdb.zst.new
    if [ -f chgk.duckdb ]; then mv -f chgk.duckdb chgk.duckdb.prev; fi
    mv -f chgk.duckdb.new chgk.duckdb
    ls -lh chgk.duckdb*"
fi

# ---- 6. (re)start stack -------------------------------------------------
if [[ ${IMAGE_ONLY} -eq 1 ]]; then
  log "Restarting only the app container…"
  $SSH "cd ${DEPLOY_DIR} && docker compose up -d --no-deps app"
else
  log "Bringing the stack up…"
  $SSH "cd ${DEPLOY_DIR} && docker compose up -d --remove-orphans"
fi

log "Status:"
$SSH "cd ${DEPLOY_DIR} && docker compose ps && curl -fsS -o /dev/null -w '  health: HTTP %{http_code} (%{time_total}s)\n' http://127.0.0.1/ || echo '  health: FAILED'"

log "Done. http://${REMOTE_HOST}/"
