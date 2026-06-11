#!/usr/bin/env bash
# One-time VPS bootstrap for server-side daily refresh.
#
# Run ON the server as root (or via: ssh root@host 'bash -s' < setup_server_refresh.sh)
#
# Creates:
#   /opt/chgk-model     — model repo + venv + artefacts
#   /opt/rating-db      — postgres for rating data
#   symlink website/data → /srv/chgk-model/data
#   4 GiB swap if none configured
#   crontab entry at 02:00 UTC
#
# After this script, rsync large seed files from the workstation (see README
# block at the end) and run refresh_production_server.sh once manually.
#
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/fabura/chgk-model.git}"
RATING_DB_URL="${RATING_DB_URL:-https://github.com/chgk-gg/rating-db.git}"
REPO_ROOT="${REPO_ROOT:-/opt/chgk-model}"
RATING_DB_DIR="${RATING_DB_DIR:-/opt/rating-db}"
WEB_STACK_DIR="${WEB_STACK_DIR:-/srv/chgk-model}"

log() { printf '[setup_server_refresh] %s\n' "$*"; }

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root" >&2
  exit 1
fi

# ---- swap (training needs headroom on 2 GiB VPS) -------------------------
if ! swapon --show | grep -q .; then
  if [[ ! -f /swapfile ]]; then
    log "creating 4G swapfile…"
    fallocate -l 4G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=4096
    chmod 600 /swapfile
    mkswap /swapfile
  fi
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  log "swap enabled: $(swapon --show)"
fi

# ---- python 3.12 via uv (Ubuntu 20.04 has no deadsnakes 3.12 packages) ---
export PATH="/root/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if ! uv python find 3.12 >/dev/null 2>&1; then
  log "installing python 3.12 via uv…"
  uv python install 3.12
fi
apt-get install -y build-essential >/dev/null 2>&1 || true

# ---- repos -----------------------------------------------------------------
if [[ ! -d "$REPO_ROOT/.git" ]]; then
  log "cloning chgk-model → $REPO_ROOT"
  git clone "$REPO_URL" "$REPO_ROOT"
fi
if [[ ! -d "$RATING_DB_DIR/.git" ]]; then
  log "cloning rating-db → $RATING_DB_DIR"
  git clone "$RATING_DB_URL" "$RATING_DB_DIR"
fi

mkdir -p "$REPO_ROOT/data" "$REPO_ROOT/results" "$REPO_ROOT/logs" "$WEB_STACK_DIR/data"
ln -sfn "$WEB_STACK_DIR/data" "$REPO_ROOT/website/data"

# ---- venv ------------------------------------------------------------------
if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  log "creating venv + installing requirements…"
  uv venv --python 3.12 "$REPO_ROOT/.venv"
  uv pip install --python "$REPO_ROOT/.venv/bin/python" -r "$REPO_ROOT/requirements.txt"
fi

chmod +x "$REPO_ROOT/scripts/refresh_production_server.sh" \
         "$REPO_ROOT/scripts/refresh_data.sh" \
         "$REPO_ROOT/scripts/refresh_postgres.sh" 2>/dev/null || true

# ---- cron ------------------------------------------------------------------
CRON_LINE="0 2 * * * $REPO_ROOT/scripts/refresh_production_server.sh"
( crontab -l 2>/dev/null | grep -v 'refresh_production' || true
  echo "# ChGK: daily API sync → retrain → reload site (02:00 UTC)"
  echo "$CRON_LINE"
) | crontab -
log "crontab:"
crontab -l

cat <<EOF

=== manual seed steps (run from workstation) ===

# 1. Rating postgres dump + questions DB (one-time, ~1.5 GB):
rsync -ah --progress -e "ssh -i ~/.ssh/id_rsa" \\
  /Users/fbr/Projects/personal/rating-db/rating.backup \\
  root@REMOTE:/opt/rating-db/rating.backup
rsync -ah --progress -e "ssh -i ~/.ssh/id_rsa" \\
  /Users/fbr/Projects/personal/chgk-embedings/data/questions.db \\
  root@REMOTE:$REPO_ROOT/data/questions.db

# 2. Start postgres (first boot restores from rating.backup):
cd /opt/rating-db && docker compose up -d

# 3. Optional: copy current artefacts to skip the first ~30 min train:
rsync -ah -e "ssh -i ~/.ssh/id_rsa" \\
  /Users/fbr/Projects/personal/сhgk-model/data.npz \\
  /Users/fbr/Projects/personal/сhgk-model/results/seq.npz \\
  root@REMOTE:$REPO_ROOT/

# 4. Smoke test:
$REPO_ROOT/scripts/refresh_production_server.sh

EOF

log "bootstrap shell done — complete the seed rsync steps above."
