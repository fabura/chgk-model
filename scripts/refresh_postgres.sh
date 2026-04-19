#!/usr/bin/env bash
# refresh_postgres.sh — download the most recent rating backup from R2 and
# re-restore the running postgres container in-place (without docker-compose
# down, so we keep volumes around for diffing if needed).
#
# Steps:
#   1. cd $RATING_DB_DIR (default: /Users/fbr/Projects/personal/rating-db)
#   2. download YYYY-MM-DD_rating.backup directly from R2, walking back day
#      by day until a valid (>=100 MB and `pg_restore --list` clean) archive
#      is found.  `download.sh` from the rating-db repo is intentionally
#      bypassed because it doesn't fail on HTTP 404 and we don't want to
#      clobber the prior backup with a 30 KB Cloudflare HTML error page.
#   3. docker-compose up -d             # start if not already
#   4. wait for pg_isready
#   5. exec /docker-entrypoint-initdb.d/restore.sh inside the container
#      (drops public schema, then pg_restore from /backup/rating.backup).
#      restore.sh exits 1 when matview refresh hits ignorable errors; we
#      accept 0/1 and use the tournaments row count as the success signal.
#
# Usage:
#   ./scripts/refresh_postgres.sh           # start from yesterday, walk back
#   ./scripts/refresh_postgres.sh --today   # start from today, walk back
#
# Env:
#   REFRESH_MAX_WALKBACK_DAYS  — how many older days to try if the start
#                                day is missing (default 7).
#
# Exit codes:
#   0 — postgres ready and re-restored
#   non-0 — fail (download or restore error)
set -euo pipefail

DBDIR="${RATING_DB_DIR:-/Users/fbr/Projects/personal/rating-db}"
DOWNLOAD_FLAG="${1:-}"

if [[ ! -d "$DBDIR" ]]; then
  echo "[refresh_postgres] error: rating-db dir not found at $DBDIR" >&2
  echo "                  set RATING_DB_DIR to override." >&2
  exit 2
fi

cd "$DBDIR"

# Preserve the previously good backup so a bad download doesn't clobber it
# until we've validated the new file.
PREV_BACKUP=""
if [[ -s rating.backup ]]; then
  PREV_BACKUP="rating.backup.prev"
  cp -p rating.backup "$PREV_BACKUP"
fi

MIN_BACKUP_BYTES=$((100 * 1024 * 1024))   # real dumps are ~800 MB; HTML 404 ~30 KB
R2_PREFIX="https://pub-5200ce7fb4b64b5ea3b6b0b0f05cfcd5.r2.dev"
MAX_WALKBACK_DAYS="${REFRESH_MAX_WALKBACK_DAYS:-7}"

# Try a specific YYYY-MM-DD: download to rating.backup and validate.
# Echoes 0 on success, non-zero on failure.  Caller decides whether to
# walk back to an older date.
try_download_date() {
  local d="$1"
  local url="$R2_PREFIX/${d}_rating.backup"
  echo "[refresh_postgres] trying ${d}_rating.backup ..."
  # --fail makes curl exit non-zero on HTTP errors so we don't write the
  # 404 HTML body over the prior backup.
  if ! curl --fail --location --silent --show-error -o rating.backup.tmp "$url"; then
    echo "[refresh_postgres]   curl: not available for $d"
    rm -f rating.backup.tmp
    return 1
  fi
  local sz
  sz=$(stat -f%z rating.backup.tmp 2>/dev/null || stat -c%s rating.backup.tmp)
  if [[ "$sz" -lt "$MIN_BACKUP_BYTES" ]]; then
    echo "[refresh_postgres]   too small ($sz bytes); skipping"
    rm -f rating.backup.tmp
    return 1
  fi
  if ! docker run --rm -v "$DBDIR/rating.backup.tmp:/backup/rating.backup:ro" \
        postgres:17 pg_restore --list /backup/rating.backup \
        >/tmp/refresh_pg_restore_list.log 2>&1; then
    echo "[refresh_postgres]   pg_restore --list failed; skipping"
    rm -f rating.backup.tmp
    return 1
  fi
  mv -f rating.backup.tmp rating.backup
  local toc
  toc=$(grep -c '^[0-9]' /tmp/refresh_pg_restore_list.log || true)
  echo "[refresh_postgres]   OK: $(du -h rating.backup | cut -f1), $toc TOC entries"
  return 0
}

# Pick start day: today (--today) or yesterday (default), then walk back
# day-by-day if R2 doesn't have it yet (backups appear ~23:00 UTC, so
# the previous day's file is sometimes still missing in the morning).
if [[ "$DOWNLOAD_FLAG" == "--today" ]]; then
  start_offset=0
else
  start_offset=1
fi

date_for_offset() {
  local off="$1"
  date -u -v "-${off}d" '+%Y-%m-%d' 2>/dev/null \
    || date -u -d "${off} days ago" '+%Y-%m-%d'
}

found=0
for off in $(seq "$start_offset" $((start_offset + MAX_WALKBACK_DAYS))); do
  d=$(date_for_offset "$off")
  if try_download_date "$d"; then
    found=1
    break
  fi
done
if [[ $found -ne 1 ]]; then
  echo "[refresh_postgres] error: no valid backup in last $MAX_WALKBACK_DAYS days." >&2
  [[ -n "$PREV_BACKUP" ]] && mv -f "$PREV_BACKUP" rating.backup
  exit 3
fi

# Validation passed — drop the saved previous backup.
[[ -n "$PREV_BACKUP" ]] && rm -f "$PREV_BACKUP"
backup_size=$(du -h rating.backup | cut -f1)
echo "[refresh_postgres] using ${d}_rating.backup ($backup_size)"

echo "[refresh_postgres] starting docker-compose (idempotent)..."
docker-compose up -d

echo "[refresh_postgres] waiting for postgres to be ready..."
for i in {1..60}; do
  if docker-compose exec -T postgres pg_isready -U postgres -q 2>/dev/null; then
    break
  fi
  sleep 2
done
if ! docker-compose exec -T postgres pg_isready -U postgres -q 2>/dev/null; then
  echo "[refresh_postgres] error: postgres not ready after 120s" >&2
  exit 4
fi

# ----------------------------------------------------------------------
# Run restore synchronously (no `tail -f --pid` — BSD tail on macOS
# ignores --pid and the parent hangs forever).  We accept exit codes 0
# (clean) and 1 (matview-refresh errors are typically benign), then rely
# on the row-count sanity check below to confirm the data is there.
# ----------------------------------------------------------------------
echo "[refresh_postgres] running restore.sh inside container (this can take 5–10 min)..."
restore_log=/tmp/refresh_postgres_restore.log
set +e
docker-compose exec -T postgres bash /docker-entrypoint-initdb.d/restore.sh \
  > "$restore_log" 2>&1
restore_rc=$?
set -e
echo "[refresh_postgres] restore.sh exited $restore_rc"
tail -20 "$restore_log" || true
if [[ $restore_rc -ne 0 && $restore_rc -ne 1 ]]; then
  echo "[refresh_postgres] error: restore.sh failed; see $restore_log" >&2
  exit $restore_rc
fi

# Sanity: tournaments table must exist + have rows.  This is the real
# success criterion — pg_restore can return 1 due to ignored matview
# refresh errors while still leaving all the data in place.
n=$(docker-compose exec -T postgres psql -U postgres -tAc \
  "SELECT COUNT(*) FROM public.tournaments" 2>/dev/null || echo 0)
n=$(echo "$n" | tr -d '[:space:]')
if [[ -z "$n" || "$n" -lt 1000 ]]; then
  echo "[refresh_postgres] error: public.tournaments has only $n rows after restore" >&2
  exit 5
fi
latest=$(docker-compose exec -T postgres psql -U postgres -tAc \
  "SELECT MAX(start_datetime)::date FROM public.tournaments WHERE start_datetime < NOW()" \
  2>/dev/null | tr -d '[:space:]')
echo "[refresh_postgres] OK — public.tournaments has $n rows, latest start: $latest"
