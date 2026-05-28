#!/usr/bin/env python3
"""Fetch sync venue assignments from api.rating.chgk.info into DuckDB.

Example:
  python scripts/fetch_venue_overlay.py --limit 20
  python scripts/fetch_venue_overlay.py --resume
  python scripts/fetch_venue_overlay.py --tournament-id 13606 --no-resume
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from venue_overlay.api import DEFAULT_API_BASE  # noqa: E402
from venue_overlay.fetch import (  # noqa: E402
    fetch_one_tournament,
    load_tournament_ids_from_cache,
    load_tournament_ids_from_db,
    print_summary,
    run_fetch,
)
from venue_overlay.store import DEFAULT_DB_PATH, ensure_schema, open_db  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch venue overlay from rating API")
    ap.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"DuckDB path (default: {DEFAULT_DB_PATH})",
    )
    ap.add_argument(
        "--api-base",
        default=os.environ.get("RATING_API_BASE", DEFAULT_API_BASE),
    )
    ap.add_argument(
        "--source",
        choices=("postgres", "cache"),
        default="postgres",
        help="Where to get tournament id list (default: postgres)",
    )
    ap.add_argument("--cache", type=Path, default=REPO_ROOT / "data.npz")
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--min-date", default="2015-01-01")
    ap.add_argument("--all-types", action="store_true", help="Not only sync tournaments")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tournament-id", type=int, action="append", dest="tournament_ids")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--no-approx", action="store_true", help="Skip synch request detail fetch")
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    if args.tournament_ids:
        tids = list(dict.fromkeys(args.tournament_ids))
    elif args.source == "cache":
        if not args.cache.is_file():
            ap.error(f"cache not found: {args.cache}")
        tids = load_tournament_ids_from_cache(
            args.cache,
            sync_only=not args.all_types,
            limit=args.limit,
        )
    else:
        tids = load_tournament_ids_from_db(
            database_url=args.database_url,
            min_tournament_date=args.min_date,
            sync_only=not args.all_types,
            limit=args.limit,
        )

    print(f"Tournaments to process: {len(tids)} (source={args.source}, resume={args.resume})")

    if len(tids) == 1 and not args.resume:
        con = open_db(args.db)
        ensure_schema(con)
        fetch_one_tournament(
            con,
            tids[0],
            api_base=args.api_base,
            fetch_approx=not args.no_approx,
            sleep_sec=args.sleep,
            timeout_sec=args.timeout,
        )
        con.close()
        print_summary(args.db)
        return

    stats = run_fetch(
        tids,
        db_path=args.db,
        api_base=args.api_base,
        resume=args.resume,
        fetch_approx=not args.no_approx,
        sleep_sec=args.sleep,
        timeout_sec=args.timeout,
        show_progress=not args.no_progress,
    )
    print(f"Done: ok={stats['ok']} err={stats['err']} skipped(resume)={stats['skipped']}")
    print_summary(args.db)


if __name__ == "__main__":
    main()
