"""CLI: python -m rating_api [--since YYYY-MM-DD] [--limit N] [--dry-run]

Examples
--------
    # Default: auto-cursor from MAX(public.tournaments.last_edited_at),
    # write changes into the local rating PG.
    python -m rating_api

    # Cap to the first 5 changed tournaments (smoke test before a full run):
    python -m rating_api --limit 5

    # Force a specific cursor (debug):
    python -m rating_api --since 2026-05-25 --limit 20

    # Inspect-only mode — never touches public.*:
    python -m rating_api --dry-run
"""
from __future__ import annotations

import argparse
import sys

from rating_api.sync import run_sync


def main() -> int:
    p = argparse.ArgumentParser(prog="rating_api", description=__doc__)
    p.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO date YYYY-MM-DD; default = MAX(public.tournaments.last_edited_at)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many discovered tournaments.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write to public.* (only api_overlay.fetch_state is updated).",
    )
    p.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Override $DATABASE_URL.",
    )
    args = p.parse_args()

    run_sync(
        since=args.since,
        limit=args.limit,
        dry_run=args.dry_run,
        database_url=args.database_url,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
