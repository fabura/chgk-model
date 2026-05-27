"""CLI: python -m rating_api [--since YYYY-MM-DD] [--limit N] [--dry-run]

Examples
--------
    # Auto-cursor from MAX(public.tournaments.last_edited_at), dry run,
    # only first 5 changed tournaments:
    python -m rating_api --limit 5 --dry-run

    # Force a specific cursor (debug):
    python -m rating_api --since 2026-05-25 --limit 20 --dry-run

F.2 supports --dry-run only; without it the CLI errors out cleanly.
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
        help="Don't write to public.* (F.2 requires this).",
    )
    p.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Override $DATABASE_URL.",
    )
    args = p.parse_args()

    if not args.dry_run:
        print(
            "error: F.2 only supports --dry-run; actual upsert lands in F.3.",
            file=sys.stderr,
        )
        return 2

    run_sync(
        since=args.since,
        limit=args.limit,
        dry_run=True,
        database_url=args.database_url,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
