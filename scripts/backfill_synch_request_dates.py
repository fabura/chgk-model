#!/usr/bin/env python3
"""Backfill synch_requests.date_start from rating API into venue_overlay.duckdb."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from venue_overlay.api import fetch_synch_request_detail  # noqa: E402
from venue_overlay.store import DEFAULT_DB_PATH, ensure_schema, open_db, utc_now  # noqa: E402

ZURICH_SQL = """
SELECT DISTINCT ttv.synch_request_id
FROM team_tournament_venue ttv
JOIN venues v ON v.venue_id = ttv.venue_id
WHERE ttv.synch_request_id IS NOT NULL
  AND (v.name = 'Цюрих' OR v.name LIKE 'Цюрих /%')
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill synch request dateStart")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--zurich-only", action="store_true")
    ap.add_argument("--since", default=None, help="Only tournaments on/after YYYY-MM-DD (needs chgk.duckdb)")
    ap.add_argument("--chgk-db", type=Path, default=REPO_ROOT / "website/data/chgk.duckdb")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    con = open_db(args.db)
    ensure_schema(con)

    if args.zurich_only and args.since:
        con.execute(f"ATTACH '{args.chgk_db}' AS site (READ_ONLY)")
        ids = [
            int(r[0])
            for r in con.execute(
                ZURICH_SQL
                + """
                  AND ttv.tournament_id IN (
                    SELECT tournament_id FROM site.tournaments
                    WHERE start_date >= ?
                  )
                """,
                [args.since],
            ).fetchall()
        ]
    elif args.zurich_only:
        ids = [int(r[0]) for r in con.execute(ZURICH_SQL).fetchall()]
    elif args.since:
        con.execute(f"ATTACH '{args.chgk_db}' AS site (READ_ONLY)")
        ids = [
            int(r[0])
            for r in con.execute(
                """
                SELECT DISTINCT synch_request_id FROM team_tournament_venue
                WHERE synch_request_id IS NOT NULL
                  AND tournament_id IN (
                    SELECT tournament_id FROM site.tournaments WHERE start_date >= ?
                  )
                """,
                [args.since],
            ).fetchall()
        ]
    else:
        ids = [
            int(r[0])
            for r in con.execute(
                "SELECT DISTINCT synch_request_id FROM team_tournament_venue WHERE synch_request_id IS NOT NULL"
            ).fetchall()
        ]

    if args.limit is not None:
        ids = ids[: args.limit]

    print(f"Fetching {len(ids)} synch requests…")
    fetched_at = utc_now()
    ok = err = 0
    for i, srid in enumerate(ids, 1):
        detail = fetch_synch_request_detail(srid, sleep_sec=args.sleep)
        if detail is None:
            err += 1
            continue
        con.execute(
            """
            INSERT INTO synch_requests (
                synch_request_id, tournament_id, venue_id, date_start,
                status, approximate_teams_count, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (synch_request_id) DO UPDATE SET
                tournament_id = excluded.tournament_id,
                venue_id = excluded.venue_id,
                date_start = excluded.date_start,
                status = excluded.status,
                approximate_teams_count = excluded.approximate_teams_count,
                fetched_at = excluded.fetched_at
            """,
            [
                detail.synch_request_id,
                detail.tournament_id,
                detail.venue_id,
                detail.date_start,
                detail.status,
                detail.approximate_teams_count,
                fetched_at,
            ],
        )
        if detail.tournament_id is not None and detail.venue_id is not None:
            con.execute(
                """
                UPDATE tournament_venues
                SET date_start = ?, synch_request_id = ?
                WHERE tournament_id = ? AND venue_id = ?
                """,
                [
                    detail.date_start,
                    detail.synch_request_id,
                    detail.tournament_id,
                    detail.venue_id,
                ],
            )
        ok += 1
        if i % 20 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)} ok={ok} err={err}")

    con.close()
    print(f"Done: ok={ok} err={err}")


if __name__ == "__main__":
    main()
