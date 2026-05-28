#!/usr/bin/env python3
"""Calibration slices by venue size (mono vs multi-team venues).

Joins venue_overlay.duckdb with website chgk.duckdb team_games.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from venue_overlay.store import DEFAULT_DB_PATH  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue-db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument(
        "--chgk-db",
        type=Path,
        default=REPO_ROOT / "website" / "data" / "chgk.duckdb",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "venue_effects_slices.csv",
    )
    ap.add_argument(
        "--min-team-games",
        type=int,
        default=0,
        help="Only teams with at least this many sync games (with venue) in the sample",
    )
    args = ap.parse_args()

    if not args.venue_db.is_file():
        raise SystemExit(f"venue overlay missing: {args.venue_db} (run fetch_venue_overlay.py)")
    if not args.chgk_db.is_file():
        raise SystemExit(f"chgk.duckdb missing: {args.chgk_db}")

    import duckdb

    args.out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"ATTACH '{args.venue_db}' AS vo (READ_ONLY)")
    con.execute(f"ATTACH '{args.chgk_db}' AS site (READ_ONLY)")

    min_g = int(args.min_team_games)
    exp_cte = ""
    exp_join = ""
    exp_params: list[int] = []
    if min_g > 0:
        exp_cte = """
        experienced_teams AS (
            SELECT tg.team_id
            FROM site.team_games tg
            JOIN vo.team_tournament_venue ttv
              ON ttv.tournament_id = tg.tournament_id AND ttv.team_id = tg.team_id
            JOIN vo.venues v ON v.venue_id = ttv.venue_id
            JOIN site.tournaments t ON t.tournament_id = tg.tournament_id
            WHERE t.type = 'sync' AND NOT coalesce(v.is_online, false)
            GROUP BY tg.team_id
            HAVING count(*) >= ?
        ),
        """
        exp_join = "JOIN experienced_teams et ON et.team_id = tg.team_id\n        "
        exp_params = [min_g]
    df = con.execute(
        f"""
        WITH {exp_cte}
        SELECT
            CASE
                WHEN tv.is_mono THEN 'mono'
                WHEN tv.teams_played BETWEEN 2 AND 3 THEN '2-3'
                WHEN tv.teams_played BETWEEN 4 AND 10 THEN '4-10'
                ELSE '11+'
            END AS venue_bucket,
            COUNT(*) AS n_team_games,
            AVG(tg.score_actual - tg.expected_takes) AS mean_residual,
            MEDIAN(tg.score_actual - tg.expected_takes) AS median_residual,
            AVG(ABS(tg.score_actual - tg.expected_takes)) AS mean_abs_residual,
            MEDIAN(ABS(tg.score_actual - tg.expected_takes)) AS median_abs_residual,
            AVG(tg.score_actual) AS mean_actual,
            AVG(tg.expected_takes) AS mean_expected
        FROM site.team_games tg
        {exp_join}JOIN vo.team_tournament_venue ttv
          ON ttv.tournament_id = tg.tournament_id AND ttv.team_id = tg.team_id
        JOIN vo.tournament_venues tv
          ON tv.tournament_id = ttv.tournament_id AND tv.venue_id = ttv.venue_id
        JOIN vo.venues v ON v.venue_id = tv.venue_id
        JOIN site.tournaments t ON t.tournament_id = tg.tournament_id
        WHERE t.type = 'sync' AND NOT coalesce(v.is_online, false)
        GROUP BY 1
        ORDER BY 1
        """,
        exp_params,
    ).df()
    con.close()

    df.to_csv(args.out, index=False)
    if min_g > 0:
        print(f"(teams with >= {min_g} sync games with venue)\n")
    print(df.to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
