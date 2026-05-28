#!/usr/bin/env python3
"""Teams / players who outperform on mono venues vs multi-team venues.

Compares median (actual − expected) on mono vs non-mono sync appearances.
Requires minimum counts on both sides (default 3 mono + 5 multi).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from venue_overlay.store import DEFAULT_DB_PATH  # noqa: E402

_EXPERIENCED_CTE = """
experienced_teams AS (
    SELECT tg.team_id
    FROM site.team_games tg
    JOIN vo.team_tournament_venue ttv
      ON ttv.tournament_id = tg.tournament_id AND ttv.team_id = tg.team_id
    JOIN vo.venues v ON v.venue_id = ttv.venue_id
    JOIN site.tournaments t ON t.tournament_id = tg.tournament_id
    WHERE t.type = 'sync' AND NOT coalesce(v.is_online, false)
    GROUP BY tg.team_id
    HAVING count(*) >= {min_team_games}
),
"""

TEAM_SQL = """
WITH {prefix}base AS (
    SELECT
        tg.team_id,
        tg.team_name,
        tg.score_actual - tg.expected_takes AS residual,
        tv.is_mono
    FROM site.team_games tg
    {exp_join}
    JOIN vo.team_tournament_venue ttv
      ON ttv.tournament_id = tg.tournament_id AND ttv.team_id = tg.team_id
    JOIN vo.tournament_venues tv
      ON tv.tournament_id = ttv.tournament_id AND tv.venue_id = ttv.venue_id
    JOIN vo.venues v ON v.venue_id = tv.venue_id
    JOIN site.tournaments t ON t.tournament_id = tg.tournament_id
    WHERE t.type = 'sync' AND NOT coalesce(v.is_online, false)
),
team_players AS (
    SELECT
        pg.team_id,
        string_agg(
            DISTINCT cast(pg.player_id AS VARCHAR),
            ','
            ORDER BY cast(pg.player_id AS VARCHAR)
        ) AS player_ids
    FROM site.player_games pg
    JOIN site.tournaments t ON t.tournament_id = pg.tournament_id
    WHERE t.type = 'sync'
    GROUP BY pg.team_id
),
agg AS (
    SELECT
        team_id,
        max(team_name) AS team_name,
        count(*) FILTER (WHERE is_mono) AS n_mono,
        count(*) FILTER (WHERE NOT is_mono) AS n_multi,
        median(residual) FILTER (WHERE is_mono) AS median_residual_mono,
        median(residual) FILTER (WHERE NOT is_mono) AS median_residual_multi,
        avg(residual) FILTER (WHERE is_mono) AS mean_residual_mono,
        avg(residual) FILTER (WHERE NOT is_mono) AS mean_residual_multi
    FROM base
    GROUP BY team_id
    HAVING count(*) FILTER (WHERE is_mono) >= ? AND count(*) FILTER (WHERE NOT is_mono) >= ?
)
SELECT
    a.team_id,
    a.team_name,
    tp.player_ids,
    a.n_mono,
    a.n_multi,
    a.median_residual_mono,
    a.median_residual_multi,
    a.median_residual_mono - a.median_residual_multi AS median_lift_mono,
    a.mean_residual_mono,
    a.mean_residual_multi,
    a.mean_residual_mono - a.mean_residual_multi AS mean_lift_mono
FROM agg a
LEFT JOIN team_players tp ON tp.team_id = a.team_id
ORDER BY median_lift_mono DESC
"""

PLAYER_SQL = """
WITH {prefix}base AS (
    SELECT
        pg.player_id,
        pg.team_id,
        pg.n_takes_team - pg.expected_takes_team AS residual,
        tv.is_mono
    FROM site.player_games pg
    {exp_join}
    JOIN vo.team_tournament_venue ttv
      ON ttv.tournament_id = pg.tournament_id AND ttv.team_id = pg.team_id
    JOIN vo.tournament_venues tv
      ON tv.tournament_id = ttv.tournament_id AND tv.venue_id = ttv.venue_id
    JOIN vo.venues v ON v.venue_id = tv.venue_id
    JOIN site.tournaments t ON t.tournament_id = pg.tournament_id
    WHERE t.type = 'sync' AND NOT coalesce(v.is_online, false)
),
mono_team_counts AS (
    SELECT player_id, team_id, count(*) AS cnt
    FROM base
    WHERE is_mono
    GROUP BY player_id, team_id
),
primary_mono_team AS (
    SELECT player_id, arg_max(team_id, cnt) AS primary_team_id_mono
    FROM mono_team_counts
    GROUP BY player_id
),
agg AS (
    SELECT
        player_id,
        count(*) FILTER (WHERE is_mono) AS n_mono,
        count(*) FILTER (WHERE NOT is_mono) AS n_multi,
        string_agg(
            DISTINCT cast(team_id AS VARCHAR),
            ','
            ORDER BY cast(team_id AS VARCHAR)
        ) FILTER (WHERE is_mono) AS team_ids_mono,
        string_agg(
            DISTINCT cast(team_id AS VARCHAR),
            ','
            ORDER BY cast(team_id AS VARCHAR)
        ) FILTER (WHERE NOT is_mono) AS team_ids_multi,
        median(residual) FILTER (WHERE is_mono) AS median_residual_mono,
        median(residual) FILTER (WHERE NOT is_mono) AS median_residual_multi,
        avg(residual) FILTER (WHERE is_mono) AS mean_residual_mono,
        avg(residual) FILTER (WHERE NOT is_mono) AS mean_residual_multi
    FROM base
    GROUP BY player_id
    HAVING count(*) FILTER (WHERE is_mono) >= ? AND count(*) FILTER (WHERE NOT is_mono) >= ?
)
SELECT
    a.player_id,
    p.last_name,
    p.first_name,
    p.theta,
    a.team_ids_mono,
    a.team_ids_multi,
    pmt.primary_team_id_mono,
    a.n_mono,
    a.n_multi,
    a.median_residual_mono,
    a.median_residual_multi,
    a.median_residual_mono - a.median_residual_multi AS median_lift_mono,
    a.mean_residual_mono,
    a.mean_residual_multi,
    a.mean_residual_mono - a.mean_residual_multi AS mean_lift_mono
FROM agg a
LEFT JOIN site.players p ON p.player_id = a.player_id
LEFT JOIN primary_mono_team pmt ON pmt.player_id = a.player_id
ORDER BY median_lift_mono DESC
"""

def _with_experience(template: str, min_team_games: int, *, player: bool) -> str:
    if min_team_games <= 0:
        return template.format(prefix="", exp_join="")
    alias = "pg" if player else "tg"
    exp_join = f"JOIN experienced_teams et ON et.team_id = {alias}.team_id\n    "
    return template.format(
        prefix=_EXPERIENCED_CTE.format(min_team_games=min_team_games),
        exp_join=exp_join,
    )


SUMMARY_SQL = """
SELECT
    count(*) AS n_entities,
    median(median_lift_mono) AS median_of_lifts,
    avg(median_lift_mono) AS mean_of_lifts,
    quantile_cont(median_lift_mono, 0.25) AS q25_lift,
    quantile_cont(median_lift_mono, 0.75) AS q75_lift
FROM ({subquery}) q
WHERE median_lift_mono IS NOT NULL
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue-db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument(
        "--chgk-db",
        type=Path,
        default=REPO_ROOT / "website" / "data" / "chgk.duckdb",
    )
    ap.add_argument("--min-mono", type=int, default=3)
    ap.add_argument("--min-multi", type=int, default=5)
    ap.add_argument(
        "--min-team-games",
        type=int,
        default=0,
        help="Only teams with at least this many sync games (with venue) in the sample",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=25,
        help="Print top-N by median_lift_mono",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results",
    )
    args = ap.parse_args()

    if not args.venue_db.is_file():
        raise SystemExit(f"venue overlay missing: {args.venue_db}")
    if not args.chgk_db.is_file():
        raise SystemExit(f"chgk.duckdb missing: {args.chgk_db}")

    import duckdb

    args.out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"ATTACH '{args.venue_db}' AS vo (READ_ONLY)")
    con.execute(f"ATTACH '{args.chgk_db}' AS site (READ_ONLY)")

    params = [args.min_mono, args.min_multi]
    team_sql = _with_experience(TEAM_SQL, args.min_team_games, player=False)
    player_sql = _with_experience(PLAYER_SQL, args.min_team_games, player=True)

    teams = con.execute(team_sql, params).df()
    players = con.execute(player_sql, params).df()

    teams_path = args.out_dir / "venue_mono_lift_teams.csv"
    players_path = args.out_dir / "venue_mono_lift_players.csv"
    teams.to_csv(teams_path, index=False)
    players.to_csv(players_path, index=False)

    if args.min_team_games > 0:
        print(f"(teams with >= {args.min_team_games} sync games with venue)\n")
    print(
        f"Teams with >={args.min_mono} mono and >={args.min_multi} multi sync games: {len(teams)}"
    )
    if len(teams):
        ts = con.execute(
            SUMMARY_SQL.format(
                subquery=team_sql.replace("ORDER BY median_lift_mono DESC", "")
            ),
            params,
        ).df()
        print("Team lift summary (median_residual_mono − median_residual_multi):")
        print(ts.to_string(index=False))
        print(f"\nTop {args.top} teams by median lift:")
        cols = [
            "team_id",
            "team_name",
            "n_mono",
            "n_multi",
            "median_residual_mono",
            "median_residual_multi",
            "median_lift_mono",
        ]
        print(teams[cols].head(args.top).to_string(index=False))

    print(
        f"\nPlayers with >={args.min_mono} mono and >={args.min_multi} multi sync games: {len(players)}"
    )
    if len(players):
        ps = con.execute(
            SUMMARY_SQL.format(
                subquery=player_sql.replace("ORDER BY median_lift_mono DESC", "")
            ),
            params,
        ).df()
        print("Player lift summary:")
        print(ps.to_string(index=False))
        print(f"\nTop {args.top} players by median lift:")
        cols = [
            "player_id",
            "last_name",
            "first_name",
            "n_mono",
            "n_multi",
            "median_residual_mono",
            "median_residual_multi",
            "median_lift_mono",
        ]
        print(players[cols].head(args.top).to_string(index=False))

    con.close()
    print(f"\nWrote {teams_path}")
    print(f"Wrote {players_path}")


if __name__ == "__main__":
    main()
