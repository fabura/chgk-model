#!/usr/bin/env python3
"""Flag sync teams/players with suspicious mono-venue overperformance.

Heuristic "жулик"-profile (tunable via CLI):
  - enough mono AND multi games;
  - on mono: beats model (median residual > 0) by at least ``--min-mono-residual``;
  - lift mono−multi at least ``--min-lift``;
  - on multi: not a global overperformer (median residual ≤ ``--max-multi-residual``);
  - optional: exclude online venues;
  - bonus: many mono games at one venue (``venue_loyalty``).

Outputs ranked CSVs for manual review — not proof of cheating.
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


def _with_experience(template: str, min_team_games: int, *, player: bool) -> str:
    if min_team_games <= 0:
        return template.format(prefix="", exp_join="")
    alias = "pg" if player else "tg"
    exp_join = f"JOIN experienced_teams et ON et.team_id = {alias}.team_id\n    "
    return template.format(
        prefix=_EXPERIENCED_CTE.format(min_team_games=min_team_games),
        exp_join=exp_join,
    )


TEAM_SUSPECTS_SQL = """
WITH {prefix}game AS (
    SELECT
        tg.team_id,
        tg.team_name,
        tg.tournament_id,
        tg.score_actual,
        tg.expected_takes,
        tg.score_actual - tg.expected_takes AS residual,
        tg.team_theta_implied,
        tv.is_mono,
        tv.venue_id,
        v.name AS venue_name,
        coalesce(v.is_online, false) AS is_online
    FROM site.team_games tg
    {exp_join}
    JOIN vo.team_tournament_venue ttv
      ON ttv.tournament_id = tg.tournament_id AND ttv.team_id = tg.team_id
    JOIN vo.tournament_venues tv
      ON tv.tournament_id = ttv.tournament_id AND tv.venue_id = ttv.venue_id
    JOIN vo.venues v ON v.venue_id = tv.venue_id
    JOIN site.tournaments t ON t.tournament_id = tg.tournament_id
    WHERE t.type = 'sync'
      AND (? OR NOT coalesce(v.is_online, false))
),
team_agg AS (
    SELECT
        team_id,
        max(team_name) AS team_name,
        count(*) FILTER (WHERE is_mono) AS n_mono,
        count(*) FILTER (WHERE NOT is_mono) AS n_multi,
        median(residual) FILTER (WHERE is_mono) AS median_residual_mono,
        median(residual) FILTER (WHERE NOT is_mono) AS median_residual_multi,
        median(residual) FILTER (WHERE is_mono)
            - median(residual) FILTER (WHERE NOT is_mono) AS median_lift_mono,
        avg(score_actual) FILTER (WHERE is_mono) AS avg_score_mono,
        avg(expected_takes) FILTER (WHERE is_mono) AS avg_expected_mono,
        median(team_theta_implied) FILTER (WHERE is_mono) AS med_theta_impl_mono,
        median(team_theta_implied) FILTER (WHERE NOT is_mono) AS med_theta_impl_multi
    FROM game
    GROUP BY team_id
    HAVING count(*) FILTER (WHERE is_mono) >= ?
       AND count(*) FILTER (WHERE NOT is_mono) >= ?
),
venue_loyalty AS (
    SELECT
        team_id,
        arg_max(venue_id, cnt) AS top_venue_id,
        max(cnt) AS top_venue_n_mono,
        max(cnt)::DOUBLE / nullif(sum(cnt), 0) AS venue_loyalty
    FROM (
        SELECT team_id, venue_id, count(*) AS cnt
        FROM game
        WHERE is_mono
        GROUP BY team_id, venue_id
    ) x
    GROUP BY team_id
),
loyalty_names AS (
    SELECT vl.team_id, vl.top_venue_id, v.name AS top_venue_name,
           vl.top_venue_n_mono, vl.venue_loyalty
    FROM venue_loyalty vl
    LEFT JOIN vo.venues v ON v.venue_id = vl.top_venue_id
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
)
SELECT
    a.team_id,
    a.team_name,
    tp.player_ids,
    a.n_mono,
    a.n_multi,
    round(a.median_residual_mono, 3) AS median_residual_mono,
    round(a.median_residual_multi, 3) AS median_residual_multi,
    round(a.median_lift_mono, 3) AS median_lift_mono,
    round(a.avg_score_mono, 2) AS avg_score_mono,
    round(a.avg_expected_mono, 2) AS avg_expected_mono,
    round(a.med_theta_impl_mono, 3) AS med_theta_impl_mono,
    round(a.med_theta_impl_multi, 3) AS med_theta_impl_multi,
    round(a.med_theta_impl_mono - a.med_theta_impl_multi, 3) AS theta_impl_lift_mono,
    ln.top_venue_id,
    ln.top_venue_name,
    ln.top_venue_n_mono,
    round(ln.venue_loyalty, 3) AS venue_loyalty,
    round(
        a.median_lift_mono
        * sqrt(a.n_mono::DOUBLE)
        * greatest(a.median_residual_mono, 0)
        * (1.0 + 0.5 * coalesce(ln.venue_loyalty, 0)),
        3
    ) AS suspicion_score
FROM team_agg a
LEFT JOIN loyalty_names ln ON ln.team_id = a.team_id
LEFT JOIN team_players tp ON tp.team_id = a.team_id
WHERE a.median_lift_mono >= ?
  AND a.median_residual_mono >= ?
  AND a.median_residual_multi <= ?
ORDER BY suspicion_score DESC, median_lift_mono DESC
"""

PLAYER_SUSPECTS_SQL = """
WITH {prefix}game AS (
    SELECT
        pg.player_id,
        pg.team_id,
        pg.n_takes_team - pg.expected_takes_team AS residual,
        tv.is_mono,
        tv.venue_id,
        coalesce(v.is_online, false) AS is_online
    FROM site.player_games pg
    {exp_join}
    JOIN vo.team_tournament_venue ttv
      ON ttv.tournament_id = pg.tournament_id AND ttv.team_id = pg.team_id
    JOIN vo.tournament_venues tv
      ON tv.tournament_id = ttv.tournament_id AND tv.venue_id = ttv.venue_id
    JOIN vo.venues v ON v.venue_id = tv.venue_id
    JOIN site.tournaments t ON t.tournament_id = pg.tournament_id
    WHERE t.type = 'sync'
      AND (? OR NOT coalesce(v.is_online, false))
),
mono_team_counts AS (
    SELECT player_id, team_id, count(*) AS cnt
    FROM game
    WHERE is_mono
    GROUP BY player_id, team_id
),
primary_mono_team AS (
    SELECT player_id, arg_max(team_id, cnt) AS primary_team_id_mono
    FROM mono_team_counts
    GROUP BY player_id
),
player_agg AS (
    SELECT
        g.player_id,
        count(*) FILTER (WHERE g.is_mono) AS n_mono,
        count(*) FILTER (WHERE NOT g.is_mono) AS n_multi,
        count(DISTINCT g.team_id) FILTER (WHERE g.is_mono) AS n_teams_mono,
        string_agg(
            DISTINCT cast(g.team_id AS VARCHAR),
            ','
            ORDER BY cast(g.team_id AS VARCHAR)
        ) FILTER (WHERE g.is_mono) AS team_ids_mono,
        string_agg(
            DISTINCT cast(g.team_id AS VARCHAR),
            ','
            ORDER BY cast(g.team_id AS VARCHAR)
        ) FILTER (WHERE NOT g.is_mono) AS team_ids_multi,
        median(g.residual) FILTER (WHERE g.is_mono) AS median_residual_mono,
        median(g.residual) FILTER (WHERE NOT g.is_mono) AS median_residual_multi,
        median(g.residual) FILTER (WHERE g.is_mono)
            - median(g.residual) FILTER (WHERE NOT g.is_mono) AS median_lift_mono
    FROM game g
    GROUP BY g.player_id
    HAVING count(*) FILTER (WHERE g.is_mono) >= ?
       AND count(*) FILTER (WHERE NOT g.is_mono) >= ?
)
SELECT
    a.player_id,
    p.last_name,
    p.first_name,
    round(p.theta, 3) AS theta,
    a.team_ids_mono,
    a.team_ids_multi,
    pmt.primary_team_id_mono,
    a.n_mono,
    a.n_multi,
    a.n_teams_mono,
    round(a.median_residual_mono, 3) AS median_residual_mono,
    round(a.median_residual_multi, 3) AS median_residual_multi,
    round(a.median_lift_mono, 3) AS median_lift_mono,
    round(
        a.median_lift_mono
        * sqrt(a.n_mono::DOUBLE)
        * greatest(a.median_residual_mono, 0),
        3
    ) AS suspicion_score
FROM player_agg a
LEFT JOIN site.players p ON p.player_id = a.player_id
LEFT JOIN primary_mono_team pmt ON pmt.player_id = a.player_id
WHERE a.median_lift_mono >= ?
  AND a.median_residual_mono >= ?
  AND a.median_residual_multi <= ?
ORDER BY suspicion_score DESC, median_lift_mono DESC
"""

VENUE_HUBS_SQL = """
SELECT
    tv.venue_id,
    v.name AS venue_name,
    v.town_name,
    count(*) AS n_mono_games,
    count(DISTINCT ttv.team_id) AS n_distinct_teams,
    median(tg.score_actual - tg.expected_takes) AS median_residual
FROM vo.tournament_venues tv
JOIN vo.venues v ON v.venue_id = tv.venue_id
JOIN vo.team_tournament_venue ttv
  ON ttv.tournament_id = tv.tournament_id AND ttv.venue_id = tv.venue_id
JOIN site.team_games tg
  ON tg.tournament_id = ttv.tournament_id AND tg.team_id = ttv.team_id
JOIN site.tournaments t ON t.tournament_id = tg.tournament_id
WHERE tv.is_mono AND t.type = 'sync' AND NOT coalesce(v.is_online, false)
GROUP BY tv.venue_id, v.name, v.town_name
HAVING count(*) >= ?
ORDER BY n_mono_games DESC
LIMIT ?
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Suspicious mono-venue overperformance")
    ap.add_argument("--venue-db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument(
        "--chgk-db",
        type=Path,
        default=REPO_ROOT / "website" / "data" / "chgk.duckdb",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--min-mono", type=int, default=5)
    ap.add_argument("--min-multi", type=int, default=5)
    ap.add_argument(
        "--min-team-games",
        type=int,
        default=0,
        help="Only teams with at least this many sync games (with venue) in the sample",
    )
    ap.add_argument(
        "--min-lift",
        type=float,
        default=3.0,
        help="Median residual mono − multi (questions per game)",
    )
    ap.add_argument(
        "--min-mono-residual",
        type=float,
        default=1.0,
        help="Must beat model on mono (median actual − expected)",
    )
    ap.add_argument(
        "--max-multi-residual",
        type=float,
        default=0.0,
        help="On multi venues, median residual at most this (0 = no systematic overperformance)",
    )
    ap.add_argument(
        "--include-online",
        action="store_true",
        help="Include online venues (default: offline-ish only)",
    )
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-venue-mono-games", type=int, default=15)
    args = ap.parse_args()

    import duckdb

    args.out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"ATTACH '{args.venue_db}' AS vo (READ_ONLY)")
    con.execute(f"ATTACH '{args.chgk_db}' AS site (READ_ONLY)")

    include_online = args.include_online
    team_params = [
        include_online,
        args.min_mono,
        args.min_multi,
        args.min_lift,
        args.min_mono_residual,
        args.max_multi_residual,
    ]
    team_sql = _with_experience(TEAM_SUSPECTS_SQL, args.min_team_games, player=False)
    player_sql = _with_experience(PLAYER_SUSPECTS_SQL, args.min_team_games, player=True)
    teams = con.execute(team_sql, team_params).df()
    players = con.execute(
        player_sql,
        [
            include_online,
            args.min_mono,
            args.min_multi,
            args.min_lift,
            args.min_mono_residual,
            args.max_multi_residual,
        ],
    ).df()
    hubs = con.execute(
        VENUE_HUBS_SQL,
        [args.min_venue_mono_games, args.top],
    ).df()

    teams_path = args.out_dir / "venue_suspect_teams.csv"
    players_path = args.out_dir / "venue_suspect_players.csv"
    hubs_path = args.out_dir / "venue_mono_hubs.csv"
    teams.to_csv(teams_path, index=False)
    players.to_csv(players_path, index=False)
    hubs.to_csv(hubs_path, index=False)

    print("Criteria:")
    if args.min_team_games > 0:
        print(f"  team sync games (with venue)>={args.min_team_games}")
    print(f"  n_mono>={args.min_mono}, n_multi>={args.min_multi}")
    print(f"  median_lift>={args.min_lift}, median_residual_mono>={args.min_mono_residual}")
    print(f"  median_residual_multi<={args.max_multi_residual}")
    print(f"  online venues: {'included' if include_online else 'excluded'}")
    print()

    print(f"Suspicious teams: {len(teams)} → {teams_path}")
    if len(teams):
        show = [
            "team_id",
            "team_name",
            "player_ids",
            "n_mono",
            "n_multi",
            "median_residual_mono",
            "median_residual_multi",
            "median_lift_mono",
            "top_venue_name",
            "venue_loyalty",
            "suspicion_score",
        ]
        print(teams[show].head(args.top).to_string(index=False))

    print(f"\nSuspicious players: {len(players)} → {players_path}")
    if len(players):
        show = [
            "player_id",
            "last_name",
            "first_name",
            "theta",
            "team_ids_mono",
            "primary_team_id_mono",
            "n_mono",
            "n_multi",
            "median_lift_mono",
            "suspicion_score",
        ]
        print(players[show].head(args.top).to_string(index=False))

    print(f"\nFrequent mono venues (hubs): {hubs_path}")
    if len(hubs):
        print(hubs.head(min(15, len(hubs))).to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
