"""Diagnose systematic θ mis-estimation from stable-roster noisy-OR effects.

Level-1 metrics (read-only over baked DuckDB):

1. **Roster sticking** — for active veterans (≥200 games), what fraction of
   tournaments they are the min/max θ in the roster?
2. **Per-(player, mode) calibration** — cumulative ``actual − expected``
   takes; flags players whose θ persistently lags observed performance.
3. **Team-departure recovery** — players who left a core team (≥50 shared
   games) and played ≥20 games elsewhere: did θ move toward
   ``team_theta_implied`` after the split?

Outputs ``results/diagnostic_roster_sticking.csv`` (per-player rows) and
prints a short console summary including the three case-study players.

Usage::

    python scripts/diagnostic_roster_sticking.py \\
        --duckdb website/data/chgk.duckdb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import csv
from pathlib import Path

CASE_STUDY_IDS = (34909, 26818, 158668)  # Чернуха, Рекшинская, Монина


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    ap.add_argument("--out", default="results/diagnostic_roster_sticking.csv")
    ap.add_argument("--min_games", type=int, default=200)
    ap.add_argument(
        "--recent_games",
        type=int,
        default=50,
        help="window for recent actual−expected calibration",
    )
    args = ap.parse_args()

    try:
        import duckdb
    except ImportError:
        raise SystemExit("pip install duckdb")

    con = duckdb.connect(args.duckdb, read_only=True)

    # ------------------------------------------------------------------
    # 1. Per-game roster rank (one row per player × tournament)
    # ------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE TEMP TABLE per_game AS
        SELECT
            pg.player_id,
            pg.tournament_id,
            pg.team_id,
            t.type,
            t.start_date,
            tg.team_name,
            tg.team_theta_implied,
            tg.score_actual - tg.expected_takes AS overperf,
            p.theta AS model_theta,
            (
                SELECT MIN(p2.theta)
                FROM player_games pg2
                JOIN players p2 ON p2.player_id = pg2.player_id
                WHERE pg2.tournament_id = pg.tournament_id
                  AND pg2.team_id = pg.team_id
            ) AS roster_min_theta,
            (
                SELECT MAX(p2.theta)
                FROM player_games pg2
                JOIN players p2 ON p2.player_id = pg2.player_id
                WHERE pg2.tournament_id = pg.tournament_id
                  AND pg2.team_id = pg.team_id
            ) AS roster_max_theta,
            (
                SELECT AVG(p2.theta)
                FROM player_games pg2
                JOIN players p2 ON p2.player_id = pg2.player_id
                WHERE pg2.tournament_id = pg.tournament_id
                  AND pg2.team_id = pg.team_id
                  AND pg2.player_id != pg.player_id
            ) AS mate_avg_theta
        FROM player_games pg
        JOIN players p ON p.player_id = pg.player_id
        JOIN team_games tg
          ON tg.tournament_id = pg.tournament_id
         AND tg.team_id = pg.team_id
        JOIN tournaments t ON t.tournament_id = pg.tournament_id
    """)

    # ------------------------------------------------------------------
    # 2. Aggregate per player
    # ------------------------------------------------------------------
    rows = con.execute(f"""
        WITH base AS (
            SELECT
                pg.player_id,
                pl.last_name,
                pl.first_name,
                pl.theta AS final_theta,
                pl.games,
                pl.last_game_date,
                COUNT(*) AS n_games,
                AVG(CASE WHEN pg.model_theta <= pg.roster_min_theta + 0.001
                         THEN 1.0 ELSE 0.0 END) AS frac_lowest,
                AVG(CASE WHEN pg.model_theta >= pg.roster_max_theta - 0.001
                         THEN 1.0 ELSE 0.0 END) AS frac_highest,
                AVG(pg.team_theta_implied) AS avg_implied,
                AVG(pg.team_theta_implied) FILTER (
                    WHERE pg.start_date >= CURRENT_DATE - INTERVAL '365 days'
                ) AS avg_implied_1y,
                AVG(pg.overperf) AS avg_overperf,
                SUM(pg.overperf) AS total_overperf,
                AVG(pg.mate_avg_theta) AS avg_mate_theta,
                AVG(pg.mate_avg_theta - pg.model_theta) AS avg_mate_gap
            FROM per_game pg
            JOIN players pl ON pl.player_id = pg.player_id
            WHERE pl.games >= ?
              AND pl.last_game_date >= CURRENT_DATE - INTERVAL '365 days'
            GROUP BY 1, 2, 3, 4, 5, 6
        ),
        mode_cal AS (
            SELECT
                pg.player_id,
                pg.type,
                SUM(pg.overperf) AS mode_overperf,
                COUNT(*) AS mode_n
            FROM per_game pg
            GROUP BY 1, 2
        ),
        recent AS (
            SELECT player_id, SUM(overperf) AS recent_overperf, COUNT(*) AS recent_n
            FROM (
                SELECT
                    player_id,
                    overperf,
                    ROW_NUMBER() OVER (
                        PARTITION BY player_id ORDER BY start_date DESC
                    ) AS rn
                FROM per_game
            ) sub
            WHERE rn <= {int(args.recent_games)}
            GROUP BY 1
        )
        SELECT
            b.player_id,
            b.last_name,
            b.first_name,
            b.final_theta,
            b.games,
            b.n_games,
            b.frac_lowest,
            b.frac_highest,
            b.avg_implied,
            b.avg_implied_1y,
            b.avg_implied_1y - b.final_theta AS implied_gap_1y,
            b.total_overperf,
            b.avg_overperf,
            b.avg_mate_gap,
            r.recent_overperf,
            r.recent_n,
            MAX(CASE WHEN m.type = 'offline' THEN m.mode_overperf END) AS over_offline,
            MAX(CASE WHEN m.type = 'sync' THEN m.mode_overperf END) AS over_sync,
            MAX(CASE WHEN m.type = 'async' THEN m.mode_overperf END) AS over_async
        FROM base b
        LEFT JOIN mode_cal m ON m.player_id = b.player_id
        LEFT JOIN recent r ON r.player_id = b.player_id
        GROUP BY
            b.player_id, b.last_name, b.first_name, b.final_theta, b.games,
            b.n_games, b.frac_lowest, b.frac_highest, b.avg_implied,
            b.avg_implied_1y, b.total_overperf, b.avg_overperf,
            b.avg_mate_gap, r.recent_overperf, r.recent_n
        ORDER BY b.frac_lowest DESC, b.total_overperf DESC
    """, [args.min_games]).fetchdf()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # 3. Population summary
    # ------------------------------------------------------------------
    n = len(rows)
    stuck = int((rows["frac_lowest"] >= 0.70).sum())
    high_impl_gap = int((rows["implied_gap_1y"] >= 0.20).sum())
    pos_overperf = int((rows["total_overperf"] >= 200).sum())

    print(f"[ok] {n} active veterans (≥{args.min_games} games) → {out_path}")
    print(f"\n=== Roster sticking (frac_lowest) ===")
    print(f"  ≥70% lowest in roster: {stuck}/{n} ({100*stuck/n:.1f}%)")
    for q in (0.5, 0.75, 0.90, 0.95):
        print(f"  p{int(q*100)}: {rows['frac_lowest'].quantile(q):.2f}")

    print(f"\n=== Implied-vs-model gap (1y avg team_theta_implied − θ) ===")
    print(f"  ≥0.20 gap: {high_impl_gap}/{n} ({100*high_impl_gap/n:.1f}%)")
    for q in (0.5, 0.75, 0.90, 0.95):
        print(f"  p{int(q*100)}: {rows['implied_gap_1y'].quantile(q):+.3f}")

    print(f"\n=== Career over-performance (actual − expected, total) ===")
    print(f"  ≥+200 takes: {pos_overperf}/{n}")
    top_over = rows.nlargest(8, "total_overperf")
    for _, r in top_over.iterrows():
        print(
            f"  {r['first_name']} {r['last_name']:15} "
            f"θ={r['final_theta']:+.3f} total={r['total_overperf']:+.0f} "
            f"lowest={r['frac_lowest']:.0%}"
        )

    print(f"\n=== Case-study players ===")
    case = rows[rows["player_id"].isin(CASE_STUDY_IDS)]
    for _, r in case.iterrows():
        print(
            f"  {r['first_name']} {r['last_name']:15} "
            f"θ={r['final_theta']:+.3f}  "
            f"lowest={r['frac_lowest']:.0%}  "
            f"implied_gap_1y={r['implied_gap_1y']:+.3f}  "
            f"total_over={r['total_overperf']:+.0f}  "
            f"recent_{args.recent_games}={r['recent_overperf']:+.0f}  "
            f"async={r['over_async']:+.0f} offline={r['over_offline']:+.0f}"
        )

    # ------------------------------------------------------------------
    # 4. Team-departure recovery
    # ------------------------------------------------------------------
    departures = con.execute("""
        WITH pair_games AS (
            SELECT
                pg.player_id,
                tg.team_name,
                t.start_date,
                COUNT(*) OVER (
                    PARTITION BY pg.player_id, tg.team_name
                    ORDER BY t.start_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cum_on_team
            FROM player_games pg
            JOIN team_games tg
              ON tg.tournament_id = pg.tournament_id
             AND tg.team_id = pg.team_id
            JOIN tournaments t ON t.tournament_id = pg.tournament_id
        ),
        core_teams AS (
            SELECT player_id, team_name, MAX(cum_on_team) AS n_core
            FROM pair_games
            GROUP BY 1, 2
            HAVING MAX(cum_on_team) >= 50
        ),
        last_core AS (
            SELECT
                pg.player_id,
                ct.team_name AS core_team,
                MAX(t.start_date) AS last_core_date
            FROM player_games pg
            JOIN team_games tg
              ON tg.tournament_id = pg.tournament_id
             AND tg.team_id = pg.team_id
            JOIN tournaments t ON t.tournament_id = pg.tournament_id
            JOIN core_teams ct
              ON ct.player_id = pg.player_id
             AND ct.team_name = tg.team_name
            GROUP BY 1, 2
        ),
        post AS (
            SELECT
                lc.player_id,
                lc.core_team,
                lc.last_core_date,
                COUNT(*) FILTER (
                    WHERE t.start_date > lc.last_core_date
                      AND tg.team_name != lc.core_team
                ) AS n_post,
                AVG(ph.theta) FILTER (
                    WHERE t.start_date <= lc.last_core_date + INTERVAL '30 days'
                ) AS theta_at_exit,
                AVG(ph.theta) FILTER (
                    WHERE t.start_date >= lc.last_core_date + INTERVAL '365 days'
                ) AS theta_1y_after
            FROM last_core lc
            JOIN player_games pg ON pg.player_id = lc.player_id
            JOIN team_games tg
              ON tg.tournament_id = pg.tournament_id
             AND tg.team_id = pg.team_id
            JOIN tournaments t ON t.tournament_id = pg.tournament_id
            LEFT JOIN player_history ph
              ON ph.player_id = pg.player_id
             AND ph.tournament_id = pg.tournament_id
            GROUP BY 1, 2, 3
            HAVING n_post >= 20
               AND theta_at_exit IS NOT NULL
               AND theta_1y_after IS NOT NULL
        )
        SELECT
            player_id,
            core_team,
            last_core_date,
            n_post,
            theta_at_exit,
            theta_1y_after,
            theta_1y_after - theta_at_exit AS delta_1y
        FROM post
        ORDER BY delta_1y
        LIMIT 20
    """).fetchdf()

    dep_out = out_path.with_name("diagnostic_team_departures.csv")
    departures.to_csv(dep_out, index=False)
    print(f"\n=== Team-departure recovery (worst 10 by Δθ after 1y) ===")
    print(f"  → {dep_out}")
    for _, r in departures.head(10).iterrows():
        pl = con.execute(
            "SELECT first_name, last_name FROM players WHERE player_id=?",
            [int(r["player_id"])],
        ).fetchone()
        name = f"{pl[0]} {pl[1]}" if pl else str(r["player_id"])
        print(
            f"  {name:25} left '{str(r['core_team'])[:20]}' "
            f"θ {r['theta_at_exit']:+.3f}→{r['theta_1y_after']:+.3f} "
            f"(Δ={r['delta_1y']:+.3f}, n_post={int(r['n_post'])})"
        )

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
