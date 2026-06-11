"""Offline test of the adaptive-learning-rate hypothesis.

The idea behind a residual-aware / Adam-style per-player learning rate is:
if a player's per-game residual ``r = actual − expected`` is *persistently*
one-signed, their θ is stale/biased and the slow ``η = η0/√games`` schedule
cannot catch up.  If residuals are memoryless noise, boosting η would just
chase noise and hurt.

This script measures, **read-only over the baked DuckDB**, whether past
residual predicts future residual — the precondition for the idea to work.
No engine change; this is the go/no-go gate.

Key metrics (per-question-normalised residual ``r = (actual−expected)/n_q``):

1. **Headline persistence** — pooled ``corr`` and OLS slope of the current
   game's residual on a trailing mean of the player's previous 10 games,
   restricted to veterans.  High slope ⇒ stale θ ⇒ adaptive η has signal.
2. **By experience bucket** — does persistence *rise* with cumulative games?
   If yes, the 1/√games decay is demonstrably too slow for veterans.
3. **Carry-game persistence** — same test on games where teammates are
   weaker than the player (``mate_avg_theta < player_theta``), where the
   player's own θ dominates the team prediction.  This separates a genuine
   under-rating from a "passenger on a strong team" artefact.
4. **Block split** — per-player correlation between the mean residual of the
   prior window and the recent window (coarse form persistence).
5. **Case studies** — recent vs prior residual for the three flagged players.

Decision rule (printed at the end):
    GO   if veteran slope ≥ 0.15 AND carry-game slope ≥ 0.10
         AND persistence rises from rookie→veteran bucket.
    WEAK if 0.05 ≤ slope < 0.15.
    NO-GO if slope < 0.05 (residuals are ~memoryless; adaptive η won't help).

Usage::

    python scripts/diagnostic_residual_persistence.py \\
        --duckdb website/data/chgk.duckdb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CASE_STUDY = {34909: "Чернуха", 26818: "Рекшинская", 158668: "Монина"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    ap.add_argument("--out", default="results/diagnostic_residual_persistence.csv")
    ap.add_argument("--min_games", type=int, default=150)
    ap.add_argument(
        "--trail",
        type=int,
        default=10,
        help="number of preceding games in the trailing-mean predictor",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=30,
        help="block size for the prior-vs-recent split test",
    )
    args = ap.parse_args()

    try:
        import duckdb
    except ImportError:
        raise SystemExit("pip install duckdb")

    con = duckdb.connect(args.duckdb, read_only=True)

    # ------------------------------------------------------------------
    # Per (tournament, team) roster θ sum/count, so per-player teammate
    # mean θ is (sum − own)/(cnt − 1) without a correlated subquery.
    # ------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE TEMP TABLE roster_theta AS
        SELECT
            pg.tournament_id,
            pg.team_id,
            SUM(p.theta) AS sum_theta,
            COUNT(*)     AS cnt
        FROM player_games pg
        JOIN players p ON p.player_id = pg.player_id
        GROUP BY 1, 2
    """)

    # ------------------------------------------------------------------
    # Per (player, tournament) normalised residual + context.
    # residual uses team-level score vs expected (consistent with the
    # earlier diagnostics); normalised by question count so offline
    # (60–90 Q) and async (~30 Q) are comparable.
    # ------------------------------------------------------------------
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pg_resid AS
        SELECT
            pg.player_id,
            pg.tournament_id,
            t.start_date,
            t.type,
            GREATEST(t.n_questions, 1) AS n_q,
            (tg.score_actual - tg.expected_takes)
                / GREATEST(t.n_questions, 1) AS resid_norm,
            p.theta AS player_theta,
            CASE WHEN rt.cnt > 1
                 THEN (rt.sum_theta - p.theta) / (rt.cnt - 1)
                 ELSE NULL END AS mate_avg_theta,
            rt.cnt AS roster_size
        FROM player_games pg
        JOIN team_games tg
          ON tg.tournament_id = pg.tournament_id AND tg.team_id = pg.team_id
        JOIN tournaments t ON t.tournament_id = pg.tournament_id
        JOIN players p ON p.player_id = pg.player_id
        JOIN roster_theta rt
          ON rt.tournament_id = pg.tournament_id AND rt.team_id = pg.team_id
        WHERE tg.expected_takes IS NOT NULL
          AND tg.score_actual IS NOT NULL
          AND p.games >= ?
    """, [args.min_games])

    # ------------------------------------------------------------------
    # Chronological per-player sequence + trailing-mean predictor.
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE seq AS
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY player_id ORDER BY start_date, tournament_id
            ) AS game_no,
            COUNT(*) OVER (PARTITION BY player_id) AS total_games,
            AVG(resid_norm) OVER (
                PARTITION BY player_id ORDER BY start_date, tournament_id
                ROWS BETWEEN {int(args.trail)} PRECEDING AND 1 PRECEDING
            ) AS trail_resid,
            (mate_avg_theta < player_theta) AS is_carry
        FROM pg_resid
    """)

    def persistence(where: str) -> tuple[float, float, int]:
        row = con.execute(f"""
            SELECT
                corr(resid_norm, trail_resid),
                regr_slope(resid_norm, trail_resid),
                COUNT(*)
            FROM seq
            WHERE trail_resid IS NOT NULL AND {where}
        """).fetchone()
        c = float(row[0]) if row[0] is not None else float("nan")
        s = float(row[1]) if row[1] is not None else float("nan")
        return c, s, int(row[2])

    print("=== 1. Headline persistence (current resid ~ trailing-10 mean) ===")
    c_all, s_all, n_all = persistence("game_no > 10")
    print(f"  all veteran games : corr={c_all:+.3f}  slope={s_all:+.3f}  n={n_all}")

    print("\n=== 2. By cumulative-experience bucket ===")
    buckets = [
        ("rookie  (game ≤50)", "game_no <= 50"),
        ("mid    (50–200)", "game_no > 50 AND game_no <= 200"),
        ("veteran (>200)", "game_no > 200"),
    ]
    bucket_slopes = {}
    for label, cond in buckets:
        c, s, n = persistence(cond)
        bucket_slopes[label] = s
        print(f"  {label:20} corr={c:+.3f}  slope={s:+.3f}  n={n}")

    print("\n=== 3. Carry games (teammates weaker than player) ===")
    c_carry, s_carry, n_carry = persistence("is_carry AND game_no > 10")
    c_pass, s_pass, n_pass = persistence("NOT is_carry AND game_no > 10")
    print(f"  carry (θ matters most): corr={c_carry:+.3f}  slope={s_carry:+.3f}  n={n_carry}")
    print(f"  passenger             : corr={c_pass:+.3f}  slope={s_pass:+.3f}  n={n_pass}")

    # ------------------------------------------------------------------
    # 4. Per-player prior-vs-recent block split.
    # ------------------------------------------------------------------
    w = int(args.window)
    block = con.execute(f"""
        WITH tagged AS (
            SELECT
                player_id, resid_norm, total_games,
                ROW_NUMBER() OVER (
                    PARTITION BY player_id ORDER BY start_date DESC, tournament_id DESC
                ) AS rn_desc
            FROM seq
        )
        SELECT
            player_id,
            AVG(resid_norm) FILTER (WHERE rn_desc <= {w})                 AS recent_mean,
            AVG(resid_norm) FILTER (WHERE rn_desc > {w} AND rn_desc <= {2*w}) AS prior_mean,
            MAX(total_games) AS total_games
        FROM tagged
        GROUP BY player_id
        HAVING COUNT(*) >= {2*w}
    """).fetchdf()

    valid = block.dropna(subset=["recent_mean", "prior_mean"])
    block_corr = valid["recent_mean"].corr(valid["prior_mean"])
    # OLS slope recent ~ prior
    import numpy as np
    x = valid["prior_mean"].to_numpy()
    y = valid["recent_mean"].to_numpy()
    block_slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else float("nan")
    print(f"\n=== 4. Prior-vs-recent block split (window={w}, {len(valid)} players) ===")
    print(f"  corr(prior_mean, recent_mean)={block_corr:+.3f}  slope={block_slope:+.3f}")
    print("  (high ⇒ a player's residual sign persists across ~2 years ⇒ θ stale)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    block.to_csv(out_path, index=False)
    print(f"  per-player block means → {out_path}")

    # ------------------------------------------------------------------
    # 5. Case studies.
    # ------------------------------------------------------------------
    print(f"\n=== 5. Case studies (prior {w} → recent {w} normalised residual) ===")
    for pid, name in CASE_STUDY.items():
        r = block[block["player_id"] == pid]
        if len(r) == 0:
            print(f"  {name}: not in veteran pool")
            continue
        r = r.iloc[0]
        carry = con.execute("""
            SELECT
                AVG(resid_norm) FILTER (WHERE is_carry),
                COUNT(*) FILTER (WHERE is_carry),
                AVG(resid_norm) FILTER (WHERE NOT is_carry),
                COUNT(*) FILTER (WHERE NOT is_carry)
            FROM seq WHERE player_id = ?
        """, [pid]).fetchone()
        cm = f"{carry[0]:+.4f}" if carry[0] is not None else "  N/A "
        pm = f"{carry[2]:+.4f}" if carry[2] is not None else "  N/A "
        print(
            f"  {name:12} prior={r['prior_mean']:+.4f}  recent={r['recent_mean']:+.4f}  "
            f"| carry_resid={cm} (n={carry[1]})  passenger={pm} (n={carry[3]})"
        )

    # ------------------------------------------------------------------
    # Decision.
    # ------------------------------------------------------------------
    print("\n=== DECISION ===")
    rises = (
        bucket_slopes.get("veteran (>200)", 0)
        > bucket_slopes.get("rookie  (game ≤50)", 0)
    )
    vet_slope = bucket_slopes.get("veteran (>200)", float("nan"))
    if vet_slope >= 0.15 and s_carry >= 0.10 and rises:
        verdict = "GO — strong persistence; adaptive η has real signal to exploit"
    elif vet_slope >= 0.05 or s_carry >= 0.05:
        verdict = "WEAK — some signal; expect a small, targeted gain only"
    else:
        verdict = "NO-GO — residuals ~memoryless; adaptive η would chase noise"
    print(f"  veteran slope={vet_slope:+.3f}  carry slope={s_carry:+.3f}  "
          f"rises_with_experience={rises}")
    print(f"  → {verdict}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
