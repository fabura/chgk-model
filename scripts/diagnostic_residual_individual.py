"""Separate the *individual* part of the residual from team context.

Follow-up to ``diagnostic_residual_persistence.py``, which found strong
residual persistence (slope ≈ 0.54) but with carry ≈ passenger, hinting
the persistence is team-context, not individual θ-staleness.  This script
settles it two ways:

A. **Cross-team consistency.**  For each player split games by team
   (cells with ≥``min_cell`` games).  If a player is genuinely under/over-
   rated, they over/under-perform on *all* their teams, so their team-mean
   residuals correlate across teams.  If the residual is a team property,
   the same player's residual differs by team → low cross-team correlation.
   We report:
     * corr of a player's two largest-team mean residuals (across players);
     * a one-way ICC (between-player variance fraction of cell means).

B. **Context-residualised persistence.**  Regress ``resid_norm`` on
   ``mate_avg_theta`` and ``roster_size`` (the cheap context proxies),
   take the residual ``resid_ind``, and recompute the trailing-10
   persistence slope.  If individual persistence survives context removal,
   there is a fixable individual signal; if it collapses, the persistence
   was context.

Decision rule:
    INDIVIDUAL  if cross-team corr ≥ 0.35 AND ICC ≥ 0.30
                AND residualised slope ≥ 0.15
    MIXED       if one of the two families of evidence is positive
    CONTEXT     otherwise (persistence is team-level; an individual η /
                attribution tweak won't move the flagged players)

Usage::

    python scripts/diagnostic_residual_individual.py \\
        --duckdb website/data/chgk.duckdb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CASE_STUDY = {34909: "Чернуха", 26818: "Рекшинская", 158668: "Монина"}


def _build_pg_resid(con, min_games: int) -> None:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE roster_theta AS
        SELECT pg.tournament_id, pg.team_id,
               SUM(p.theta) AS sum_theta, COUNT(*) AS cnt
        FROM player_games pg
        JOIN players p ON p.player_id = pg.player_id
        GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE pg_resid AS
        SELECT
            pg.player_id,
            pg.team_id,
            t.start_date,
            tg.team_name,
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
    """, [min_games])


def part_a_cross_team(con, min_cell: int) -> tuple[float, float, "object"]:
    """Cross-team consistency + one-way ICC of (player, team) cell means."""
    cells = con.execute(f"""
        SELECT player_id, team_id, team_name,
               AVG(resid_norm) AS cell_mean, COUNT(*) AS n
        FROM pg_resid
        GROUP BY 1, 2, 3
        HAVING COUNT(*) >= {int(min_cell)}
    """).fetchdf()

    # Keep players with >= 2 qualifying team cells.
    counts = cells.groupby("player_id").size()
    multi = counts[counts >= 2].index
    multi_cells = cells[cells["player_id"].isin(multi)].copy()

    # (a) correlation of the two largest-n team cell means per player.
    pairs = []
    for pid, grp in multi_cells.groupby("player_id"):
        top2 = grp.nlargest(2, "n")
        if len(top2) == 2:
            pairs.append((top2.iloc[0]["cell_mean"], top2.iloc[1]["cell_mean"]))
    pairs = np.array(pairs)
    cross_corr = (
        float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
        if len(pairs) > 2 else float("nan")
    )

    # (b) one-way ICC on cell means grouped by player (unweighted).
    grand = multi_cells["cell_mean"].mean()
    grp = multi_cells.groupby("player_id")["cell_mean"]
    pmeans = grp.mean()
    nper = grp.size()
    k = len(pmeans)
    N = len(multi_cells)
    ss_between = float((nper * (pmeans - grand) ** 2).sum())
    ss_within = float(
        ((multi_cells["cell_mean"] - multi_cells["player_id"].map(pmeans)) ** 2).sum()
    )
    msb = ss_between / (k - 1) if k > 1 else float("nan")
    msw = ss_within / (N - k) if N > k else float("nan")
    n0 = (N - (nper ** 2).sum() / N) / (k - 1) if k > 1 else float("nan")
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if msw and not np.isnan(msw) else float("nan")

    print(f"=== A. Cross-team consistency (cells ≥{min_cell} games) ===")
    print(f"  players with ≥2 team cells: {k}  (total cells: {N})")
    print(f"  corr(top-2 team residuals)  = {cross_corr:+.3f}   "
          f"(high ⇒ residual is an individual trait)")
    print(f"  one-way ICC (between frac)  = {icc:+.3f}   "
          f"(fraction of cell-mean variance that is between-player)")
    return cross_corr, icc, cells


def part_b_residualised(con, trail: int) -> tuple[float, float]:
    df = con.execute("""
        SELECT player_id, start_date, team_id, resid_norm,
               mate_avg_theta, roster_size
        FROM pg_resid
        WHERE mate_avg_theta IS NOT NULL
        ORDER BY player_id, start_date, team_id
    """).fetchdf()

    y = df["resid_norm"].to_numpy(dtype=np.float64)
    X = np.column_stack([
        np.ones(len(df)),
        df["mate_avg_theta"].to_numpy(dtype=np.float64),
        df["roster_size"].to_numpy(dtype=np.float64),
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_ind = y - X @ beta
    df["resid_ind"] = resid_ind

    # Trailing-10 mean of resid_ind per player (excluding current row).
    g = df.groupby("player_id", sort=False)["resid_ind"]
    trail_ind = g.transform(
        lambda s: s.shift(1).rolling(int(trail), min_periods=3).mean()
    )
    df["trail_ind"] = trail_ind

    # Trailing of the RAW residual too, for an apples-to-apples baseline on
    # the same (mate-known) row subset.
    g_raw = df.groupby("player_id", sort=False)["resid_norm"]
    df["trail_raw"] = g_raw.transform(
        lambda s: s.shift(1).rolling(int(trail), min_periods=3).mean()
    )

    def slope(xcol: str, ycol: str) -> float:
        sub = df[[xcol, ycol]].dropna()
        if len(sub) < 100:
            return float("nan")
        xs = sub[xcol].to_numpy()
        ys = sub[ycol].to_numpy()
        return float(np.polyfit(xs, ys, 1)[0])

    raw_slope = slope("trail_raw", "resid_norm")
    ind_slope = slope("trail_ind", "resid_ind")
    var_explained = 1.0 - float(np.var(resid_ind) / np.var(y))

    print(f"\n=== B. Context-residualised persistence (trail={trail}) ===")
    print(f"  context (mate θ + roster size) explains "
          f"{100*var_explained:.1f}% of residual variance")
    print(f"  raw trailing slope (this subset) = {raw_slope:+.3f}")
    print(f"  individual (residualised) slope  = {ind_slope:+.3f}   "
          f"(survives context removal?)")
    return ind_slope, raw_slope


def case_studies(con, cells, min_cell: int) -> None:
    print(f"\n=== Case studies: per-team mean residual (cells ≥{min_cell}) ===")
    for pid, name in CASE_STUDY.items():
        grp = cells[cells["player_id"] == pid].sort_values("n", ascending=False)
        if len(grp) == 0:
            print(f"  {name}: no qualifying team cells")
            continue
        parts = [
            f"{str(r['team_name'])[:18]}:{r['cell_mean']:+.4f}(n{int(r['n'])})"
            for _, r in grp.head(5).iterrows()
        ]
        spread = grp["cell_mean"].max() - grp["cell_mean"].min()
        print(f"  {name:12} spread={spread:.4f}  " + "  ".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    ap.add_argument("--min_games", type=int, default=150)
    ap.add_argument("--min_cell", type=int, default=15)
    ap.add_argument("--trail", type=int, default=10)
    args = ap.parse_args()

    try:
        import duckdb
    except ImportError:
        raise SystemExit("pip install duckdb")

    con = duckdb.connect(args.duckdb, read_only=True)
    _build_pg_resid(con, args.min_games)

    cross_corr, icc, cells = part_a_cross_team(con, args.min_cell)
    ind_slope, raw_slope = part_b_residualised(con, args.trail)
    case_studies(con, cells, args.min_cell)

    print("\n=== DECISION ===")
    a_pos = (cross_corr >= 0.35) and (icc >= 0.30)
    b_pos = ind_slope >= 0.15
    if a_pos and b_pos:
        verdict = ("INDIVIDUAL — residual is a stable per-player trait; a "
                   "targeted individual mechanism is justified")
    elif a_pos or b_pos:
        verdict = ("MIXED — partial individual signal; expect a small, "
                   "narrowly-targeted gain only")
    else:
        verdict = ("CONTEXT — persistence is team-level; individual η / "
                   "attribution tweaks won't move the flagged players")
    print(f"  cross_corr={cross_corr:+.3f}  ICC={icc:+.3f}  "
          f"residualised_slope={ind_slope:+.3f}")
    print(f"  → {verdict}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
