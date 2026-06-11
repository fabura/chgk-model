"""Offline test of the "up-weight roster-variation games" idea (class C #7).

A stable-roster player's θ is under-identified: if you always play with
the same cast, your θ and theirs are confounded by team-level data.  The
proposed fix up-weights games played in an *unusual* cast (the natural
experiment that separates teammates).  This script tests the precondition
**without touching the engine**:

A. **Concentration** — how locked is each veteran's roster?
   Metrics per player: top-teammate share (games with the single most
   frequent partner / games), distinct partners per game, Herfindahl of
   the partner distribution.  Places the three flagged players in the
   population distribution.

B. **Does variation carry a different signal?** — for each player split
   games into "familiar cast" (teammates are the player's frequent
   partners) vs "unfamiliar cast", and compare the per-game implied θ
   (``team_theta_implied``) and normalised residual.  The idea only helps
   if, for locked players, unfamiliar-cast games imply a *systematically
   different* θ than the familiar-cast games the model mostly trained on.

Decision rule:
    GO   if concentration correlates with |implied gap| (locked ⇒ worse
         identified) AND, for high-concentration players, unfamiliar-cast
         implied θ differs from familiar-cast by ≥ 0.10 on average.
    NO-GO if the familiar/unfamiliar implied θ agree (no correction to
         make) or the flagged players are not actually roster-locked.

Usage::

    python scripts/diagnostic_roster_identifiability.py \\
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    ap.add_argument("--out", default="results/diagnostic_roster_identifiability.csv")
    ap.add_argument("--min_games", type=int, default=150)
    args = ap.parse_args()

    try:
        import duckdb
    except ImportError:
        raise SystemExit("pip install duckdb")

    con = duckdb.connect(args.duckdb, read_only=True)
    mg = int(args.min_games)

    # ------------------------------------------------------------------
    # Co-appearance counts: (veteran player, mate) → games together.
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pairs AS
        SELECT a.player_id, b.player_id AS mate_id, COUNT(*) AS n_co
        FROM player_games a
        JOIN player_games b
          ON a.tournament_id = b.tournament_id
         AND a.team_id = b.team_id
         AND a.player_id <> b.player_id
        JOIN players pa ON pa.player_id = a.player_id AND pa.games >= {mg}
        GROUP BY 1, 2
    """)

    # ------------------------------------------------------------------
    # A. Per-player concentration metrics.
    # ------------------------------------------------------------------
    conc = con.execute(f"""
        SELECT
            pr.player_id,
            pl.last_name, pl.first_name, pl.theta, pl.games,
            SUM(pr.n_co)                                   AS total_co,
            MAX(pr.n_co)                                   AS top_mate_co,
            COUNT(*)                                       AS distinct_mates,
            SUM(pr.n_co * pr.n_co) * 1.0
                / (SUM(pr.n_co) * SUM(pr.n_co))            AS herfindahl,
            MAX(pr.n_co) * 1.0 / pl.games                  AS top_mate_share,
            COUNT(*) * 1.0 / pl.games                      AS mates_per_game
        FROM pairs pr
        JOIN players pl ON pl.player_id = pr.player_id
        GROUP BY 1, 2, 3, 4, 5
    """).fetchdf()

    print(f"=== A. Roster concentration ({len(conc)} veterans ≥{mg} games) ===")
    for col, lbl in [
        ("top_mate_share", "top-teammate share"),
        ("mates_per_game", "distinct partners / game"),
        ("herfindahl", "partner Herfindahl"),
    ]:
        qs = conc[col].quantile([0.5, 0.75, 0.9, 0.95])
        print(f"  {lbl:26} p50={qs[0.5]:.3f} p75={qs[0.75]:.3f} "
              f"p90={qs[0.9]:.3f} p95={qs[0.95]:.3f}")

    print("\n  Case players (higher top-share / Herfindahl = more locked):")
    for pid, name in CASE_STUDY.items():
        r = conc[conc["player_id"] == pid]
        if len(r) == 0:
            print(f"    {name}: not in pool")
            continue
        r = r.iloc[0]
        pct_share = float((conc["top_mate_share"] < r["top_mate_share"]).mean())
        pct_herf = float((conc["herfindahl"] < r["herfindahl"]).mean())
        print(f"    {name:12} top_share={r['top_mate_share']:.3f} "
              f"(pctl {pct_share:.0%})  herf={r['herfindahl']:.3f} "
              f"(pctl {pct_herf:.0%})  distinct_mates={int(r['distinct_mates'])}")

    # ------------------------------------------------------------------
    # Per-game familiarity: average co-appearance count of the game's
    # teammates with the player (high = the player's usual cast).
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE game_fam AS
        SELECT a.player_id, a.tournament_id, a.team_id,
               AVG(pr.n_co) AS fam, COUNT(*) AS n_mates
        FROM player_games a
        JOIN player_games b
          ON a.tournament_id = b.tournament_id
         AND a.team_id = b.team_id
         AND a.player_id <> b.player_id
        JOIN pairs pr
          ON pr.player_id = a.player_id AND pr.mate_id = b.player_id
        JOIN players pa ON pa.player_id = a.player_id AND pa.games >= {mg}
        GROUP BY 1, 2, 3
    """)

    # Join familiarity to per-game implied θ + residual; classify each
    # game as familiar/unfamiliar by the player's own median familiarity.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE game_signal AS
        SELECT
            gf.player_id,
            gf.fam,
            tg.team_theta_implied,
            (tg.score_actual - tg.expected_takes)
                / GREATEST(t.n_questions, 1) AS resid_norm,
            MEDIAN(gf.fam) OVER (PARTITION BY gf.player_id) AS med_fam
        FROM game_fam gf
        JOIN team_games tg
          ON tg.tournament_id = gf.tournament_id AND tg.team_id = gf.team_id
        JOIN tournaments t ON t.tournament_id = gf.tournament_id
        WHERE tg.team_theta_implied IS NOT NULL
          AND tg.expected_takes IS NOT NULL
    """)

    split = con.execute("""
        SELECT
            player_id,
            AVG(team_theta_implied) FILTER (WHERE fam <= med_fam)  AS impl_varied,
            AVG(team_theta_implied) FILTER (WHERE fam >  med_fam)  AS impl_stable,
            AVG(resid_norm)        FILTER (WHERE fam <= med_fam)   AS resid_varied,
            AVG(resid_norm)        FILTER (WHERE fam >  med_fam)   AS resid_stable,
            COUNT(*)                                               AS n
        FROM game_signal
        GROUP BY 1
        HAVING COUNT(*) >= 30
    """).fetchdf()
    split["impl_delta"] = split["impl_varied"] - split["impl_stable"]

    merged = conc.merge(split, on="player_id", how="inner")
    merged["implied_gap"] = merged["impl_stable"]  # proxy reference

    print(f"\n=== B. Familiar vs unfamiliar cast ({len(split)} players) ===")
    print(f"  mean impl_delta (unfamiliar − familiar) = "
          f"{split['impl_delta'].mean():+.4f}  "
          f"(≈0 ⇒ variation implies the same θ ⇒ nothing to correct)")
    print(f"  std  impl_delta = {split['impl_delta'].std():.4f}")

    # Correlation: do more locked players show a larger delta (i.e. their
    # stable games hold θ away from what varied games imply)?
    corr_lock_delta = merged["herfindahl"].corr(merged["impl_delta"])
    corr_lock_absdelta = merged["herfindahl"].corr(merged["impl_delta"].abs())
    print(f"  corr(Herfindahl, impl_delta)      = {corr_lock_delta:+.3f}")
    print(f"  corr(Herfindahl, |impl_delta|)    = {corr_lock_absdelta:+.3f}")

    # Among the most-locked quartile, is the delta systematically positive?
    q75 = merged["herfindahl"].quantile(0.75)
    locked = merged[merged["herfindahl"] >= q75]
    print(f"  most-locked quartile (herf≥{q75:.3f}, n={len(locked)}): "
          f"mean impl_delta={locked['impl_delta'].mean():+.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"  per-player table → {out_path}")

    print("\n  Case players (impl θ from unfamiliar vs familiar casts):")
    for pid, name in CASE_STUDY.items():
        r = split[split["player_id"] == pid]
        if len(r) == 0:
            print(f"    {name}: <30 games")
            continue
        r = r.iloc[0]
        print(f"    {name:12} impl_varied={r['impl_varied']:+.3f}  "
              f"impl_stable={r['impl_stable']:+.3f}  "
              f"Δ={r['impl_delta']:+.3f}  "
              f"(resid varied={r['resid_varied']:+.4f} stable={r['resid_stable']:+.4f})")

    # ------------------------------------------------------------------
    # Decision.
    # ------------------------------------------------------------------
    print("\n=== DECISION ===")
    locked_delta = float(locked["impl_delta"].mean())
    # The premise of idea #7 is that *more locked* players carry a larger
    # familiar-vs-unfamiliar discrepancy (the correction the up-weighting
    # would surface) — i.e. corr(lock, |Δ|) must be POSITIVE.  A negative
    # correlation refutes it: locked players show *less* discrepancy, and
    # the population Δ is then driven by the team-strength confound
    # (unfamiliar casts = weaker ad-hoc teams ⇒ lower implied θ), not by
    # identifiability.
    go = (corr_lock_absdelta >= 0.15) and (abs(locked_delta) >= 0.10)
    if go:
        verdict = ("GO — locked players' unfamiliar-cast games imply a "
                   "different θ; up-weighting them has signal")
    elif corr_lock_absdelta <= -0.05:
        verdict = ("NO-GO — locked players show *smaller* discrepancy "
                   "(corr<0); the population Δ is the team-strength "
                   "confound, not an identifiability signal")
    elif corr_lock_absdelta >= 0.05:
        verdict = ("WEAK — weak positive link between lock and discrepancy; "
                   "expect a narrow gain at best")
    else:
        verdict = ("NO-GO — familiar/unfamiliar casts imply the same θ; "
                   "roster-variation up-weighting has nothing to correct")
    print(f"  corr(lock, |Δ|)={corr_lock_absdelta:+.3f}  "
          f"locked-quartile mean Δ={locked_delta:+.4f}")
    print(f"  → {verdict}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
