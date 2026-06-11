"""Compare online θ to stationary batch fits; flag largest discrepancies.

Runs two batch fits (θ-only, fixed b/a/δ/lapse/recal from ``results/seq.npz``):

* **all** — all observations in the cache (or pilot: see ``--pilot``).
* **recent** — observations in the last ``--recent-days`` calendar days.

Pilot mode (``--pilot``): recent window only, fewer L-BFGS iterations — fast
sanity check.  Full mode: all-history fit + recent fit.

Outputs ``results/diagnostic_batch_theta_outliers.csv``.

Usage::

    python scripts/diagnostic_batch_theta_outliers.py --pilot
    python scripts/diagnostic_batch_theta_outliers.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import load_cached
from rating.batch_theta import build_batch_context, fit_batch_theta
from rating.io import load_results_npz

CASE_STUDY = {34909: "Чернуха", 26818: "Рекшинская", 158668: "Монина", 32919: "Фаттахов"}


def _roster_concentration(maps, min_games: int) -> dict[int, float]:
    """top_mate_share per player_id via DuckDB if available, else empty."""
    duckdb_path = REPO_ROOT / "website/data/chgk.duckdb"
    if not duckdb_path.is_file():
        return {}
    try:
        import duckdb
    except ImportError:
        return {}
    con = duckdb.connect(str(duckdb_path), read_only=True)
    mg = int(min_games)
    rows = con.execute(f"""
        WITH pairs AS (
            SELECT a.player_id, b.player_id AS mate_id, COUNT(*) AS n_co
            FROM player_games a
            JOIN player_games b
              ON a.tournament_id=b.tournament_id AND a.team_id=b.team_id
             AND a.player_id<>b.player_id
            JOIN players pl ON pl.player_id=a.player_id AND pl.games>={mg}
            GROUP BY 1, 2
        ),
        agg AS (
            SELECT pr.player_id, MAX(pr.n_co) AS top_co
            FROM pairs pr
            GROUP BY 1
        )
        SELECT a.player_id, a.top_co * 1.0 / pl.games
        FROM agg a
        JOIN players pl ON pl.player_id = a.player_id
    """).fetchall()
    con.close()
    return {int(r[0]): float(r[1]) for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--results", default="results/seq.npz")
    ap.add_argument("--out", default="results/diagnostic_batch_theta_outliers.csv")
    ap.add_argument("--min_games", type=int, default=150)
    ap.add_argument("--recent-days", type=int, default=548)
    ap.add_argument("--reg_theta", type=float, default=0.01)
    ap.add_argument("--pilot", action="store_true", help="fast: recent obs + 12% subsample")
    ap.add_argument(
        "--obs-subsample",
        type=float,
        default=None,
        help="fraction of obs in mask (default 0.12 pilot, 1.0 full)",
    )
    ap.add_argument("--maxiter", type=int, default=None)
    args = ap.parse_args()

    maxiter = args.maxiter
    if maxiter is None:
        maxiter = 20 if args.pilot else 35
    obs_sub = args.obs_subsample
    if obs_sub is None:
        obs_sub = 0.12 if args.pilot else 1.0

    print(f"[load] {args.cache_file} + {args.results}", flush=True)
    arrays, maps = load_cached(args.cache_file)
    res = load_results_npz(args.results)

    top_share = _roster_concentration(maps, args.min_games)

    theta_online = np.zeros(maps.num_players, dtype=np.float64)
    games = np.zeros(maps.num_players, dtype=np.int64)
    pid_to_idx = {int(p): i for i, p in enumerate(maps.idx_to_player_id)}
    for pid, th, gm in zip(res.player_id, res.theta, res.games):
        idx = pid_to_idx.get(int(pid))
        if idx is not None:
            theta_online[idx] = float(th)
            games[idx] = int(gm)

    veteran_mask = games >= args.min_games
    print(f"[pool] {int(veteran_mask.sum())} veterans (>={args.min_games} games)", flush=True)

    theta_batch_all: np.ndarray | None = None
    theta_batch_recent: np.ndarray | None = None

    if args.pilot:
        print("\n=== PILOT: batch fit (recent window only) ===", flush=True)
        t0 = time.time()
        ctx = build_batch_context(
            arrays, maps, res,
            min_games=args.min_games,
            recent_days=args.recent_days,
            obs_subsample=obs_sub,
        )
        theta_batch_recent = fit_batch_theta(
            ctx, reg_theta=args.reg_theta, maxiter=maxiter, verbose=True,
        )
        theta_batch_all = theta_batch_recent  # same window in pilot
        print(f"  elapsed {time.time()-t0:.0f}s", flush=True)
    else:
        print("\n=== FULL: batch fit ALL history ===", flush=True)
        t0 = time.time()
        ctx_all = build_batch_context(
            arrays, maps, res,
            min_games=args.min_games,
            recent_days=None,
            obs_subsample=obs_sub if obs_sub < 1.0 else None,
        )
        print(f"  obs in fit: {int(ctx_all.obs_mask.sum())}", flush=True)
        theta_batch_all = fit_batch_theta(
            ctx_all, reg_theta=args.reg_theta, maxiter=maxiter, verbose=True,
        )
        print(f"  elapsed {time.time()-t0:.0f}s", flush=True)

        print("\n=== FULL: batch fit RECENT window ===", flush=True)
        t0 = time.time()
        ctx_rec = build_batch_context(
            arrays, maps, res,
            min_games=args.min_games,
            recent_days=args.recent_days,
            obs_subsample=obs_sub if obs_sub < 1.0 else None,
        )
        print(f"  obs in fit: {int(ctx_rec.obs_mask.sum())}", flush=True)
        theta_batch_recent = fit_batch_theta(
            ctx_rec, reg_theta=args.reg_theta, maxiter=maxiter, verbose=True,
        )
        print(f"  elapsed {time.time()-t0:.0f}s", flush=True)

    assert theta_batch_all is not None and theta_batch_recent is not None

    rows = []
    for pidx in np.where(veteran_mask)[0]:
        pid = int(maps.idx_to_player_id[pidx])
        th_o = float(theta_online[pidx])
        th_a = float(theta_batch_all[pidx])
        th_r = float(theta_batch_recent[pidx])
        rows.append({
            "player_id": pid,
            "theta_online": round(th_o, 4),
            "theta_batch_all": round(th_a, 4),
            "theta_batch_recent": round(th_r, 4),
            "delta_all": round(th_a - th_o, 4),
            "delta_recent": round(th_r - th_o, 4),
            "delta_recent_vs_all": round(th_r - th_a, 4),
            "games": int(games[pidx]),
            "top_mate_share": round(top_share.get(pid, float("nan")), 4),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Diagnosis column: estimation error vs non-stationarity
    df["diag"] = np.where(
        np.abs(df["delta_recent"]) < 0.05,
        "online≈batch_recent",
        np.where(
            np.abs(df["delta_all"] - df["delta_recent"]) < 0.05,
            "likely_online_artifact",
            "likely_form_change",
        ),
    )

    print(f"\n[ok] {len(df)} rows → {out_path}", flush=True)

    print("\n=== Top 15: batch_all − online (positive = batch higher) ===")
    for _, r in df.nlargest(15, "delta_all").iterrows():
        print(
            f"  pid={int(r.player_id):6}  Δ_all={r['delta_all']:+.3f}  "
            f"Δ_rec={r['delta_recent']:+.3f}  share={r['top_mate_share']:.2f}  "
            f"{r['diag']}"
        )

    print("\n=== Top 15: batch_all − online (negative) ===")
    for _, r in df.nsmallest(15, "delta_all").iterrows():
        print(
            f"  pid={int(r.player_id):6}  Δ_all={r['delta_all']:+.3f}  "
            f"Δ_rec={r['delta_recent']:+.3f}  share={r['top_mate_share']:.2f}  "
            f"{r['diag']}"
        )

    print("\n=== Case studies ===")
    for pid, name in CASE_STUDY.items():
        sub = df[df["player_id"] == pid]
        if len(sub) == 0:
            print(f"  {name} (id={pid}): not in veteran pool")
            continue
        r = sub.iloc[0]
        print(
            f"  {name:12} online={r['theta_online']:+.3f}  "
            f"batch_all={r['theta_batch_all']:+.3f}  "
            f"batch_recent={r['theta_batch_recent']:+.3f}  "
            f"Δ_all={r['delta_all']:+.3f} Δ_rec={r['delta_recent']:+.3f}  "
            f"share={r['top_mate_share']:.2f}  → {r['diag']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
