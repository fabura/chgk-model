"""Compare player ranks: 1D baseline vs Model C (per-player difficulty
slope γ_k), and define a 1D projection of the 2D (θ, γ) player so players
stay comparable.

Both models are trained on the FULL dataset (holdout=0) with the same
two-pass schedule, so the only difference is the γ dimension — making the
rank comparison apples-to-apples.

1D projection of the 2D player:

    score(b*) = θ_k + γ_k · b*

This is exactly the player's ability contribution to the noisy-OR logit
at a question of difficulty ``b*`` (since z_k = θ_k + γ_k·b − b − δ).
We rank by ``b* = mean question difficulty`` by default, and also report
``b*=0`` (pure baseline θ) and a hard-pack reference (p90 of b).

Permutation metrics (1D rank order vs 2D-projection rank order):
  - Kendall τ + implied discordant-pair count (= number of swaps)
  - Spearman ρ
  - #players whose rank changed, mean / max |Δrank|
reported globally and at a games≥50 cut (to suppress 1-game churn).

Outputs:
  results/exp_modelc_rank_shift.csv          (per-player, focus set)
  results/exp_modelc_rank_shift_summary.csv  (permutation metrics)

Usage::

    python scripts/exp_modelc_rank_shift.py --cache_file data.npz --me 32919

Cost: 2 full-data runs ≈ 25 min.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import load_cached
from rating.engine import Config, run_sequential

try:
    from scipy.stats import kendalltau, spearmanr
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _ranks_desc(score: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    """Dense competition rank (1 = highest score) over masked indices."""
    idx = np.where(mask)[0]
    order = idx[np.argsort(-score[idx], kind="mergesort")]
    return {int(p): r + 1 for r, p in enumerate(order)}


def _perm_metrics(s1: np.ndarray, s2: np.ndarray, mask: np.ndarray, label: str) -> dict:
    idx = np.where(mask)[0]
    a, b = s1[idx], s2[idx]
    n = len(idx)
    r1 = (-a).argsort(kind="mergesort").argsort()
    r2 = (-b).argsort(kind="mergesort").argsort()
    dr = r2 - r1
    out = {
        "set": label,
        "n_players": n,
        "n_changed": int((dr != 0).sum()),
        "mean_abs_drank": round(float(np.abs(dr).mean()), 2),
        "max_abs_drank": int(np.abs(dr).max()) if n else 0,
    }
    if _HAVE_SCIPY and n > 2:
        tau, _ = kendalltau(a, b)
        rho, _ = spearmanr(a, b)
        n_pairs = n * (n - 1) / 2
        discordant = (1.0 - tau) / 2.0 * n_pairs
        out["kendall_tau"] = round(float(tau), 5)
        out["spearman_rho"] = round(float(rho), 5)
        out["discordant_pairs"] = int(round(discordant))
        out["total_pairs"] = int(n_pairs)
    return out


def _run(arrays, maps, *, two_d: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    cfg = Config(
        use_2d_players=two_d,
        eta_gamma=0.01 if two_d else 0.001,
        freeze_log_a=True,
        holdout_obs_fraction=0.0,
    )
    t0 = time.time()
    res = run_sequential(arrays, maps, cfg, verbose=False, collect_predictions=False)
    elapsed = time.time() - t0
    theta = np.asarray(res.players.theta, dtype=np.float64).copy()
    gamma = (
        np.asarray(res.players.gamma, dtype=np.float64).copy()
        if two_d else np.zeros_like(theta)
    )
    pgames = np.asarray(res.players.games, dtype=np.int64).copy()
    qb = np.asarray(res.questions.b, dtype=np.float64)
    qinit = np.asarray(res.questions.initialized, dtype=bool)
    b_mean = float(qb[qinit].mean()) if qinit.any() else 0.0
    b_p90 = float(np.percentile(qb[qinit], 90)) if qinit.any() else 0.0
    print(f"  [{'Model C' if two_d else '1D'}] {elapsed:.0f}s  "
          f"b_mean={b_mean:+.3f} b_p90={b_p90:+.3f} "
          f"γ_std={gamma[res.players.seen].std():.3f}", flush=True)
    return theta, gamma, res.players.seen.copy(), pgames, b_mean, b_p90


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--me", type=int, default=32919)
    ap.add_argument("--min-games-together", type=int, default=5)
    ap.add_argument("--db", default="website/data/chgk.duckdb")
    ap.add_argument("--out", default="results/exp_modelc_rank_shift.csv")
    ap.add_argument("--out-summary", default="results/exp_modelc_rank_shift_summary.csv")
    args = ap.parse_args()

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)
    pid_to_idx = {int(p): i for i, p in enumerate(maps.idx_to_player_id)}

    print("[run] 1D baseline (full data)…", flush=True)
    th1, _, seen1, _, _, _ = _run(arrays, maps, two_d=False)
    print("[run] Model C (full data)…", flush=True)
    th2, gm2, seen2, n_games, b_mean, b_p90 = _run(arrays, maps, two_d=True)

    seen = seen1 & seen2

    # 1D projection of 2D player.
    score_1d = th1
    proj_mean = th2 + gm2 * b_mean
    proj_zero = th2 + gm2 * 0.0
    proj_hard = th2 + gm2 * b_p90

    # --- permutation metrics: 1D θ-order vs 2D mean-projection order ---
    summary = []
    summary.append(_perm_metrics(score_1d, proj_mean, seen, "all_seen"))
    summary.append(_perm_metrics(score_1d, proj_mean, seen & (n_games >= 50), "games>=50"))
    summary.append(_perm_metrics(score_1d, proj_mean, seen & (n_games >= 200), "games>=200"))
    for row in summary:
        print("  ", row, flush=True)

    # --- focus set: me + teammates with enough shared games ---
    import duckdb
    c = duckdb.connect(args.db, read_only=True)
    tm = c.execute(
        """
        SELECT pg2.player_id, count(*) gt
        FROM player_games pg1
        JOIN player_games pg2
          ON pg1.tournament_id=pg2.tournament_id AND pg1.team_id=pg2.team_id
        WHERE pg1.player_id=? AND pg2.player_id != ?
        GROUP BY pg2.player_id HAVING count(*) >= ?
        """,
        [args.me, args.me, args.min_games_together],
    ).fetchall()
    names = {
        int(pid): (ln or "", fn or "")
        for pid, ln, fn in c.execute(
            "SELECT player_id, last_name, first_name FROM players"
        ).fetchall()
    }
    focus_ids = [args.me] + [int(p) for p, _ in tm]
    gt_map = {int(p): int(g) for p, g in tm}
    gt_map[args.me] = 0

    rank_1d = _ranks_desc(score_1d, seen)
    rank_mean = _ranks_desc(proj_mean, seen)
    rank_zero = _ranks_desc(proj_zero, seen)
    rank_hard = _ranks_desc(proj_hard, seen)

    rows = []
    for pid in focus_ids:
        idx = pid_to_idx.get(pid)
        if idx is None or not seen[idx]:
            continue
        ln, fn = names.get(pid, ("?", "?"))
        r1 = rank_1d.get(idx)
        rm = rank_mean.get(idx)
        rows.append({
            "player_id": pid,
            "name": f"{ln} {fn}".strip(),
            "games_together": gt_map.get(pid, 0),
            "theta_1d": round(float(th1[idx]), 4),
            "theta_2d": round(float(th2[idx]), 4),
            "gamma": round(float(gm2[idx]), 4),
            "proj_mean": round(float(proj_mean[idx]), 4),
            "rank_1d": r1,
            "rank_proj_mean": rm,
            "drank": (r1 - rm) if (r1 and rm) else None,
            "rank_proj_zero": rank_zero.get(idx),
            "rank_proj_hard": rank_hard.get(idx),
        })

    rows.sort(key=lambda r: (r["rank_proj_mean"] or 1 << 30))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with Path(args.out_summary).open("w", newline="") as f:
        keys = sorted({k for r in summary for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)

    print(f"\n[wrote] {out} ({len(rows)} players)", flush=True)
    print(f"[wrote] {args.out_summary}", flush=True)
    print(f"\nb_mean={b_mean:+.3f}  b_p90(hard)={b_p90:+.3f}", flush=True)
    print(f"\n{'rk1D':>5} {'rk2D':>5} {'Δ':>6}  {'θ1D':>6} {'θ2D':>6} {'γ':>6}  {'gt':>4}  name")
    print("-" * 78)
    for r in rows:
        print(f"{r['rank_1d']:>5} {r['rank_proj_mean']:>5} {str(r['drank']):>6}  "
              f"{r['theta_1d']:>+6.2f} {r['theta_2d']:>+6.2f} {r['gamma']:>+6.2f}  "
              f"{r['games_together']:>4}  {r['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
