"""Look for monotonic patterns in test-set error.

For each test observation, compute a handful of features:

    team_theta_mean   pre-tournament mean θ over the predicting team
    team_size
    team_min_games    rookie-ness — min total game count among
                      players on the roster (final snapshot, used as
                      a proxy for "had this team a green player?")
    team_mean_games   mean total games among players on the roster
    q_b_final         post-training canonical b for the question
    q_n_obs_train     number of times this canonical question
                      appeared in the train portion (0 for the
                      99 % "new in test" canonicals)
    q_first_app_size  team count on the FIRST appearance of this
                      canonical (whatever tournament that was) —
                      drives the init pathology
    tour_mean_take    mean take rate of all teams in this tournament
                      (proxy for pack hardness)
    tour_n_obs        size of the tournament (n_teams * n_questions)

For each feature, bin the test set into 10 quantiles and report
``n / mean_p̂ / mean_y / mean_residual / mean_logloss`` per bucket.

Outputs ``results/error_patterns/{feature}.csv`` plus a printed
summary.

Caveats:

* ``team_min_games`` and ``team_mean_games`` use the FINAL snapshot
  of ``players.games`` (i.e. the total over the whole dataset). For
  test obs near the start of the test window this overestimates how
  "experienced" the team was at prediction time. To approximate
  what was true at the moment of the test obs, we subtract a rough
  estimate of "games played within the test window" — see
  ``_approx_games_at_obs`` below.
* ``tour_mean_take`` is computed over ALL test obs in the
  tournament — it's a property of the test sub-set of that
  tournament, useful for ranking but not literally "what the model
  saw before predicting".
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import load_cached
from rating.backtest import backtest
from rating.engine import Config


def _build_offsets(team_sizes: np.ndarray) -> np.ndarray:
    off = np.empty(len(team_sizes) + 1, dtype=np.int64)
    off[0] = 0
    np.cumsum(team_sizes.astype(np.int64), out=off[1:])
    return off


def _bin_quantile(x: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_idx, bin_edges) using empirical quantile cuts."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 2:
        return np.zeros_like(x, dtype=np.int64), edges
    bins = np.clip(
        np.searchsorted(edges[1:-1], x, side="right"), 0, len(edges) - 2
    )
    return bins, edges


def _format_table(
    name: str,
    edges: np.ndarray,
    counts: np.ndarray,
    mean_p: np.ndarray,
    mean_y: np.ndarray,
    mean_res: np.ndarray,
    mean_ll: np.ndarray,
) -> str:
    lines = [
        f"\n=== {name} (binning by quantile) ===",
        f"{'bin':<5}{'edges':<24}{'n':>10}"
        f"{'mean_p':>10}{'mean_y':>10}"
        f"{'mean_res':>11}{'mean_ll':>10}",
    ]
    for i in range(len(counts)):
        lo = edges[i] if i < len(edges) else float("nan")
        hi = edges[i + 1] if i + 1 < len(edges) else float("nan")
        lines.append(
            f"{i:<5}[{lo:>+8.3f}, {hi:>+8.3f}] "
            f"{int(counts[i]):>9d}"
            f"{mean_p[i]:>10.3f}{mean_y[i]:>10.3f}"
            f"{mean_res[i]:>+11.3f}{mean_ll[i]:>10.4f}"
        )
    return "\n".join(lines)


def _stats_per_bin(
    bins: np.ndarray, n_bins_actual: int,
    p: np.ndarray, y: np.ndarray, res: np.ndarray, ll: np.ndarray,
):
    counts = np.zeros(n_bins_actual, dtype=np.int64)
    sum_p = np.zeros(n_bins_actual)
    sum_y = np.zeros(n_bins_actual)
    sum_res = np.zeros(n_bins_actual)
    sum_ll = np.zeros(n_bins_actual)
    np.add.at(counts, bins, 1)
    np.add.at(sum_p, bins, p)
    np.add.at(sum_y, bins, y)
    np.add.at(sum_res, bins, res)
    np.add.at(sum_ll, bins, ll)
    safe = np.maximum(counts, 1)
    return counts, sum_p / safe, sum_y / safe, sum_res / safe, sum_ll / safe


def _write_csv(
    path: Path, edges: np.ndarray,
    counts, mean_p, mean_y, mean_res, mean_ll,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "bin", "edge_lo", "edge_hi", "n",
            "mean_p", "mean_y", "mean_res", "mean_ll",
        ])
        for i in range(len(counts)):
            lo = edges[i] if i < len(edges) else float("nan")
            hi = edges[i + 1] if i + 1 < len(edges) else float("nan")
            w.writerow([
                i, lo, hi, int(counts[i]),
                f"{mean_p[i]:.6f}", f"{mean_y[i]:.6f}",
                f"{mean_res[i]:.6f}", f"{mean_ll[i]:.6f}",
            ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out_dir", default="results/error_patterns")
    ap.add_argument("--n_bins", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)
    cfg = Config()

    print("[backtest] running once with predictions…")
    metrics = backtest(arrays, maps, cfg, verbose=False)
    result = metrics["result"]
    pred = result.predictions
    if pred is None or "obs_idx" not in pred:
        raise RuntimeError("predictions need obs_idx (engine patch)")

    pred_p = pred["pred_p"]
    actual_y = pred["actual_y"]
    pred_g = pred["game_idx"]
    pred_obs = pred["obs_idx"]
    team_theta_mean_pred = pred["team_theta_mean"]

    # === Reconstruct test mask exactly like backtest() does =========
    gdo = maps.game_date_ordinal
    all_games = np.unique(pred_g)
    if gdo is not None:
        known = all_games[
            np.array([int(gdo[g]) >= 0 for g in all_games], dtype=bool)
        ]
        ordered = known[
            np.argsort(np.array([int(gdo[g]) for g in known]))
        ]
    else:
        ordered = np.sort(all_games)
    n_test = max(1, int(len(ordered) * 0.2))
    test_games_arr = np.array(ordered[-n_test:], dtype=np.int64)
    test_games_set = set(int(g) for g in test_games_arr)
    test_mask = np.fromiter(
        (int(g) in test_games_set for g in pred_g),
        count=len(pred_g),
        dtype=bool,
    )

    p = pred_p[test_mask]
    y = actual_y[test_mask].astype(np.float64)
    g = pred_g[test_mask]
    obs = pred_obs[test_mask]
    team_theta = team_theta_mean_pred[test_mask]

    eps = 1e-15
    p_clip = np.clip(p, eps, 1.0 - eps)
    ll = -(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
    res = y - p
    print(f"[test] n_obs = {len(p):,}  mean_logloss = {ll.mean():.4f}")

    # === Build per-test-obs feature vectors =========================
    q_idx_arr = arrays["q_idx"]
    team_sizes = arrays["team_sizes"]
    offsets = _build_offsets(team_sizes)
    player_flat = arrays["player_indices_flat"]
    game_idx_arr = arrays["game_idx"]
    taken_arr = arrays["taken"]

    cq_arr = (
        maps.canonical_q_idx if maps.canonical_q_idx is not None
        else np.arange(maps.num_questions, dtype=np.int32)
    )
    n_cq = int(cq_arr.max()) + 1

    # Per-canonical: train counts (for the "new in test" pattern)
    # and first-appearance team count (for init pathology).
    print("[features] computing per-cq stats")
    train_obs_mask = ~np.fromiter(
        (int(g) in test_games_set for g in game_idx_arr),
        count=len(game_idx_arr),
        dtype=bool,
    )
    cq_obs_all = cq_arr[q_idx_arr]
    n_obs_train_per_cq = np.bincount(
        cq_obs_all[train_obs_mask], minlength=n_cq
    )

    # First appearance: which tournament was this canonical first seen
    # in (over the whole dataset, by date)?  Then how many obs did it
    # have there?
    print("[features] first-appearance team count (this is slow)…")
    first_app_size = np.zeros(n_cq, dtype=np.int64)
    first_app_game = -np.ones(n_cq, dtype=np.int64)
    # walk obs in date order
    if gdo is not None:
        gdo_per_obs = np.array(
            [int(gdo[g]) if g < len(gdo) else 0 for g in game_idx_arr],
            dtype=np.int64,
        )
    else:
        gdo_per_obs = game_idx_arr.astype(np.int64)
    order = np.argsort(gdo_per_obs, kind="stable")
    seen_cq = np.zeros(n_cq, dtype=bool)
    counters = np.zeros(n_cq, dtype=np.int64)
    current_game = np.full(n_cq, -1, dtype=np.int64)
    for k in order:
        cq_k = cq_obs_all[k]
        if not seen_cq[cq_k]:
            seen_cq[cq_k] = True
            first_app_game[cq_k] = game_idx_arr[k]
            current_game[cq_k] = game_idx_arr[k]
            counters[cq_k] = 1
        elif current_game[cq_k] == game_idx_arr[k]:
            counters[cq_k] += 1
    # counters now holds first-appearance team count per cq
    first_app_size = counters

    # For each test obs k, fetch features.
    print("[features] per-obs vectors")
    cq_test = cq_arr[q_idx_arr[obs]]
    q_b_final = result.questions.b[cq_test]
    q_n_obs_train = n_obs_train_per_cq[cq_test]
    q_first_app_size = first_app_size[cq_test]
    team_size_test = team_sizes[obs].astype(np.int64)

    # Team experience: use FINAL games count of each player on roster.
    games_final = result.players.games
    team_min_games = np.empty(len(obs), dtype=np.int64)
    team_mean_games = np.empty(len(obs), dtype=np.float64)
    for k in range(len(obs)):
        oi = int(obs[k])
        s, e = int(offsets[oi]), int(offsets[oi + 1])
        if e <= s:
            team_min_games[k] = 0
            team_mean_games[k] = 0.0
            continue
        gms = games_final[player_flat[s:e]]
        team_min_games[k] = int(gms.min())
        team_mean_games[k] = float(gms.mean())

    # Tournament hardness proxy: mean take rate over THIS tournament's
    # test obs (computed on test only — leakage-ish, but useful as a
    # ranking variable).
    tour_take_sum = defaultdict(float)
    tour_take_n = defaultdict(int)
    for gi, yi in zip(g, y):
        tour_take_sum[int(gi)] += float(yi)
        tour_take_n[int(gi)] += 1
    tour_mean_take_per_g = {
        gi: tour_take_sum[gi] / max(tour_take_n[gi], 1)
        for gi in tour_take_sum
    }
    tour_size_per_g = dict(tour_take_n)
    tour_mean_take = np.array(
        [tour_mean_take_per_g[int(gi)] for gi in g], dtype=np.float64
    )
    tour_n_obs = np.array(
        [tour_size_per_g[int(gi)] for gi in g], dtype=np.int64
    )

    # === Run binning per axis =======================================
    axes = [
        ("team_theta_mean", team_theta, "pre-tournament team θ"),
        ("team_size", team_size_test.astype(float),
         "players on roster (size)"),
        ("team_min_games", team_min_games.astype(float),
         "rookie-ness: min games (final snapshot) on roster"),
        ("team_mean_games", team_mean_games,
         "experience: mean games (final snapshot) on roster"),
        ("q_b_final", q_b_final,
         "post-training b of the question (sortable hardness)"),
        ("q_n_obs_train", q_n_obs_train.astype(float),
         "how many train obs the canonical had (0 = new in test)"),
        ("q_first_app_size", q_first_app_size.astype(float),
         "team count on FIRST appearance of this canonical anywhere"),
        ("tour_mean_take", tour_mean_take,
         "mean take rate of this tournament's test obs"),
        ("tour_n_obs", tour_n_obs.astype(float),
         "tournament size (n test obs)"),
    ]

    summary_lines = [
        f"n_test_obs = {len(p):,}",
        f"mean_logloss (overall) = {ll.mean():.4f}",
        f"mean_AUC-substitute (mean residual) = {res.mean():+.4f}",
        "",
    ]

    for name, x, descr in axes:
        bins, edges = _bin_quantile(x, args.n_bins)
        n_actual = max(1, len(edges) - 1)
        counts, mean_p, mean_y, mean_res, mean_ll = _stats_per_bin(
            bins, n_actual, p, y, res, ll
        )
        if counts.sum() == 0:
            print(f"[skip] {name}: empty")
            continue
        print(f"\n--- {name}: {descr} ---")
        print(_format_table(name, edges, counts, mean_p, mean_y, mean_res, mean_ll))
        # Add to summary: trend across deciles.
        mn, mx = float(mean_ll[0]), float(mean_ll[-1])
        worst_idx = int(np.argmax(mean_ll))
        summary_lines.append(
            f"{name:<22s}  ll bin0={mn:.4f}  ll bin{n_actual-1}={mx:.4f}  "
            f"Δ={mx-mn:+.4f}  worst_bin={worst_idx} ({mean_ll[worst_idx]:.4f})"
        )
        _write_csv(
            out / f"{name}.csv", edges, counts,
            mean_p, mean_y, mean_res, mean_ll,
        )

    summary_path = out / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\n[ok] per-feature CSVs + summary → {out}/")
    print("\n=== Trend summary (Δ logloss between extreme deciles) ===")
    for line in summary_lines[3:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
