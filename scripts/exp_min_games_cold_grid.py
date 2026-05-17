"""3-D sweep over (min_games × cold_init_theta × games_offset).

Motivated by 2026-05 user reports of systematic over-prediction for
teams with newbies (bias up to −13.6 on `all_new` rosters, growing
~−1.5/+1 newbie).  Tests two hypotheses simultaneously:

1. `min_games` (in data.py loader) is dropping too many actually-played
   players from rosters, leaving training to see "1 опытный + N
   синтетических-нулей" instead of the real team.
2. `cold_init_theta = -1.0` makes newbies too strong: noisy-OR sums
   their individual p_take, so 5×rookie at θ=-1 gives a team
   p_take ≈ 0.7 on an average pack, which is wrong.

This sweep deliberately re-pulls separate caches per `min_games` value
(observations differ across them, so absolute logloss is NOT comparable
across `min_games` rows; bias-by-newbie IS).  The output table is
ranked first by **bias on the "≥3 newbies" bucket** (the actual pain
point), then by overall test-set logloss.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/exp_min_games_cold_grid.py \\
        --cache-mg0 data.npz.mg0 --cache-mg5 data.npz.mg5 \\
        --cache-mg10 data.npz.mg10 \\
        --out results/exp_min_games_cold_grid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential


GRID_COLD = [-2.0, -1.5, -1.0]
GRID_OFFSET = [0.25, 0.50]


def _newbie_buckets(n_new: np.ndarray) -> np.ndarray:
    """Map per-team newbie count to {'0','1','2','3+'} string labels."""
    out = np.empty(len(n_new), dtype=object)
    out[n_new == 0] = "0"
    out[n_new == 1] = "1"
    out[n_new == 2] = "2"
    out[n_new >= 3] = "3+"
    return out


def _team_bias_diagnostic(
    result,
    arrays: dict[str, np.ndarray],
    maps,
) -> dict[str, dict[str, float]]:
    """Compute team-level (score − expected) bias bucketed by #newbies.

    Only includes observations in the holdout set (so this measures
    real out-of-sample bias).  Newbie = player whose final `games`
    counter < 30 (matches the website convention).

    Returns dict[bucket] -> {n_teams, mean_bias, mean_score, mean_exp}.
    """
    preds = result.predictions or {}
    if not preds:
        return {}
    is_holdout = preds.get("is_holdout")
    if is_holdout is None:
        return {}
    is_holdout = is_holdout.astype(bool)
    if not is_holdout.any():
        return {}

    obs_idx = preds["obs_idx"].astype(np.int64)
    pred_p = preds["pred_p"].astype(np.float64)
    actual_y = preds["actual_y"].astype(np.int64)

    # Reconstruct offsets/player_flat the same way engine.py does
    # (from `team_sizes` + `player_indices_flat`).  The team identity
    # for an obs is implicit: same (game_idx, sorted-roster) tuple.
    team_sizes = arrays["team_sizes"]
    player_flat = arrays["player_indices_flat"]
    offsets = np.empty(len(team_sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(team_sizes, out=offsets[1:])
    game_idx_all = arrays["game_idx"]
    games_final = result.players.games  # per-player game count after train

    team_key_list: list[tuple] = []
    n_new_list: list[int] = []
    for i in obs_idx:
        s, e = int(offsets[i]), int(offsets[i + 1])
        roster = player_flat[s:e]
        team_key_list.append(
            (int(game_idx_all[i]), tuple(sorted(int(p) for p in roster)))
        )
        n_new = int((games_final[roster] < 30).sum())
        n_new_list.append(n_new)
    team_key_arr = np.array(team_key_list, dtype=object)
    n_new_arr = np.array(n_new_list, dtype=np.int64)

    # Filter to holdout obs only.  Iterate over hk via Python list
    # because numpy object-array indexing returns ndarrays for tuple
    # cells, which are unhashable.
    holdout_positions = np.where(is_holdout)[0]
    hk = [team_key_list[i] for i in holdout_positions]
    hn = n_new_arr[is_holdout]
    hp = pred_p[is_holdout]
    hy = actual_y[is_holdout]

    # Aggregate per team: sum(p), sum(y), n_obs (= n_questions), n_new (constant).
    from collections import defaultdict
    agg: dict[tuple, dict] = defaultdict(
        lambda: {"sum_p": 0.0, "sum_y": 0.0, "n_obs": 0, "n_new": 0}
    )
    for k, n, p, y in zip(hk, hn, hp, hy):
        a = agg[k]
        a["sum_p"] += float(p)
        a["sum_y"] += int(y)
        a["n_obs"] += 1
        a["n_new"] = int(n)

    # Bucket.
    buckets: dict[str, list[float]] = {
        "0": [], "1": [], "2": [], "3+": [],
    }
    for k, a in agg.items():
        n_new = a["n_new"]
        if n_new == 0:
            b = "0"
        elif n_new == 1:
            b = "1"
        elif n_new == 2:
            b = "2"
        else:
            b = "3+"
        # Bias on the holdout fraction of this team's questions.
        # We compare per-question expected-vs-actual averaged over
        # the (random ~10%) cells we held out — same denominator.
        bias = (a["sum_y"] - a["sum_p"]) / max(a["n_obs"], 1)
        buckets[b].append(bias)

    out = {}
    for b, vals in buckets.items():
        if not vals:
            out[b] = {"n_teams": 0, "mean_bias_per_q": float("nan")}
        else:
            out[b] = {
                "n_teams": len(vals),
                "mean_bias_per_q": float(np.mean(vals)),
                "median_bias_per_q": float(np.median(vals)),
            }
    return out


def run_one(
    cache_file: str,
    min_games_label: int,
    cold_theta: float,
    games_offset: float,
) -> dict:
    print(
        f"\n===== mg={min_games_label}  θ_cold={cold_theta:+.2f}  "
        f"offset={games_offset:.2f} =====",
        flush=True,
    )
    arrays, maps = load_cached(cache_file)
    cfg = Config(
        cold_init_theta=cold_theta,
        games_offset=games_offset,
        holdout_obs_fraction=0.10,
        holdout_seed=42,
    )
    res = run_sequential(
        arrays, maps, cfg,
        verbose=False,
        collect_history=False,
        collect_predictions=True,
    )
    preds = res.predictions or {}
    is_holdout = preds["is_holdout"].astype(bool)
    p = preds["pred_p"][is_holdout]
    y = preds["actual_y"][is_holdout]
    m = compute_metrics(p, y)

    bias = _team_bias_diagnostic(res, arrays, maps)
    row = {
        "min_games": min_games_label,
        "cold_init_theta": cold_theta,
        "games_offset": games_offset,
        "logloss": m["logloss"],
        "brier": m["brier"],
        "auc": m["auc"],
        "n_test_obs": int(is_holdout.sum()),
        "bias_by_n_new": bias,
    }
    bias_str = "  ".join(
        f"{k}={v.get('mean_bias_per_q', float('nan')):+.4f}"
        f"(n={v.get('n_teams',0)})"
        for k, v in bias.items()
    )
    print(
        f"  logloss {row['logloss']:.4f}  Brier {row['brier']:.4f}  "
        f"AUC {row['auc']:.4f}\n  bias/q by n_new: {bias_str}",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-mg0", required=True)
    ap.add_argument("--cache-mg5", required=True)
    ap.add_argument("--cache-mg10", required=True)
    ap.add_argument(
        "--out", default="results/exp_min_games_cold_grid.json"
    )
    args = ap.parse_args()

    cells = [
        (0, args.cache_mg0),
        (5, args.cache_mg5),
        (10, args.cache_mg10),
    ]

    rows: list[dict] = []
    for mg, cache in cells:
        for theta in GRID_COLD:
            for offset in GRID_OFFSET:
                row = run_one(cache, mg, theta, offset)
                rows.append(row)
                # Snapshot results after each cell so we don't lose
                # ~30 min of compute if the next cell crashes.
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                with open(args.out, "w") as f:
                    json.dump({"rows": rows}, f, indent=2)

    # ---- Print summary tables ----
    print("\n" + "=" * 110)
    print(
        f"{'mg':>3} {'θ_cold':>7} {'offset':>7} "
        f"{'logloss':>9} {'Brier':>8} {'AUC':>7} "
        f"{'bias_0':>8} {'bias_1':>8} {'bias_2':>8} {'bias_3+':>8} "
        f"{'n_test':>9}"
    )
    print("-" * 110)
    for r in rows:
        b = r["bias_by_n_new"]
        print(
            f"{r['min_games']:>3} {r['cold_init_theta']:>+7.2f} "
            f"{r['games_offset']:>7.2f} "
            f"{r['logloss']:>9.4f} {r['brier']:>8.4f} {r['auc']:>7.4f} "
            f"{b.get('0',{}).get('mean_bias_per_q', float('nan')):>+8.4f} "
            f"{b.get('1',{}).get('mean_bias_per_q', float('nan')):>+8.4f} "
            f"{b.get('2',{}).get('mean_bias_per_q', float('nan')):>+8.4f} "
            f"{b.get('3+',{}).get('mean_bias_per_q', float('nan')):>+8.4f} "
            f"{r['n_test_obs']:>9}"
        )
    print("=" * 110)
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
