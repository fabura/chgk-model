"""Cross-tab residuals by (team_size × tournament_mode).

Purpose: figure out whether the small-team residual (§1.3,
``docs/error_structure_2026-04.md``) is uniform across modes (=>
re-tune global δ_size) or mode-dependent (=> need δ_size[mode][size]).

Single backtest, then a 2-D pivot.  Also dumps the raw test-set
predictions to ``results/test_predictions.npz`` so subsequent
small-team experiments can skip the 8-min backtest.

Output:
  * stdout: pivot table  (size × mode  →  n / mean_p̂ / mean_y / Δ / ll)
  * results/diag_size_by_mode.csv
  * results/test_predictions.npz   (one-off cache)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from data import load_cached
from rating.backtest import backtest
from rating.engine import Config


def _test_mask(pred_g: np.ndarray, gdo: np.ndarray | None) -> np.ndarray:
    """Re-derive the 20 % time-tail test set exactly like backtest()."""
    all_games = np.unique(pred_g)
    if gdo is not None:
        known = all_games[
            np.array([int(gdo[g]) >= 0 for g in all_games], dtype=bool)
        ]
        ordered = known[np.argsort(np.array([int(gdo[g]) for g in known]))]
    else:
        ordered = np.sort(all_games)
    n_test = max(1, int(len(ordered) * 0.2))
    test_games = set(int(g) for g in ordered[-n_test:])
    return np.fromiter(
        (int(g) in test_games for g in pred_g),
        count=len(pred_g),
        dtype=bool,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out_csv", default="results/diag_size_by_mode.csv")
    ap.add_argument(
        "--cache_pred",
        default="results/test_predictions.npz",
        help="dump the raw test predictions here for re-use",
    )
    args = ap.parse_args()

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)

    print("[backtest] running once with predictions…")
    cfg = Config()
    metrics = backtest(arrays, maps, cfg, verbose=False)
    pred = metrics["result"].predictions
    if pred is None or "obs_idx" not in pred:
        raise RuntimeError("predictions need obs_idx (engine patch)")

    pred_g = pred["game_idx"]
    pred_obs = pred["obs_idx"]
    pred_p = pred["pred_p"]
    actual_y = pred["actual_y"]

    test_mask = _test_mask(pred_g, getattr(maps, "game_date_ordinal", None))
    p = pred_p[test_mask]
    y = actual_y[test_mask].astype(np.float64)
    g = pred_g[test_mask]
    obs = pred_obs[test_mask].astype(np.int64)

    eps = 1e-15
    p_clip = np.clip(p, eps, 1.0 - eps)
    ll = -(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))
    res = y - p
    print(f"[test] n_obs = {len(p):,}  mean_logloss = {ll.mean():.4f}")

    team_size_arr = arrays["team_sizes"][obs]
    gtype_arr = getattr(maps, "game_type", None)
    if gtype_arr is None:
        modes = np.array(["offline"] * len(g))
    else:
        modes = np.array([str(gtype_arr[int(gi)]) for gi in g])

    sizes_to_show = [1, 2, 3, 4, 5, 6, 7, 8]
    modes_to_show = ["offline", "sync", "async"]

    rows = []
    for s in sizes_to_show:
        for m in modes_to_show:
            mask = (team_size_arr == s) & (modes == m)
            n = int(mask.sum())
            if n == 0:
                rows.append({
                    "size": s, "mode": m, "n": 0,
                    "mean_p": float("nan"), "mean_y": float("nan"),
                    "delta": float("nan"), "logloss": float("nan"),
                })
                continue
            mp = float(p[mask].mean())
            my = float(y[mask].mean())
            ml = float(ll[mask].mean())
            rows.append({
                "size": s, "mode": m, "n": n,
                "mean_p": mp, "mean_y": my,
                "delta": my - mp, "logloss": ml,
            })

    # Add row totals.
    for s in sizes_to_show:
        mask = team_size_arr == s
        n = int(mask.sum())
        if n == 0:
            continue
        mp = float(p[mask].mean())
        my = float(y[mask].mean())
        ml = float(ll[mask].mean())
        rows.append({
            "size": s, "mode": "ALL", "n": n,
            "mean_p": mp, "mean_y": my,
            "delta": my - mp, "logloss": ml,
        })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["size", "mode", "n", "mean_p", "mean_y", "delta", "logloss"],
        )
        w.writeheader()
        w.writerows(rows)

    # Pretty pivot to stdout.
    print()
    print(f"{'size':<6}{'mode':<10}{'n':>10}{'p̂':>10}{'y':>10}{'Δ=y-p̂':>12}{'logloss':>11}")
    print("-" * 60)
    for r in rows:
        if r["n"] == 0:
            continue
        print(
            f"{r['size']:<6}{r['mode']:<10}{r['n']:>10,}"
            f"{r['mean_p']:>10.3f}{r['mean_y']:>10.3f}"
            f"{r['delta']:>+12.4f}{r['logloss']:>11.4f}"
        )

    # Cache predictions for the next experiment.
    Path(args.cache_pred).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache_pred,
        pred_g=pred_g, pred_obs=pred_obs,
        pred_p=pred_p, actual_y=actual_y,
        team_theta_mean=pred["team_theta_mean"],
    )
    print(f"\n[ok] table → {args.out_csv}")
    print(f"[ok] preds  → {args.cache_pred}")

    delta_size = getattr(metrics["result"], "delta_size", None)
    if delta_size is not None:
        print(f"\nLearned δ_size (index = team_size): "
              f"{np.asarray(delta_size).round(4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
