"""Calibration analysis on the time-split backtest.

Buckets test predictions into probability bins (0.0-0.1, 0.1-0.2, …,
0.9-1.0) and compares mean predicted probability to the empirical take
rate within each bucket.  Stratifies by tournament type, team size,
and roster-strength quartile (same cuts as ``backtest()``).

Outputs a tidy CSV with one row per (slice, bucket).

Usage::

    python -m scripts.calibration --cache_file data.npz \
        --out results/calibration_2026-04.csv

The script re-runs ``run_sequential`` with ``collect_predictions=True``
so it does not depend on any pre-saved predictions file.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from data import load_cached
from rating.engine import Config, run_sequential


# Default p-bucket edges (10 equal-width bins).
P_EDGES = np.linspace(0.0, 1.0, 11)


def bucket_idx(p: np.ndarray, edges: np.ndarray = P_EDGES) -> np.ndarray:
    """Map probabilities to bucket index 0..len(edges)-2."""
    idx = np.searchsorted(edges, p, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def calibration_table(
    p: np.ndarray, y: np.ndarray, edges: np.ndarray = P_EDGES
) -> list[dict]:
    """One row per bucket: count, mean(p), mean(y), bias, mae."""
    bi = bucket_idx(p, edges)
    rows: list[dict] = []
    for k in range(len(edges) - 1):
        mask = bi == k
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bucket_lo": float(edges[k]),
                "bucket_hi": float(edges[k + 1]),
                "n": 0,
                "mean_p": float("nan"),
                "mean_y": float("nan"),
                "bias": float("nan"),
            })
            continue
        mp = float(p[mask].mean())
        my = float(y[mask].mean())
        rows.append({
            "bucket_lo": float(edges[k]),
            "bucket_hi": float(edges[k + 1]),
            "n": n,
            "mean_p": mp,
            "mean_y": my,
            "bias": mp - my,
        })
    return rows


def team_size_label(ts: int) -> str:
    """Group team sizes into a few buckets for stratification."""
    if ts <= 1:
        return "1"
    if ts == 2:
        return "2"
    if ts in (3, 4):
        return "3-4"
    if ts == 5:
        return "5"
    if ts == 6:
        return "6"
    return "7+"


def tournament_type_label(gt: str) -> str:
    if "async" in gt:
        return "async"
    if "sync" in gt:
        return "sync"
    return "offline"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument(
        "--out", default="results/calibration_2026-04.csv",
        help="output CSV path",
    )
    ap.add_argument(
        "--test_fraction", type=float, default=0.2,
        help="fraction of latest tournaments treated as test set "
             "(only used when --holdout=0.0)",
    )
    ap.add_argument(
        "--holdout", type=float, default=0.10,
        help="per-cell hold-out fraction (default 0.10 = honest "
             "leakage-free calibration). Set to 0.0 for legacy "
             "time-split calibration.",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="hold-out random seed",
    )
    args = ap.parse_args()

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)

    cfg = Config(
        holdout_obs_fraction=args.holdout,
        holdout_seed=args.seed,
    )
    mode_label = (
        f"honest cell-holdout (h={args.holdout}, seed={args.seed})"
        if args.holdout > 0.0 else
        f"legacy time-split (last {args.test_fraction:.0%})"
    )
    print(f"[run] backtest pass — {mode_label}")
    result = run_sequential(arrays, maps, cfg, verbose=False, collect_predictions=True)

    if result.predictions is None:
        print("ERROR: no predictions collected")
        return 1

    pred = result.predictions
    p_all = pred["pred_p"]
    y_all = pred["actual_y"]
    g_all = pred["game_idx"]
    obs_all = pred["obs_idx"]
    thbar_all = pred.get("team_theta_mean")
    is_holdout = pred.get("is_holdout")

    if args.holdout > 0.0 and is_holdout is not None:
        test_mask = is_holdout.astype(bool)
    else:
        # Legacy time-split: replicate backtest()'s test-game selection.
        gdo = getattr(maps, "game_date_ordinal", None)
        all_games = np.unique(g_all)
        if gdo is not None:
            known = all_games[
                np.array([gdo[g] >= 0 for g in all_games], dtype=bool)
            ]
            ordered = (
                known[np.argsort(np.array([gdo[g] for g in known]))]
                if len(known) >= 2 else np.sort(all_games)
            )
        else:
            ordered = np.sort(all_games)
        n_test = max(1, int(len(ordered) * args.test_fraction))
        test_games = set(int(g) for g in ordered[-n_test:])
        test_mask = np.array(
            [int(g) in test_games for g in g_all], dtype=bool
        )

    p = p_all[test_mask]
    y = y_all[test_mask].astype(np.float64)
    g = g_all[test_mask]
    obs = obs_all[test_mask]

    print(f"[test] {len(p)} obs across {len(np.unique(g))} tournaments")

    # Per-obs metadata for stratification.
    team_sizes = arrays["team_sizes"][obs]
    gtype = getattr(maps, "game_type", None)
    if gtype is not None:
        types = np.array(
            [tournament_type_label(str(gtype[gi])) for gi in g],
            dtype=object,
        )
    else:
        types = np.full(len(p), "offline", dtype=object)

    # Hardness quartile based on per-tournament mean θ̄ (same as backtest).
    if thbar_all is not None:
        thbar = thbar_all[test_mask]
        unique_g, inv = np.unique(g, return_inverse=True)
        sums = np.zeros(len(unique_g), dtype=np.float64)
        cnts = np.zeros(len(unique_g), dtype=np.int64)
        np.add.at(sums, inv, thbar)
        np.add.at(cnts, inv, 1)
        per_g_thbar = sums / np.maximum(cnts, 1)
        cuts = np.unique(np.quantile(per_g_thbar, [0.25, 0.5, 0.75]))
        g_to_q = np.searchsorted(cuts, per_g_thbar, side="right") + 1
        obs_q = g_to_q[inv]
    else:
        obs_q = np.ones(len(p), dtype=np.int32)

    # --- assemble rows ------------------------------------------------
    out_rows: list[dict] = []

    def _emit(slice_kind: str, slice_value: str, mask: np.ndarray) -> None:
        if not mask.any():
            return
        for r in calibration_table(p[mask], y[mask]):
            out_rows.append({
                "slice_kind": slice_kind,
                "slice_value": slice_value,
                **r,
            })

    _emit("overall", "all", np.ones(len(p), dtype=bool))

    for t in ("offline", "sync", "async"):
        _emit("type", t, types == t)

    ts_labels = np.array([team_size_label(int(s)) for s in team_sizes], dtype=object)
    for label in ("1", "2", "3-4", "5", "6", "7+"):
        _emit("team_size", label, ts_labels == label)

    for q in (1, 2, 3, 4):
        _emit("hardness_q", f"q{q}", obs_q == q)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "slice_kind", "slice_value", "bucket_lo", "bucket_hi",
                "n", "mean_p", "mean_y", "bias",
            ],
        )
        w.writeheader()
        for r in out_rows:
            w.writerow({
                "slice_kind": r["slice_kind"],
                "slice_value": r["slice_value"],
                "bucket_lo": round(r["bucket_lo"], 3),
                "bucket_hi": round(r["bucket_hi"], 3),
                "n": r["n"],
                "mean_p": (
                    "" if r["n"] == 0 else round(r["mean_p"], 4)
                ),
                "mean_y": (
                    "" if r["n"] == 0 else round(r["mean_y"], 4)
                ),
                "bias": (
                    "" if r["n"] == 0 else round(r["bias"], 4)
                ),
            })
    print(f"[ok] {len(out_rows)} rows → {out_path}")

    # --- pretty-print summary ----------------------------------------
    print("\n=== Overall calibration (10 buckets of width 0.1) ===")
    print(
        f"{'bucket':>10s}  {'n':>9s}  "
        f"{'mean_p':>7s}  {'mean_y':>7s}  {'bias':>7s}"
    )
    for r in (x for x in out_rows if x["slice_kind"] == "overall"):
        bucket = f"{r['bucket_lo']:.1f}-{r['bucket_hi']:.1f}"
        if r["n"] == 0:
            print(f"  {bucket:>8s}  {'0':>9s}  {'':>7s}  {'':>7s}  {'':>7s}")
            continue
        print(
            f"  {bucket:>8s}  {r['n']:>9d}  "
            f"{r['mean_p']:>7.4f}  {r['mean_y']:>7.4f}  {r['bias']:>+7.4f}"
        )

    print("\n=== By tournament type — mean (p, y, bias) per bucket ===")
    for t in ("offline", "sync", "async"):
        print(f"  --- {t} ---")
        for r in (
            x for x in out_rows
            if x["slice_kind"] == "type" and x["slice_value"] == t
        ):
            bucket = f"{r['bucket_lo']:.1f}-{r['bucket_hi']:.1f}"
            if r["n"] == 0:
                continue
            print(
                f"    {bucket:>7s}  n={r['n']:>8d}  "
                f"p={r['mean_p']:.3f}  y={r['mean_y']:.3f}  "
                f"bias={r['bias']:+.4f}"
            )

    print("\n=== By team size ===")
    for label in ("1", "2", "3-4", "5", "6", "7+"):
        print(f"  --- size={label} ---")
        for r in (
            x for x in out_rows
            if x["slice_kind"] == "team_size" and x["slice_value"] == label
        ):
            bucket = f"{r['bucket_lo']:.1f}-{r['bucket_hi']:.1f}"
            if r["n"] == 0:
                continue
            print(
                f"    {bucket:>7s}  n={r['n']:>8d}  "
                f"p={r['mean_p']:.3f}  y={r['mean_y']:.3f}  "
                f"bias={r['bias']:+.4f}"
            )

    print("\n=== By roster-strength quartile (q1=weakest, q4=strongest) ===")
    for q in (1, 2, 3, 4):
        print(f"  --- {q} ---")
        for r in (
            x for x in out_rows
            if x["slice_kind"] == "hardness_q" and x["slice_value"] == f"q{q}"
        ):
            bucket = f"{r['bucket_lo']:.1f}-{r['bucket_hi']:.1f}"
            if r["n"] == 0:
                continue
            print(
                f"    {bucket:>7s}  n={r['n']:>8d}  "
                f"p={r['mean_p']:.3f}  y={r['mean_y']:.3f}  "
                f"bias={r['bias']:+.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
