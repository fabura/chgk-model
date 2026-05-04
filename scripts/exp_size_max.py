"""Diagnose under-prediction on team-size 7+ by widening team_size_max.

Currently ``team_size_max=8`` collapses sizes 8, 9, 10, 11+ into a
single bucket (about 1.0 % of all observations).  The bucket gets
pulled to a positive δ that under-predicts ordinary 7-8-player teams
(calibration: bias −3..−5 p.p. in p∈[0.5, 0.7]).

This script compares (under the leakage-free 10 % cell hold-out):

* ``current``    — ``team_size_max=8``
* ``size_max10`` — ``team_size_max=10`` (separate δ per size)
* ``size_max12`` — ``team_size_max=12`` (covers ~99.99 % of obs)

Outputs ``results/exp_size_max.csv`` plus learned δ_size vectors.
Total runtime ≈ 3 × backtest pass ≈ 24 min.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential


def _bucket_type(g: str) -> str:
    if "async" in g:
        return "async"
    if "sync" in g:
        return "sync"
    return "offline"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_size_max.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)

    base = dict(holdout_obs_fraction=args.holdout, holdout_seed=args.seed)
    configs = {
        "current": Config(**base, team_size_max=8),
        "size_max10": Config(**base, team_size_max=10),
        "size_max12": Config(**base, team_size_max=12),
    }

    rows: list[dict] = []
    for name, cfg in configs.items():
        print(f"\n=== {name}  team_size_max={cfg.team_size_max} ===", flush=True)
        t0 = time.time()
        result = run_sequential(
            arrays, maps, cfg, verbose=False, collect_predictions=True
        )
        elapsed = time.time() - t0

        pred = result.predictions
        mask = pred["is_holdout"].astype(bool)
        p = pred["pred_p"][mask]
        y = pred["actual_y"][mask]
        g = pred["game_idx"][mask]

        # Per-obs team-size for stratification.
        obs = pred["obs_idx"][mask]
        team_sizes = arrays["team_sizes"][obs]

        m = compute_metrics(p, y)
        rows.append({
            "variant": name,
            "slice": "all",
            "team_size_max": cfg.team_size_max,
            "n": int(mask.sum()),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
            "elapsed_sec": round(elapsed, 1),
        })
        print(
            f"  overall : n={int(mask.sum()):>9d}  ll={m['logloss']:.4f}  "
            f"brier={m['brier']:.4f}  AUC={m['auc']:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

        # Per-size 7+ strata (and a couple of reference sizes).
        for sz_label, sel in [
            ("6", team_sizes == 6),
            ("7", team_sizes == 7),
            ("8", team_sizes == 8),
            ("9", team_sizes == 9),
            ("10+", team_sizes >= 10),
        ]:
            if not sel.any():
                continue
            mp = float(p[sel].mean())
            my = float(y[sel].mean())
            ms = compute_metrics(p[sel], y[sel])
            rows.append({
                "variant": name,
                "slice": f"size={sz_label}",
                "team_size_max": cfg.team_size_max,
                "n": int(sel.sum()),
                "logloss": round(float(ms["logloss"]), 6),
                "brier": round(float(ms["brier"]), 6),
                "auc": round(float(ms["auc"]), 6) if not np.isnan(ms["auc"]) else "",
                "elapsed_sec": "",
                "mean_p": round(mp, 4),
                "mean_y": round(my, 4),
                "bias": round(mp - my, 4),
            })
            print(
                f"  size={sz_label:>3s}: n={int(sel.sum()):>8d}  "
                f"ll={ms['logloss']:.4f}  "
                f"p={mp:.3f}  y={my:.3f}  bias={mp - my:+.4f}",
                flush=True,
            )

        # Print learned δ_size.
        ds = result.delta_size
        if ds is not None:
            print(
                "  δ_size: "
                + ", ".join(
                    f"[{i}]={ds[i]:+.4f}"
                    + ("*" if i == result.team_size_anchor else "")
                    for i in range(1, len(ds))
                ),
                flush=True,
            )

    # Save.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant", "slice", "team_size_max", "n", "logloss", "brier",
        "auc", "elapsed_sec", "mean_p", "mean_y", "bias",
    ]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
