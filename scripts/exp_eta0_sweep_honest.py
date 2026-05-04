"""Quick 1D sanity sweep over eta0 under honest cell-holdout.

After the 2026-05 cleanup (freeze_log_a=True, team_size_max=12,
n_extra_epochs=1), the previously-tuned eta0 might no longer be
optimal.  This script runs 5 trials around the current default
(0.15) to confirm or deny that we need a fuller re-tune.

Outputs ``results/exp_eta0_sweep_honest.csv``.
Cost: ~5 × (1 + 1) × ~7 min ≈ 70 min total.
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
    ap.add_argument("--out", default="results/exp_eta0_sweep_honest.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--values", default="0.10,0.12,0.15,0.18,0.22",
        help="comma-separated eta0 values",
    )
    args = ap.parse_args()

    eta_values = [float(x) for x in args.values.split(",")]
    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)

    rows: list[dict] = []
    for eta0 in eta_values:
        print(f"\n=== eta0={eta0:.3f} ===", flush=True)
        cfg = Config(
            eta0=eta0,
            holdout_obs_fraction=args.holdout,
            holdout_seed=args.seed,
        )
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

        m = compute_metrics(p, y)
        row = {
            "eta0": eta0,
            "slice": "all",
            "n": int(mask.sum()),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
            "elapsed_sec": round(elapsed, 1),
        }
        rows.append(row)
        print(
            f"  overall : ll={row['logloss']:.4f}  brier={row['brier']:.4f} "
            f"AUC={row['auc']}  ({elapsed:.1f}s)",
            flush=True,
        )

        gtype = getattr(maps, "game_type", None)
        if gtype is not None:
            types = np.array(
                [_bucket_type(str(gtype[gi])) for gi in g], dtype=object
            )
            for t in ("offline", "sync", "async"):
                tm = types == t
                if not tm.any():
                    continue
                ms = compute_metrics(p[tm], y[tm])
                rows.append({
                    "eta0": eta0,
                    "slice": t,
                    "n": int(tm.sum()),
                    "logloss": round(float(ms["logloss"]), 6),
                    "brier": round(float(ms["brier"]), 6),
                    "auc": (
                        round(float(ms["auc"]), 6)
                        if not np.isnan(ms["auc"]) else ""
                    ),
                    "elapsed_sec": "",
                })
                print(
                    f"  {t:7s}: ll={ms['logloss']:.4f} AUC={ms['auc']:.4f}",
                    flush=True,
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "eta0", "slice", "n", "logloss", "brier",
                "auc", "elapsed_sec",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    overall = [r for r in rows if r["slice"] == "all"]
    overall.sort(key=lambda r: r["logloss"])
    print("\n=== Ranked overall (best → worst) ===")
    for r in overall:
        marker = "  ★" if r is overall[0] else ""
        print(
            f"  eta0={r['eta0']:.3f}  ll={r['logloss']:.4f}  "
            f"AUC={r['auc']}{marker}"
        )

    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
