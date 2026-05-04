"""1D w_solo sweep under honest cell-holdout, lapse-rate on.

Before lapse rate, ``w_solo=0.3`` was tuned to dampen the noisy solo
gradient.  After lapse rate took over the noise-absorption role for
high-p misses, w_solo may have become over-dampening (Семушин's θ
grows too slowly because his ~35 % solo games update θ at only
0.3 × ~0.9 ≈ 27 % of nominal strength).

Outputs ``results/exp_w_solo_sweep_honest.csv`` with metrics
overall and on the solo (team_size==1) slice.
Cost: 5 × ~9 min ≈ 45 min total.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_w_solo_sweep_honest.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--values", default="0.30,0.50,0.70,0.90,1.10",
        help="comma-separated w_solo values",
    )
    args = ap.parse_args()

    values = [float(x) for x in args.values.split(",")]
    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)

    team_sizes = arrays["team_sizes"].astype(np.int32)

    rows: list[dict] = []
    for w_solo in values:
        print(f"\n=== w_solo={w_solo:.3f} ===", flush=True)
        cfg = Config(
            w_solo=w_solo,
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
        obs = pred["obs_idx"][mask]
        ts = team_sizes[obs]

        # Overall
        m = compute_metrics(p, y)
        rows.append({
            "w_solo": w_solo,
            "slice": "all",
            "n": int(mask.sum()),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
            "elapsed_sec": round(elapsed, 1),
        })
        print(
            f"  overall  : ll={m['logloss']:.4f}  AUC={m['auc']:.4f}"
            f"  ({elapsed:.1f}s)",
            flush=True,
        )

        # Solo / non-solo slices
        for label, sl in [
            ("solo", ts == 1),
            ("team_2_5", (ts >= 2) & (ts <= 5)),
            ("team_6plus", ts >= 6),
        ]:
            if not sl.any():
                continue
            ms = compute_metrics(p[sl], y[sl])
            rows.append({
                "w_solo": w_solo,
                "slice": label,
                "n": int(sl.sum()),
                "logloss": round(float(ms["logloss"]), 6),
                "brier": round(float(ms["brier"]), 6),
                "auc": (
                    round(float(ms["auc"]), 6)
                    if not np.isnan(ms["auc"]) else ""
                ),
                "elapsed_sec": "",
            })
            print(
                f"  {label:10s}: ll={ms['logloss']:.4f} AUC={ms['auc']:.4f}",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "w_solo", "slice", "n", "logloss", "brier",
                "auc", "elapsed_sec",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print("\n=== Ranked by overall logloss ===")
    overall = sorted(
        [r for r in rows if r["slice"] == "all"], key=lambda r: r["logloss"]
    )
    for r in overall:
        marker = "  ★" if r is overall[0] else ""
        print(
            f"  w_solo={r['w_solo']:.3f}  ll={r['logloss']:.4f}  "
            f"AUC={r['auc']}{marker}"
        )
    print("\n=== Ranked by solo-slice logloss ===")
    solos = sorted(
        [r for r in rows if r["slice"] == "solo"], key=lambda r: r["logloss"]
    )
    for r in solos:
        marker = "  ★" if r is solos[0] else ""
        print(
            f"  w_solo={r['w_solo']:.3f}  solo_ll={r['logloss']:.4f}"
            f"{marker}"
        )

    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
