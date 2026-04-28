"""Re-tune δ_size: drop the L2 reg and/or raise the step size.

Hypothesis (docs/error_structure_2026-04.md §1.3 + new §3): the
+0.04 / +0.02 residual on size 1 / 2 is a δ_size under-fit, pulled
back toward 0 by ``reg_size = 0.10``.  Upper bound for any size-only
calibration is −0.00051 logloss (post-hoc, in-sample).

Variants:
  v0  baseline (sanity)
  v1  reg_size = 0.0
  v2  reg_size = 0.0, eta_size = 0.005 (5× step)

Output: results/exp_size_retune.csv + the learned δ_size for each.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from data import load_cached
from rating.backtest import backtest
from rating.engine import Config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out_csv", default="results/exp_size_retune.csv")
    args = ap.parse_args()

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)

    variants = [
        ("v0_baseline", dict()),
        ("v1_no_reg", dict(reg_size=0.0)),
        ("v2_no_reg_5x_eta", dict(reg_size=0.0, eta_size=0.005)),
    ]

    rows = []
    for name, overrides in variants:
        cfg = Config(**overrides)
        print(f"\n=== {name}  overrides={overrides} ===")
        t0 = time.time()
        m = backtest(arrays, maps, cfg, verbose=False)
        elapsed = time.time() - t0

        ds = np.asarray(m["result"].delta_size).round(4).tolist()
        row = {
            "variant": name,
            "logloss": float(m["logloss"]),
            "brier": float(m["brier"]),
            "auc": float(m["auc"]),
            "elapsed_sec": round(elapsed, 1),
            "delta_size": json.dumps(ds),
        }
        for t in ("offline", "sync", "async"):
            sub = m.get("by_type", {}).get(t, {})
            row[f"ll_{t}"] = sub.get("logloss", float("nan"))
        for q in (1, 2, 3, 4):
            sub = m.get("by_hardness", {}).get(f"q{q}", {})
            row[f"ll_q{q}"] = sub.get("logloss", float("nan"))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] → {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
