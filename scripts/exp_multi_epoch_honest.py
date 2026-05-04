"""Multi-epoch warm-start ablation, re-run under the honest cell-holdout.

Earlier ``scripts/exp_multi_epoch.py`` was run before the 2026-05
leakage fix.  It found that ``n_extra_epochs=1`` gave a small
improvement (-0.0065 logloss on leaky), and ``2+`` overfit.  We
re-validate that finding now under the per-cell hold-out so we know
whether multi-pass remains useful with honest evaluation and the
2026-05 cleaned defaults (freeze_log_a=True, team_size_max=12).

Outputs ``results/exp_multi_epoch_honest.csv``.
Total runtime ≈ Σ(1 + N) × backtest pass per variant.
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
    ap.add_argument("--out", default="results/exp_multi_epoch_honest.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--epochs", default="0,1,2,4",
        help="comma-separated n_extra_epochs values to evaluate",
    )
    args = ap.parse_args()

    epochs_list = [int(x) for x in args.epochs.split(",")]
    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)

    rows: list[dict] = []
    for n_extra in epochs_list:
        print(f"\n=== n_extra_epochs={n_extra} ===", flush=True)
        cfg = Config(
            n_extra_epochs=n_extra,
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
            "n_extra_epochs": n_extra,
            "slice": "all",
            "n": int(mask.sum()),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
            "elapsed_sec": round(elapsed, 1),
        }
        rows.append(row)
        print(
            f"  overall : n={row['n']:>9d}  ll={row['logloss']:.4f}  "
            f"brier={row['brier']:.4f}  AUC={row['auc']}  ({elapsed:.1f}s)",
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
                    "n_extra_epochs": n_extra,
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
                    f"  {t:7s}: n={int(tm.sum()):>9d}  "
                    f"ll={ms['logloss']:.4f}  AUC={ms['auc']:.4f}",
                    flush=True,
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "n_extra_epochs", "slice", "n", "logloss", "brier",
                "auc", "elapsed_sec",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
