"""Sweep ``eta_teammate`` under honest cell-holdout.

Tests whether stronger per-tournament teammate θ-shrinkage reduces the
noisy-OR identifiability gap on stable rosters without hurting overall
logloss.  Also records final θ for three case-study players and their
star teammates.

Outputs ``results/exp_eta_teammate_sweep_honest.csv``.

Usage::

    python scripts/exp_eta_teammate_sweep_honest.py --cache_file data.npz

Cost: ~6 trials × ~7 min ≈ 40 min.
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
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential

CASE_STUDY = {
    34909: "Chernukha",
    26818: "Rekshinskaya",
    158668: "Monina",
}


def _bucket_type(g: str) -> str:
    if "async" in g:
        return "async"
    if "sync" in g:
        return "sync"
    return "offline"


def _player_thetas(result, maps, pids: dict[int, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    pid_to_idx = {int(pid): i for i, pid in enumerate(maps.idx_to_player_id)}
    for db_id, label in pids.items():
        idx = pid_to_idx.get(int(db_id))
        if idx is None:
            out[f"theta_{label}"] = float("nan")
        else:
            out[f"theta_{label}"] = round(float(result.players.theta[idx]), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_eta_teammate_sweep_honest.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--values",
        default="0.0,0.005,0.01,0.015,0.02,0.03",
        help="comma-separated eta_teammate values",
    )
    args = ap.parse_args()

    eta_values = [float(x) for x in args.values.split(",")]
    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)

    # Resolve reference player IDs from maps if Brief id wrong
    all_pids = {int(p) for p in maps.idx_to_player_id}
    case_ids = {k: v for k, v in CASE_STUDY.items() if k in all_pids}
    missing = set(CASE_STUDY) - set(case_ids)
    if missing:
        print(f"[warn] case-study ids not in cache: {missing}", flush=True)

    rows: list[dict] = []
    for eta in eta_values:
        print(f"\n=== eta_teammate={eta:.4f} ===", flush=True)
        cfg = Config(
            eta_teammate=eta,
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
        row: dict = {
            "eta_teammate": eta,
            "slice": "all",
            "n": int(mask.sum()),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
            "elapsed_sec": round(elapsed, 1),
        }
        row.update(_player_thetas(result, maps, case_ids))
        rows.append(row)
        print(
            f"  overall : ll={row['logloss']:.4f}  brier={row['brier']:.4f} "
            f"AUC={row['auc']}  ({elapsed:.1f}s)",
            flush=True,
        )
        for label in CASE_STUDY.values():
            key = f"theta_{label}"
            if key in row:
                print(f"  {key}={row[key]:+.4f}", flush=True)

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
                slice_row = {
                    "eta_teammate": eta,
                    "slice": t,
                    "n": int(tm.sum()),
                    "logloss": round(float(ms["logloss"]), 6),
                    "brier": round(float(ms["brier"]), 6),
                    "auc": (
                        round(float(ms["auc"]), 6)
                        if not np.isnan(ms["auc"]) else ""
                    ),
                    "elapsed_sec": "",
                }
                rows.append(slice_row)
                print(
                    f"  {t:7s}: ll={ms['logloss']:.4f} AUC={ms['auc']:.4f}",
                    flush=True,
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    overall = [r for r in rows if r["slice"] == "all"]
    overall.sort(key=lambda r: r["logloss"])
    print("\n=== Ranked overall (best → worst) ===")
    for r in overall:
        marker = "  ★" if r is overall[0] else ""
        thetas = "  ".join(
            f"{k.replace('theta_', '')}={r[k]:+.3f}"
            for k in sorted(r)
            if k.startswith("theta_") and r[k] != ""
        )
        print(
            f"  eta={r['eta_teammate']:.4f}  ll={r['logloss']:.4f}  "
            f"AUC={r['auc']}  {thetas}{marker}"
        )

    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
