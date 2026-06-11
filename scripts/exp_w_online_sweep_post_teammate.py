"""Sweep ``w_online`` (async θ-update weight) under honest cell-holdout.

Async tournaments route player θ updates through ``w_online`` (see
``_type_update_weights``).  After bumping ``eta_teammate`` to 0.02,
this sweep checks whether discounting async θ steps helps players whose
ratings are dragged down by weak/noisy async rosters (case study:
Rekshinskaya).

Fixed: ``eta_teammate=0.02`` (new production default).

Outputs ``results/exp_w_online_sweep_post_teammate.csv``.

Usage::

    python scripts/exp_w_online_sweep_post_teammate.py --cache_file data.npz

Cost: ~6 trials × ~10 min ≈ 60 min.
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
    ap.add_argument("--out", default="results/exp_w_online_sweep_post_teammate.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--values",
        default="0.30,0.50,0.70,0.85,1.0",
        help="comma-separated w_online values (deduped at runtime)",
    )
    args = ap.parse_args()

    # Deduplicate while preserving order.
    seen: set[float] = set()
    w_values: list[float] = []
    for x in args.values.split(","):
        v = float(x)
        if v not in seen:
            seen.add(v)
            w_values.append(v)

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)

    all_pids = {int(p) for p in maps.idx_to_player_id}
    case_ids = {k: v for k, v in CASE_STUDY.items() if k in all_pids}

    rows: list[dict] = []
    for w_online in w_values:
        print(f"\n=== w_online={w_online:.3f} (eta_teammate=0.02) ===", flush=True)
        cfg = Config(
            eta_teammate=0.02,
            w_online=w_online,
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
            "w_online": w_online,
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
                rows.append({
                    "w_online": w_online,
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
            f"  w_online={r['w_online']:.3f}  ll={r['logloss']:.4f}  "
            f"AUC={r['auc']}  {thetas}{marker}"
        )

    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
