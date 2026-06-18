"""Sweep minimum take-probability floor on honest cell-holdout.

Variants:
  baseline   — production (no floor)
  floor_005  — p_min = 0.5 %
  floor_010  — p_min = 1.0 %

Usage::

    python scripts/exp_take_floor.py --cache_file data.npz

Outputs ``results/exp_take_floor.csv`` and a console summary.
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


def _calibration_bias(p: np.ndarray, y: np.ndarray, lo: float, hi: float) -> dict:
    mask = (p >= lo) & (p < hi)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "mean_p": float("nan"), "mean_y": float("nan"), "bias": float("nan")}
    mp = float(p[mask].mean())
    my = float(y[mask].mean())
    return {"n": n, "mean_p": mp, "mean_y": my, "bias": my - mp}


def _build_configs(holdout: float, seed: int) -> dict[str, Config]:
    base = dict(holdout_obs_fraction=holdout, holdout_seed=seed)
    return {
        "baseline": Config(**base, use_take_floor=False),
        "floor_005": Config(**base, use_take_floor=True, take_floor_min=0.005),
        "floor_010": Config(**base, use_take_floor=True, take_floor_min=0.010),
    }


def _veteran_theta_deltas(
    arrays: dict,
    maps,
    holdout: float,
    seed: int,
) -> list[dict]:
    """θ shift for prolific veterans (≥500 games) vs baseline."""
    configs = _build_configs(holdout, seed)
    baseline = run_sequential(arrays, maps, configs["baseline"], verbose=False)
    rows: list[dict] = []
    # Motivating floor-player ids from docs/floor_player_experiments_2026-06.md
    # plus a few top-θ veterans for the "overperform at hard events" angle.
    watch_ids = [34909, 26818, 158668, 32919, 735, 1204]
    pid_to_idx = {int(pid): i for i, pid in enumerate(maps.idx_to_player_id)}
    for name, cfg in configs.items():
        if name == "baseline":
            continue
        res = run_sequential(arrays, maps, cfg, verbose=False)
        for pid in watch_ids:
            pidx = pid_to_idx.get(pid)
            if pidx is None:
                continue
            rows.append({
                "variant": name,
                "player_id": pid,
                "theta_baseline": float(baseline.players.theta[pidx]),
                "theta_variant": float(res.players.theta[pidx]),
                "delta_theta": float(
                    res.players.theta[pidx] - baseline.players.theta[pidx]
                ),
                "games": int(res.players.games[pidx]),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_take_floor.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-veteran", action="store_true",
        help="Skip extra veteran θ diagnostic (saves 2 training passes)",
    )
    args = ap.parse_args()

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)

    configs = _build_configs(args.holdout, args.seed)
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    pred_by_variant: dict[str, dict] = {}

    for name, cfg in configs.items():
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        result = run_sequential(
            arrays, maps, cfg, verbose=False, collect_predictions=True
        )
        elapsed = time.time() - t0
        pred = result.predictions
        assert pred is not None
        mask = pred["is_holdout"].astype(bool)
        p = pred["pred_p"][mask]
        y = pred["actual_y"][mask]
        g = pred["game_idx"][mask]
        obs_idx = pred.get("obs_idx")
        pred_by_variant[name] = {"p": p, "y": y, "g": g, "obs_idx": obs_idx}

        m = compute_metrics(p, y)
        row = {
            "variant": name,
            "slice": "all",
            "n": int(mask.sum()),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
            "elapsed_sec": round(elapsed, 1),
        }
        rows.append(row)
        summaries[name] = {"all": row}
        print(
            f"  overall : n={row['n']:>9d}  ll={row['logloss']:.4f}  "
            f"brier={row['brier']:.4f}  AUC={row['auc']}  "
            f"({row['elapsed_sec']:.1f}s)",
            flush=True,
        )

        # Mode slices.
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
                r2 = {
                    "variant": name,
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
                rows.append(r2)
                summaries[name][t] = r2

        # Team-size slices (from raw obs indices).
        if obs_idx is not None:
            ho_obs = np.asarray(obs_idx)[mask]
            sizes = arrays["team_sizes"][ho_obs]
            for label, lo, hi in (
                ("size1", 1, 1),
                ("size2", 2, 2),
                ("size3_5", 3, 5),
                ("size6p", 6, 99),
            ):
                sm = (sizes >= lo) & (sizes <= hi)
                if not sm.any():
                    continue
                ms = compute_metrics(p[sm], y[sm])
                r2 = {
                    "variant": name,
                    "slice": label,
                    "n": int(sm.sum()),
                    "logloss": round(float(ms["logloss"]), 6),
                    "brier": round(float(ms["brier"]), 6),
                    "auc": (
                        round(float(ms["auc"]), 6)
                        if not np.isnan(ms["auc"]) else ""
                    ),
                    "elapsed_sec": "",
                }
                rows.append(r2)

        # Question difficulty quartile by predicted p (low p = hard).
        cuts = np.unique(np.quantile(p, [0.25, 0.5, 0.75]))
        q_labels = np.searchsorted(cuts, p, side="right") + 1
        for q in (1, 2, 3, 4):
            qm = q_labels == q
            if not qm.any():
                continue
            ms = compute_metrics(p[qm], y[qm])
            rows.append({
                "variant": name,
                "slice": f"pq{q}",
                "n": int(qm.sum()),
                "logloss": round(float(ms["logloss"]), 6),
                "brier": round(float(ms["brier"]), 6),
                "auc": (
                    round(float(ms["auc"]), 6)
                    if not np.isnan(ms["auc"]) else ""
                ),
                "elapsed_sec": "",
            })

        # Calibration tails.
        for label, lo, hi in (
            ("cal_p_lt_05", 0.0, 0.05),
            ("cal_p_05_10", 0.05, 0.10),
            ("cal_p_90_100", 0.90, 1.01),
        ):
            cb = _calibration_bias(p, y, lo, hi)
            rows.append({
                "variant": name,
                "slice": label,
                "n": cb["n"],
                "logloss": "",
                "brier": round(cb["bias"], 6) if cb["n"] else "",
                "auc": "",
                "elapsed_sec": "",
            })
            if cb["n"]:
                print(
                    f"  {label:12s}: n={cb['n']:>7d}  "
                    f"bias(actual-p)={cb['bias']:+.4f}",
                    flush=True,
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "variant", "slice", "n", "logloss", "brier", "auc", "elapsed_sec",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)

    if "baseline" in summaries:
        print("\n=== Δ logloss vs baseline (positive = worse) ===", flush=True)
        base = summaries["baseline"]
        for name, by_slice in summaries.items():
            if name == "baseline":
                continue
            for sl in ("all", "offline", "sync", "async", "size1", "pq1", "pq4"):
                if sl not in by_slice and sl not in summaries[name]:
                    continue
                r = summaries[name].get(sl)
                br = base.get(sl)
                if not r or not br or r.get("logloss") in ("", None):
                    continue
                d = float(r["logloss"]) - float(br["logloss"])
                print(f"  {name:10s} {sl:8s}: Δll={d:+.4f}", flush=True)

    if not args.skip_veteran:
        print("\n=== Veteran θ deltas vs baseline ===", flush=True)
        vrows = _veteran_theta_deltas(arrays, maps, args.holdout, args.seed)
        for r in vrows:
            print(
                f"  {r['variant']:10s} pid={r['player_id']:>6d}  "
                f"θ {r['theta_baseline']:+.3f} → {r['theta_variant']:+.3f}  "
                f"Δ={r['delta_theta']:+.4f}  games={r['games']}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
