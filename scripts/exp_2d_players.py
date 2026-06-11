"""Sweep 2D player model (Model C: per-player difficulty slope γ_k).

Tests use_2d_players with various eta_gamma values under honest cell-holdout.
Also compares freeze_log_a=True vs False.

Outputs ``results/exp_2d_players.csv``.

Usage::

    python scripts/exp_2d_players.py --cache_file data.npz

Cost: ~8 trials × ~10 min ≈ 80 min.
"""
from __future__ import annotations

import argparse, csv, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential

CASE_STUDY = {34909: "Chernukha", 26818: "Rekshinskaya", 158668: "Monina"}

def _bucket_type(g: str) -> str:
    if "async" in g: return "async"
    if "sync" in g: return "sync"
    return "offline"

def _player_thetas(result, maps, pids):
    out = {}
    pid_to_idx = {int(p): i for i, p in enumerate(maps.idx_to_player_id)}
    for db_id, label in pids.items():
        idx = pid_to_idx.get(int(db_id))
        out[f"theta_{label}"] = round(float(result.players.theta[idx]), 4) if idx is not None else float("nan")
        out[f"gamma_{label}"] = round(float(result.players.gamma[idx]), 4) if idx is not None else float("nan")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_2d_players.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    configs = [
        # Baseline: 1D model, freeze_log_a=True (a ≡ 1)
        ("baseline_frzA",  False, 0.0, True),
        # Model C with different eta_gamma values (comparable to eta0=0.22)
        ("2d_eta010",    True, 0.010, True),
        ("2d_eta025",    True, 0.025, True),
        ("2d_eta050",    True, 0.050, True),
        ("2d_eta100",    True, 0.100, True),
        ("2d_eta200",    True, 0.200, True),
        # Baseline with freeze_log_a=False (for comparison)
        ("baseline_frzA0",  False, 0.0, False),
        # Model C with freeze_log_a=False (learn a_i + gamma)
        ("2d_eta050_frzA0", True, 0.050, False),
    ]

    arrays, maps = load_cached(args.cache_file)
    all_pids = {int(p) for p in maps.idx_to_player_id}
    case_ids = {k: v for k, v in CASE_STUDY.items() if k in all_pids}

    rows = []
    for label, use_2d, eta_g, freeze_a in configs:
        print(f"\n=== {label} (2d={use_2d}, eta_g={eta_g}, freeze_a={freeze_a}) ===", flush=True)
        cfg = Config(
            use_2d_players=use_2d,
            eta_gamma=eta_g,
            freeze_log_a=freeze_a,
            holdout_obs_fraction=args.holdout,
            holdout_seed=args.seed,
        )
        t0 = time.time()
        result = run_sequential(arrays, maps, cfg, verbose=False, collect_predictions=True)
        elapsed = time.time() - t0

        pred = result.predictions
        mask = pred["is_holdout"].astype(bool)
        p, y, g = pred["pred_p"][mask], pred["actual_y"][mask], pred["game_idx"][mask]
        m = compute_metrics(p, y)

        # Gamma statistics
        gamma_vals = result.players.gamma
        gamma_active = gamma_vals[result.players.seen]

        row = {"config": label, "use_2d": use_2d, "eta_gamma": eta_g,
               "freeze_log_a": freeze_a, "slice": "all",
               "n": int(mask.sum()),
               "logloss": round(float(m["logloss"]), 6),
               "brier": round(float(m["brier"]), 6),
               "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
               "gamma_mean": round(float(gamma_active.mean()), 6) if len(gamma_active) > 0 else 0.0,
               "gamma_std": round(float(gamma_active.std()), 6) if len(gamma_active) > 0 else 0.0,
               "gamma_min": round(float(gamma_active.min()), 6) if len(gamma_active) > 0 else 0.0,
               "gamma_max": round(float(gamma_active.max()), 6) if len(gamma_active) > 0 else 0.0,
               "elapsed_sec": round(elapsed, 1)}
        row.update(_player_thetas(result, maps, case_ids))
        rows.append(row)
        print(f"  ll={row['logloss']:.4f} AUC={row['auc']} γ_mean={row['gamma_mean']:.4f} γ_std={row['gamma_std']:.4f} ({elapsed:.0f}s)", flush=True)
        for lbl in CASE_STUDY.values():
            for param in ("theta", "gamma"):
                k = f"{param}_{lbl}"
                if k in row: print(f"  {k}={row[k]:+.4f}", flush=True)

        gtype = getattr(maps, "game_type", None)
        if gtype is not None:
            types = np.array([_bucket_type(str(gtype[gi])) for gi in g], dtype=object)
            for t in ("offline", "sync", "async"):
                tm = types == t
                if not tm.any(): continue
                ms = compute_metrics(p[tm], y[tm])
                sr = {"config": label, "use_2d": use_2d, "eta_gamma": eta_g,
                      "freeze_log_a": freeze_a, "slice": t,
                      "n": int(tm.sum()),
                      "logloss": round(float(ms["logloss"]), 6),
                      "auc": round(float(ms["auc"]), 6) if not np.isnan(ms["auc"]) else ""}
                rows.append(sr)
                print(f"  {t:7s}: ll={ms['logloss']:.4f} AUC={ms['auc']:.4f}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fns = sorted({k for r in rows for k in r})
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fns, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    overall = [r for r in rows if r["slice"] == "all"]
    overall.sort(key=lambda r: r["logloss"])
    print("\n=== Ranked overall (best → worst) ===")
    for r in overall:
        marker = "  ★" if r is overall[0] else ""
        ts = "  ".join(f"{k}={r[k]:+.3f}" for k in sorted(r) if (k.startswith("theta_") or k.startswith("gamma_")) and r[k] != "")
        print(f"  {r['config']:20s} ll={r['logloss']:.4f} AUC={r['auc']} {ts}{marker}")
    print(f"\n[ok] → {out_path}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
