#!/usr/bin/env python3
"""Quick sanity check after eta_teammate default bump to 0.02.

Runs one honest cell-holdout pass with production defaults and compares
θ / logloss to the values recorded in exp_eta_teammate_sweep_honest.csv
at eta_teammate=0.02.

Usage::

    python scripts/sanity_eta_teammate.py --cache_file data.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential

EXPECTED = {
    "logloss": 0.5018,
    "theta_Chernukha": -0.392,
    "theta_Rekshinskaya": -0.154,
    "theta_Monina": -0.192,
}
TOL = {"logloss": 0.002, "theta": 0.02}

CASE = {34909: "Chernukha", 26818: "Rekshinskaya", 158668: "Monina"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    args = ap.parse_args()

    cfg = Config(holdout_obs_fraction=0.10, holdout_seed=42)
    assert cfg.eta_teammate == 0.02, f"expected eta_teammate=0.02, got {cfg.eta_teammate}"

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)
    result = run_sequential(
        arrays, maps, cfg, verbose=False, collect_predictions=True
    )

    pred = result.predictions
    mask = pred["is_holdout"].astype(bool)
    m = compute_metrics(pred["pred_p"][mask], pred["actual_y"][mask])
    ll = float(m["logloss"])

    pid_to_idx = {int(pid): i for i, pid in enumerate(maps.idx_to_player_id)}
    thetas = {}
    for pid, name in CASE.items():
        idx = pid_to_idx.get(pid)
        thetas[name] = float(result.players.theta[idx]) if idx is not None else float("nan")

    ok = True
    print(f"\n=== Sanity check (eta_teammate=0.02) ===")
    print(f"  logloss={ll:.4f}  (expected {EXPECTED['logloss']:.4f})")
    if abs(ll - EXPECTED["logloss"]) > TOL["logloss"]:
        print(f"  [FAIL] logloss delta {ll - EXPECTED['logloss']:+.4f}")
        ok = False
    else:
        print(f"  [ok] logloss within ±{TOL['logloss']}")

    for name in ("Chernukha", "Rekshinskaya", "Monina"):
        got = thetas[name]
        exp = EXPECTED[f"theta_{name}"]
        print(f"  θ {name:12}={got:+.4f}  (expected {exp:+.3f})")
        if abs(got - exp) > TOL["theta"]:
            print(f"  [FAIL] delta {got - exp:+.4f}")
            ok = False

    if ok:
        print("\n[ok] all checks passed", flush=True)
        return 0
    print("\n[warn] some checks out of tolerance (may be float drift)", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
