"""Compare cheap-to-implement model changes against the t6 baseline.

Runs ONE config per Python invocation to keep peak RSS bounded; the
caller (a small shell loop) iterates through CONFIG_NAMES and appends
each run's metrics to ``results/simple_experiments.csv``.

Usage:
    python scripts/run_simple_experiments.py <config_name>
    python scripts/run_simple_experiments.py --list
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import load_cached
from rating.backtest import backtest
from rating.engine import Config


CACHE = os.path.join(ROOT, "data.npz")
OUT = os.path.join(ROOT, "results", "simple_experiments.csv")


def base_cfg(**overrides) -> Config:
    """Start from current defaults (t6) and apply overrides."""
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


CONFIGS = {
    "baseline_t6": base_cfg(),

    # Single-knob ablations (small reg_theta values -- previous 0.01 was
    # already too strong because shrinkage is applied per observation)
    "reg_theta_0.0001": base_cfg(reg_theta=0.0001),
    "reg_theta_0.001":  base_cfg(reg_theta=0.001),

    "cold_init_0.5": base_cfg(cold_init_factor=0.5),

    "cal_decay_0.99":  base_cfg(use_calendar_decay=True, rho_calendar=0.99),
    "cal_decay_0.995": base_cfg(use_calendar_decay=True, rho_calendar=0.995),
    "cal_decay_0.997": base_cfg(use_calendar_decay=True, rho_calendar=0.997),
    "cal_decay_0.998": base_cfg(use_calendar_decay=True, rho_calendar=0.998),
    "cal_decay_0.999": base_cfg(use_calendar_decay=True, rho_calendar=0.999),
    "cal_decay_1.000": base_cfg(use_calendar_decay=True, rho_calendar=1.0),

    # Combined: pick the cheapest meaningful overlay (will be revised
    # based on single-knob results)
    "combo_v1": base_cfg(
        cold_init_factor=0.5,
        use_calendar_decay=True,
        rho_calendar=0.99,
    ),
}


FIELDNAMES = [
    "name", "logloss", "brier", "auc",
    "logloss_offline", "brier_offline", "auc_offline",
    "logloss_sync",    "brier_sync",    "auc_sync",
    "logloss_async",   "brier_async",   "auc_async",
    "n_test_obs", "n_test_games",
    "elapsed_sec",
    "config_json",
]


def _append_csv(row: dict) -> None:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    write_header = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_one(name: str, cfg: Config) -> None:
    print(f"Loading {CACHE} ...", flush=True)
    arrays, maps = load_cached(CACHE)
    print(
        f"Data: {len(arrays['q_idx'])} obs, "
        f"{maps.num_players} players, "
        f"{maps.num_questions} questions",
        flush=True,
    )
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    metrics = backtest(arrays, maps, cfg, verbose=False)
    dt = time.time() - t0
    by_type = metrics.get("by_type", {})

    row = {
        "name": name,
        "logloss": round(metrics["logloss"], 6),
        "brier": round(metrics["brier"], 6),
        "auc": round(metrics["auc"], 6),
        "n_test_obs": metrics.get("n_test_obs", 0),
        "n_test_games": metrics.get("n_test_games", 0),
        "elapsed_sec": round(dt, 1),
        "config_json": str(asdict(cfg)),
    }
    for t in ("offline", "sync", "async"):
        m = by_type.get(t, {})
        row[f"logloss_{t}"] = round(m.get("logloss", float("nan")), 6) if m else float("nan")
        row[f"brier_{t}"]   = round(m.get("brier", float("nan")), 6)   if m else float("nan")
        row[f"auc_{t}"]     = round(m.get("auc", float("nan")), 6)     if m else float("nan")
    _append_csv(row)
    print(
        f"  done in {dt:.1f}s  "
        f"logloss={metrics['logloss']:.4f}  "
        f"brier={metrics['brier']:.4f}  "
        f"auc={metrics['auc']:.4f}",
        flush=True,
    )
    for t in ("offline", "sync", "async"):
        m = by_type.get(t, {})
        if not m or m.get("n_obs", 0) == 0:
            continue
        print(
            f"    {t:7s} n={m['n_obs']:>9d} "
            f"logloss={m['logloss']:.4f} "
            f"brier={m['brier']:.4f} "
            f"auc={m['auc']:.4f}",
            flush=True,
        )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 1
    if sys.argv[1] == "--list":
        for n in CONFIGS:
            print(n)
        return 0
    name = sys.argv[1]
    if name not in CONFIGS:
        print(f"unknown config '{name}'.  Use --list to see options.")
        return 2
    run_one(name, CONFIGS[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
