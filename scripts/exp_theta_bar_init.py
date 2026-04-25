"""θ̄-aware question init: ablation backtest.

Two configs only — current defaults (which already have noisy_or_init
plus the 2026-04 retuned step sizes) with and without theta_bar_init.
Run sequentially; each appends one row to
``results/exp_theta_bar_init.csv``, including the new per-roster-
strength-quartile metrics.

Usage:
    .venv/bin/python scripts/exp_theta_bar_init.py <config_name>
    .venv/bin/python scripts/exp_theta_bar_init.py --list
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
OUT = os.path.join(ROOT, "results", "exp_theta_bar_init.csv")


def base(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


CONFIGS = {
    "no_theta_bar":  base(theta_bar_init=False),
    "theta_bar_g3":  base(theta_bar_init=True, theta_bar_min_games=3),
    # NOTE: pack-level shrinkage variants (`thbar_pack_*`,
    # `b_pack_shrinkage`, `pack_prior_w`) were tested in this slot and
    # all degraded logloss; the corresponding Config fields were
    # removed.  See docs/theta_bar_init_experiments.md for the full
    # account.
}


FIELDNAMES = [
    "name", "logloss", "brier", "auc",
    "logloss_offline", "brier_offline", "auc_offline",
    "logloss_sync",    "brier_sync",    "auc_sync",
    "logloss_async",   "brier_async",   "auc_async",
    "ll_q1", "ll_q2", "ll_q3", "ll_q4",
    "auc_q1", "auc_q2", "auc_q3", "auc_q4",
    "thbar_q1", "thbar_q2", "thbar_q3", "thbar_q4",
    "n_q1", "n_q2", "n_q3", "n_q4",
    "hardness_cuts",
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
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    metrics = backtest(arrays, maps, cfg, verbose=False)
    dt = time.time() - t0
    by_type = metrics.get("by_type", {})
    by_h = metrics.get("by_hardness", {})

    row = {
        "name": name,
        "logloss": round(metrics["logloss"], 6),
        "brier": round(metrics["brier"], 6),
        "auc": round(metrics["auc"], 6),
        "n_test_obs": metrics.get("n_test_obs", 0),
        "n_test_games": metrics.get("n_test_games", 0),
        "elapsed_sec": round(dt, 1),
        "config_json": str(asdict(cfg)),
        "hardness_cuts": " | ".join(
            f"{c:+.3f}" for c in metrics.get("hardness_cuts", [])
        ),
    }
    for t in ("offline", "sync", "async"):
        m = by_type.get(t, {})
        row[f"logloss_{t}"] = round(m.get("logloss", float("nan")), 6) if m else float("nan")
        row[f"brier_{t}"]   = round(m.get("brier", float("nan")), 6)   if m else float("nan")
        row[f"auc_{t}"]     = round(m.get("auc", float("nan")), 6)     if m else float("nan")
    for q in (1, 2, 3, 4):
        m = by_h.get(f"q{q}", {})
        row[f"ll_q{q}"]    = round(m.get("logloss", float("nan")), 6) if m else float("nan")
        row[f"auc_q{q}"]   = round(m.get("auc", float("nan")), 6)     if m else float("nan")
        row[f"thbar_q{q}"] = round(m.get("mean_team_theta", float("nan")), 4) if m else float("nan")
        row[f"n_q{q}"]     = m.get("n_obs", 0) if m else 0
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
            f"auc={m['auc']:.4f}",
            flush=True,
        )
    if by_h:
        print(f"    by hardness (cuts {row['hardness_cuts']}):")
        for q in (1, 2, 3, 4):
            m = by_h.get(f"q{q}", {})
            if not m or m.get("n_obs", 0) == 0:
                continue
            print(
                f"      q{q} (θ̄≈{m['mean_team_theta']:+.3f}): "
                f"n={m['n_obs']:>9d} games={m.get('n_games', 0):>4d}  "
                f"logloss={m['logloss']:.4f}  "
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
