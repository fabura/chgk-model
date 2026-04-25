"""Re-tune key learning rates with theta_bar_init=True.

After adding the θ̄-aware init (logloss 0.5182 → 0.4950 untuned),
the b values start much closer to equilibrium for ALL packs (not
just hard ones), so the previous retune defaults — which were chosen
for the noisy_or_init-only world — may now be too aggressive again.

Sweep: 5-knob coord descent on the same 20 % time-split hold-out.
After single-knob, the top combos are tested.

Usage:
    .venv/bin/python scripts/exp_theta_bar_retune.py <config_name>
    .venv/bin/python scripts/exp_theta_bar_retune.py --list
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
OUT = os.path.join(ROOT, "results", "exp_theta_bar_retune.csv")


def base(**overrides) -> Config:
    cfg = Config()
    cfg.theta_bar_init = True
    cfg.theta_bar_min_games = 3
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


CONFIGS = {
    "anchor_thbar":         base(),  # current defaults + theta_bar

    # eta0 sweep
    "eta0_0.02":            base(eta0=0.02),
    "eta0_0.03":            base(eta0=0.03),
    "eta0_0.05":            base(eta0=0.05),
    "eta0_0.07":            base(eta0=0.07),

    # w_sync sweep
    "wsync_0.3":            base(w_sync=0.3),
    "wsync_0.4":            base(w_sync=0.4),
    "wsync_0.6":            base(w_sync=0.6),
    "wsync_0.7":            base(w_sync=0.7),

    # w_online_questions sweep — guess: theta_bar already gets b right,
    # so async noise should weigh less.
    "woq_0.05":             base(w_online_questions=0.05),
    "woq_0.10":             base(w_online_questions=0.10),
    "woq_0.30":             base(w_online_questions=0.30),

    # eta_size / eta_pos — quick check around current 0.001
    "etasize_0.0005":       base(eta_size=0.0005),
    "etasize_0.002":        base(eta_size=0.002),
    "etapos_0.0005":        base(eta_pos=0.0005),
    "etapos_0.002":         base(eta_pos=0.002),

    # ---- Round 2: extend eta0 + w_sync upward (initial sweep was monotone) ----
    "eta0_0.10":            base(eta0=0.10),
    "eta0_0.15":            base(eta0=0.15),
    "eta0_0.20":            base(eta0=0.20),
    "eta0_0.25":            base(eta0=0.25),
    "wsync_0.85":           base(w_sync=0.85),
    "wsync_1.0":            base(w_sync=1.0),

    # ---- Round 3: combos of the 2 best knobs ----
    "combo_eta0.07_ws0.7":  base(eta0=0.07, w_sync=0.7),
    "combo_eta0.10_ws0.7":  base(eta0=0.10, w_sync=0.7),
    "combo_eta0.10_ws0.85": base(eta0=0.10, w_sync=0.85),
    "combo_eta0.15_ws0.85": base(eta0=0.15, w_sync=0.85),
    "combo_eta0.15_ws1.0":  base(eta0=0.15, w_sync=1.0),
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
