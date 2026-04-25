"""Re-tune key learning rates with noisy_or_init=True.

The April 2026 defaults (eta0=0.07, w_sync=0.7, w_online_questions=0.30,
eta_size=eta_pos=0.005) were tuned with the legacy ``b = -log(p_take)``
initialisation.  After switching to noisy-OR-aware init, the structural
under-estimation of b on hard packs is gone, which in turn changes the
optimal step sizes:

  * gradients on b are smaller from the start (init is much closer to
    equilibrium), so question-update weights can probably go down;
  * the per-mode shift δ_size/δ_pos no longer needs to absorb the
    old init bias, so their learning rates / regularisation may need
    to shrink as well;
  * w_sync / w_online_questions affect how fast sync/async populations
    move θ — these were the two buckets where logloss got slightly
    worse with the new init at the old defaults.

Coordinate-descent sweep (5 params × 3 values = 15 single-knob
configs).  Each config writes one row to ``results/exp_noisy_or_init_retune.csv``.

Usage:
    .venv/bin/python scripts/exp_noisy_or_init_retune.py <config_name>
    .venv/bin/python scripts/exp_noisy_or_init_retune.py --list
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
OUT = os.path.join(ROOT, "results", "exp_noisy_or_init_retune.csv")


def base(**overrides) -> Config:
    """Defaults + noisy_or_init=True (already the new default)."""
    cfg = Config()
    cfg.noisy_or_init = True
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


CONFIGS = {
    # Anchor points: legacy init (for reference) and new init at current
    # defaults.  The retune below should beat (or at least match) the
    # noisy-OR baseline on overall logloss/AUC.
    "legacy_baseline":      base(noisy_or_init=False),
    "noisy_or_baseline":    base(),

    # eta0 sweep
    "eta0_0.04":            base(eta0=0.04),
    "eta0_0.05":            base(eta0=0.05),
    "eta0_0.09":            base(eta0=0.09),
    "eta0_0.12":            base(eta0=0.12),

    # w_online_questions sweep
    "woq_0.15":             base(w_online_questions=0.15),
    "woq_0.20":             base(w_online_questions=0.20),
    "woq_0.45":             base(w_online_questions=0.45),

    # w_sync sweep
    "wsync_0.5":            base(w_sync=0.5),
    "wsync_0.6":            base(w_sync=0.6),
    "wsync_0.8":            base(w_sync=0.8),
    "wsync_0.9":            base(w_sync=0.9),

    # eta_size sweep (with reg_size kept at 0.10)
    "etasize_0.001":        base(eta_size=0.001),
    "etasize_0.002":        base(eta_size=0.002),
    "etasize_0.010":        base(eta_size=0.010),

    # eta_pos sweep
    "etapos_0.001":         base(eta_pos=0.001),
    "etapos_0.002":         base(eta_pos=0.002),
    "etapos_0.010":         base(eta_pos=0.010),

    # ---- combo configs based on single-knob winners ----
    # eta0=0.04 and w_sync=0.5 were both monotonic best singles; the
    # other knobs were flat.  Try the obvious 2-way combo plus a few
    # extensions to check if there's headroom past the singles.
    "eta0_0.03":            base(eta0=0.03),
    "wsync_0.4":            base(w_sync=0.4),
    "combo_e04_w05":        base(eta0=0.04, w_sync=0.5),
    "combo_e03_w05":        base(eta0=0.03, w_sync=0.5),
    "combo_e04_w04":        base(eta0=0.04, w_sync=0.4),
    "combo_e04_w05_woq015": base(eta0=0.04, w_sync=0.5, w_online_questions=0.15),
    "combo_e04_w05_etas001_etap001": base(
        eta0=0.04, w_sync=0.5, eta_size=0.001, eta_pos=0.001,
    ),
    "combo_full":           base(
        eta0=0.04, w_sync=0.5, w_online_questions=0.15,
        eta_size=0.001, eta_pos=0.001,
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
