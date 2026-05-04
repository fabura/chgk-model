"""Cell-holdout evaluation: leakage-free vs leaky logloss comparison.

Runs two independent training passes:

* **Leaky** baseline: ``Config()`` with ``holdout_obs_fraction=0`` —
  the current default.  Predictions are recorded for every observation
  but the model has seen each (team, question) cell during init and
  SGD, so logloss on tail tournaments includes test-set leakage via
  the question initialisation.

* **Clean** holdout: ``Config(holdout_obs_fraction=H)`` —
  randomly drops a fraction of observations from question/player init
  and from SGD updates.  Predictions on the held-out subset are
  genuinely out-of-sample with respect to those (team, question)
  cells.

Both runs use the same random seed for the holdout mask, so we can
compare the same set of (team, question) cells under the two
methodologies and obtain a clean estimate of the leakage magnitude.

Outputs:

* ``results/exp_cell_holdout.csv`` — overall and per-type metrics
  for both configurations on the held-out subset.

Usage::

    python -m scripts.exp_cell_holdout --cache_file data.npz \
        --holdout_fraction 0.10 --seed 42

Total runtime ≈ 2 × backtest pass ≈ 16 min on cache.
"""
from __future__ import annotations

import argparse
import csv
import json
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
    ap.add_argument(
        "--out", default="results/exp_cell_holdout.csv",
        help="output CSV path",
    )
    ap.add_argument(
        "--holdout_fraction", type=float, default=0.10,
        help="fraction of observations to hold out (default 0.10)",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="random seed for the holdout mask (same across runs)",
    )
    args = ap.parse_args()

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)

    # The held-out mask is computed inside run_sequential when
    # holdout_obs_fraction > 0; we mirror the exact computation here
    # so the leaky baseline can be evaluated on the same obs set.
    n_obs = len(arrays["q_idx"])
    rng = np.random.default_rng(args.seed)
    is_holdout = rng.random(n_obs) < args.holdout_fraction
    print(
        f"[holdout] {int(is_holdout.sum())}/{n_obs} obs marked "
        f"(seed={args.seed})"
    )

    # ============================================================
    # Run 1: leaky baseline (full training data, no holdout from
    # train).  We record predictions for ALL obs and later select
    # those flagged in `is_holdout`.
    # ============================================================
    print("\n=== run 1: leaky baseline (no holdout) ===")
    t0 = time.time()
    cfg_leaky = Config()  # holdout_obs_fraction default = 0.0
    res_leaky = run_sequential(
        arrays, maps, cfg_leaky,
        verbose=False, collect_predictions=True,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    pred_l = res_leaky.predictions
    # Order is deterministic: predictions are recorded in tournament
    # order, but obs_idx tells us which raw obs each row belongs to.
    obs_l = pred_l["obs_idx"]
    # For each obs_idx row, look up is_holdout[obs_idx] to filter to
    # the same set of cells the holdout run will evaluate.
    leaky_mask = is_holdout[obs_l]
    p_l = pred_l["pred_p"][leaky_mask]
    y_l = pred_l["actual_y"][leaky_mask]
    g_l = pred_l["game_idx"][leaky_mask]

    # ============================================================
    # Run 2: clean holdout (same seed → same masked obs).
    # ============================================================
    print(f"\n=== run 2: clean holdout (h={args.holdout_fraction}, seed={args.seed}) ===")
    t0 = time.time()
    cfg_clean = Config(
        holdout_obs_fraction=args.holdout_fraction,
        holdout_seed=args.seed,
    )
    res_clean = run_sequential(
        arrays, maps, cfg_clean,
        verbose=False, collect_predictions=True,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    pred_c = res_clean.predictions
    obs_c = pred_c["obs_idx"]
    # Filter to obs flagged as held-out by the engine.
    if "is_holdout" in pred_c:
        clean_mask = pred_c["is_holdout"].astype(bool)
    else:
        clean_mask = is_holdout[obs_c]
    p_c = pred_c["pred_p"][clean_mask]
    y_c = pred_c["actual_y"][clean_mask]
    g_c = pred_c["game_idx"][clean_mask]

    # Sanity: same obs sets?
    obs_set_l = set(obs_l[leaky_mask].tolist())
    obs_set_c = set(obs_c[clean_mask].tolist())
    if obs_set_l != obs_set_c:
        print(
            "WARN: leaky/clean obs sets differ "
            f"(leaky={len(obs_set_l)}, clean={len(obs_set_c)}, "
            f"intersection={len(obs_set_l & obs_set_c)})"
        )
    else:
        print(
            f"[ok] both runs evaluated on the same {len(obs_set_l)} held-out obs"
        )

    # ============================================================
    # Metrics
    # ============================================================
    rows: list[dict] = []

    def _add(label: str, slice_value: str, p, y) -> dict:
        m = compute_metrics(np.asarray(p), np.asarray(y))
        row = {
            "config": label,
            "slice": slice_value,
            "n": int(len(p)),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
        }
        rows.append(row)
        return row

    print("\n=== Overall (held-out obs only) ===")
    r_leaky = _add("leaky_baseline", "all", p_l, y_l)
    r_clean = _add("clean_holdout", "all", p_c, y_c)
    print(
        f"  leaky_baseline : n={r_leaky['n']:>9d}  "
        f"logloss={r_leaky['logloss']:.4f}  brier={r_leaky['brier']:.4f}  AUC={r_leaky['auc']}"
    )
    print(
        f"  clean_holdout  : n={r_clean['n']:>9d}  "
        f"logloss={r_clean['logloss']:.4f}  brier={r_clean['brier']:.4f}  AUC={r_clean['auc']}"
    )
    delta = float(r_clean['logloss']) - float(r_leaky['logloss'])
    pct = 100.0 * delta / float(r_leaky['logloss'])
    print(
        f"  Δ logloss (clean − leaky) : {delta:+.4f}  "
        f"({pct:+.2f}%)  ← magnitude of test-set leakage"
    )

    # By tournament type.
    gtype = getattr(maps, "game_type", None)
    if gtype is not None:
        print("\n=== By tournament type (held-out obs only) ===")
        types_l = np.array(
            [_bucket_type(str(gtype[gi])) for gi in g_l], dtype=object
        )
        types_c = np.array(
            [_bucket_type(str(gtype[gi])) for gi in g_c], dtype=object
        )
        for t in ("offline", "sync", "async"):
            mask_l = types_l == t
            mask_c = types_c == t
            if mask_l.any() and mask_c.any():
                rl = _add("leaky_baseline", t, p_l[mask_l], y_l[mask_l])
                rc = _add("clean_holdout", t, p_c[mask_c], y_c[mask_c])
                d = float(rc['logloss']) - float(rl['logloss'])
                print(
                    f"  {t:7s}  leaky n={rl['n']:>9d} ll={rl['logloss']:.4f}  "
                    f"clean n={rc['n']:>9d} ll={rc['logloss']:.4f}  "
                    f"Δ={d:+.4f}"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["config", "slice", "n", "logloss", "brier", "auc"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] {len(rows)} rows → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
