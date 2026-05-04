"""Re-validate the noisy-OR / θ̄-init / a_i ablations on a leakage-
free per-cell hold-out.

Each variant is trained from scratch with ``holdout_obs_fraction=0.10``
and ``holdout_seed=42`` (same mask across runs, so the metric is
computed on the same set of (team, question) cells for every config).

Variants:

* ``current``        — current production defaults
* ``no_theta_bar``   — drop θ̄-aware ``b_init`` (Round 2 ablation)
* ``no_noisy_or``    — also drop noisy-OR ``b_init`` → legacy
                       ``b = -log(p_take)``
* ``a_const1``       — freeze ``log_a = 0`` (i.e. ``a_i ≡ 1``);
                       no learned discrimination
* ``a_strong_prior`` — keep learning ``log_a`` but pull it toward 0
                       with ``reg_log_a=1.0``

Outputs:

* ``results/exp_holdout_ablations.csv`` — overall + per-type
  metrics for every variant on the held-out subset
* console summary: deltas vs ``current`` for quick reading

Usage::

    python -m scripts.exp_holdout_ablations --cache_file data.npz \
        --holdout 0.10 --seed 42

Total runtime ≈ 5 × backtest pass ≈ 40 min on cache.
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


def _build_configs(holdout: float, seed: int) -> dict[str, Config]:
    base = dict(holdout_obs_fraction=holdout, holdout_seed=seed)
    return {
        "current": Config(**base),
        "no_theta_bar": Config(**base, theta_bar_init=False),
        "no_noisy_or": Config(
            **base, theta_bar_init=False, noisy_or_init=False
        ),
        "a_const1": Config(**base, freeze_log_a=True),
        "a_strong_prior": Config(**base, reg_log_a=1.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument(
        "--out", default="results/exp_holdout_ablations.csv",
    )
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--variants", type=str, default=None,
        help="comma-separated subset of variant names; default = all",
    )
    args = ap.parse_args()

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)
    n_obs = len(arrays["q_idx"])
    rng = np.random.default_rng(args.seed)
    is_holdout = rng.random(n_obs) < args.holdout
    print(
        f"[holdout] {int(is_holdout.sum())}/{n_obs} obs marked "
        f"(seed={args.seed})", flush=True,
    )

    configs = _build_configs(args.holdout, args.seed)
    if args.variants:
        wanted = [v.strip() for v in args.variants.split(",")]
        configs = {k: v for k, v in configs.items() if k in wanted}
    print(f"[plan] running {len(configs)} variants: {list(configs)}", flush=True)

    rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for name, cfg in configs.items():
        print(f"\n=== {name} ===", flush=True)
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
                print(
                    f"  {t:7s}: n={r2['n']:>9d}  ll={r2['logloss']:.4f}  "
                    f"AUC={r2['auc']}",
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

    # Δ vs current.
    if "current" in summaries:
        print("\n=== Δ vs current (positive = worse) ===", flush=True)
        base = summaries["current"]
        print(
            f"{'variant':>16s}  {'slice':>7s}  "
            f"{'logloss':>9s}  {'Δll':>9s}  {'AUC':>7s}  {'Δauc':>8s}",
            flush=True,
        )
        for name, by_slice in summaries.items():
            for sl, r in by_slice.items():
                if sl not in base:
                    continue
                br = base[sl]
                d_ll = float(r["logloss"]) - float(br["logloss"])
                if r.get("auc") in ("", None) or br.get("auc") in ("", None):
                    d_auc_str = ""
                    auc_str = ""
                else:
                    d_auc = float(r["auc"]) - float(br["auc"])
                    d_auc_str = f"{d_auc:+.4f}"
                    auc_str = f"{r['auc']:.4f}"
                print(
                    f"{name:>16s}  {sl:>7s}  "
                    f"{r['logloss']:>9.4f}  {d_ll:>+9.4f}  "
                    f"{auc_str:>7s}  {d_auc_str:>+8s}",
                    flush=True,
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
