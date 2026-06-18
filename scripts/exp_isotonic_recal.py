"""Post-hoc isotonic recalibration experiment (mid-range p ∈ [0.5, 0.8]).

Trains the production sequential model once with honest cell-holdout
(``holdout_obs_fraction=0.10``, ``holdout_seed=42``), then compares
logit-affine baseline predictions on the held-out eval set against
several isotonic corrections fitted *without* using the eval labels.

Variants (all evaluated on the same held-out cells unless noted):

* ``baseline_affine`` — production lapse + logit-affine recal on holdout
* ``isotonic_global_train`` — global isotonic fit on non-holdout preds,
  applied to holdout (Option A: train=90 % cells)
* ``isotonic_per_mode_train`` — six isotonic curves (mode × solo) on
  non-holdout, applied to holdout
* ``isotonic_global_nested`` — 50/50 split *within* holdout: fit on
  half, eval on other half (no label leakage for isotonic)
* ``isotonic_per_mode_nested`` — per-(mode × solo) nested split

Metrics: overall logloss, ECE (10 equal-width bins), mean bias
(actual − predicted) in [0.5, 0.6), [0.6, 0.7), [0.7, 0.8).

Leakage notes (printed and saved in CSV metadata row):

* Baseline model θ / b / lapse / affine recal were learned on the
  90 % non-holdout cells only — same as production backtest.
* ``*_train`` isotonic fits use non-holdout labels → zero leakage on
  the 10 % holdout eval set for the isotonic layer (but isotonic sees
  a much larger training set than affine recal had during SGD).
* ``*_nested`` isotonic fits use only half of the holdout → fully
  leakage-free for isotonic, but eval set is 5 % of all obs (noisier).

Usage::

    python -m scripts.exp_isotonic_recal --cache_file data.npz

Runtime ≈ one full training pass (~8–16 min on cache).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential

MID_BINS = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8))


def _bucket_type(g: str) -> str:
    if "async" in g:
        return "async"
    if "sync" in g:
        return "sync"
    return "offline"


def _mode_solo_channel(
    game_idx: np.ndarray,
    obs_idx: np.ndarray,
    maps,
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    """Integer channel 0..5 = (mode_idx, is_solo)."""
    gtype = getattr(maps, "game_type", None)
    team_sizes = arrays["team_sizes"]
    mode_idx = np.zeros(len(game_idx), dtype=np.int32)
    is_solo = np.zeros(len(game_idx), dtype=bool)
    for i, (gi, oi) in enumerate(zip(game_idx, obs_idx)):
        gt = str(gtype[int(gi)]) if gtype is not None else "offline"
        if "async" in gt:
            mode_idx[i] = 2
        elif "sync" in gt:
            mode_idx[i] = 1
        else:
            mode_idx[i] = 0
        is_solo[i] = int(team_sizes[int(oi)]) == 1
    return mode_idx * 2 + is_solo.astype(np.int32)


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bid = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bid == b
        if not m.any():
            continue
        ece += m.sum() * abs(float(p[m].mean()) - float(y[m].mean()))
    return ece / max(1, len(p))


def _mid_bin_bias(
    p: np.ndarray, y: np.ndarray, lo: float, hi: float
) -> tuple[float, int]:
    m = (p >= lo) & (p < hi)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float((y[m] - p[m]).mean()), n


def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
    iso.fit(p, y)
    return iso


def _apply_isotonic(iso: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return np.clip(iso.predict(p), 1e-15, 1.0 - 1e-15)


def _fit_per_channel(
    p: np.ndarray,
    y: np.ndarray,
    channel: np.ndarray,
    n_channels: int = 6,
    min_n: int = 500,
) -> dict[int, IsotonicRegression | None]:
    models: dict[int, IsotonicRegression | None] = {}
    for ch in range(n_channels):
        m = channel == ch
        if int(m.sum()) < min_n:
            models[ch] = None
            continue
        models[ch] = _fit_isotonic(p[m], y[m])
    return models


def _apply_per_channel(
    models: dict[int, IsotonicRegression | None],
    p: np.ndarray,
    channel: np.ndarray,
    fallback: IsotonicRegression | None = None,
) -> np.ndarray:
    out = p.copy()
    for ch, iso in models.items():
        m = channel == ch
        if not m.any():
            continue
        if iso is not None:
            out[m] = _apply_isotonic(iso, p[m])
        elif fallback is not None:
            out[m] = _apply_isotonic(fallback, p[m])
    return out


def _metrics_row(
    variant: str,
    slice_name: str,
    p: np.ndarray,
    y: np.ndarray,
    *,
    n_eval: int | None = None,
) -> dict:
    m = compute_metrics(p, y)
    row: dict = {
        "variant": variant,
        "slice": slice_name,
        "n": int(len(p)),
        "logloss": round(float(m["logloss"]), 6),
        "brier": round(float(m["brier"]), 6),
        "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
        "ece": round(_ece(p, y), 6),
    }
    if n_eval is not None:
        row["n_eval_total"] = n_eval
    for lo, hi in MID_BINS:
        bias, nb = _mid_bin_bias(p, y, lo, hi)
        key = f"bias_{lo:.1f}_{hi:.1f}"
        row[key] = round(bias, 6) if not np.isnan(bias) else ""
        row[f"n_{lo:.1f}_{hi:.1f}"] = nb
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_isotonic_recal.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--nested_seed", type=int, default=43,
        help="seed for 50/50 split within holdout (isotonic fit vs eval)",
    )
    ap.add_argument(
        "--min_channel_n", type=int, default=500,
        help="min obs to fit per-(mode×solo) isotonic curve",
    )
    args = ap.parse_args()

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)

    print(
        f"\n=== train production model (holdout={args.holdout}, "
        f"seed={args.seed}) ===",
        flush=True,
    )
    t0 = time.time()
    cfg = Config(holdout_obs_fraction=args.holdout, holdout_seed=args.seed)
    result = run_sequential(
        arrays, maps, cfg, verbose=False, collect_predictions=True,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s", flush=True)

    pred = result.predictions
    assert pred is not None
    p = pred["pred_p"]
    y = pred["actual_y"].astype(np.float64)
    obs_idx = pred["obs_idx"]
    game_idx = pred["game_idx"]
    is_ho = pred["is_holdout"].astype(bool)
    channel = _mode_solo_channel(game_idx, obs_idx, maps, arrays)

    train_m = ~is_ho
    eval_m = is_ho
    n_train, n_eval = int(train_m.sum()), int(eval_m.sum())
    print(
        f"[split] train (non-holdout)={n_train:,}  "
        f"eval (holdout)={n_eval:,}",
        flush=True,
    )

    p_train, y_train = p[train_m], y[train_m]
    p_eval, y_eval = p[eval_m], y[eval_m]
    ch_train, ch_eval = channel[train_m], channel[eval_m]

    # Nested 50/50 within holdout.
    ho_rng = np.random.default_rng(args.nested_seed)
    nested_fit = np.zeros(len(p), dtype=bool)
    ho_positions = np.where(eval_m)[0]
    nested_fit[ho_positions] = ho_rng.random(len(ho_positions)) < 0.5
    nested_eval = eval_m & ~nested_fit
    n_nested_fit = int(nested_fit.sum())
    n_nested_eval = int(nested_eval.sum())
    print(
        f"[nested] fit={n_nested_fit:,}  eval={n_nested_eval:,} "
        f"(within holdout, seed={args.nested_seed})",
        flush=True,
    )

    rows: list[dict] = []

    def _record(variant: str, p_out: np.ndarray, mask: np.ndarray) -> None:
        row = _metrics_row(
            variant, "all", p_out[mask], y[mask], n_eval=n_eval,
        )
        row["elapsed_sec"] = round(elapsed, 1)
        rows.append(row)
        mid = "  ".join(
            f"[{lo:.1f},{hi:.1f}) bias={row[f'bias_{lo:.1f}_{hi:.1f}']:+.4f}"
            for lo, hi in MID_BINS
            if row.get(f"n_{lo:.1f}_{hi:.1f}", 0)
        )
        print(
            f"  {variant:28s} n={row['n']:>9,}  "
            f"ll={row['logloss']:.4f}  ece={row['ece']:.4f}  {mid}",
            flush=True,
        )

    print("\n=== metrics on holdout eval (10 % cells) ===", flush=True)
    _record("baseline_affine", p, eval_m)

    iso_global = _fit_isotonic(p_train, y_train)
    p_iso_global = p.copy()
    p_iso_global[eval_m] = _apply_isotonic(iso_global, p_eval)
    _record("isotonic_global_train", p_iso_global, eval_m)

    iso_per_mode = _fit_per_channel(
        p_train, y_train, ch_train, min_n=args.min_channel_n,
    )
    p_iso_pm = p.copy()
    p_iso_pm[eval_m] = _apply_per_channel(
        iso_per_mode, p_eval, ch_eval, fallback=iso_global,
    )
    _record("isotonic_per_mode_train", p_iso_pm, eval_m)

    print("\n=== nested holdout (5 % cells, isotonic fit isolated) ===", flush=True)
    iso_nested = _fit_isotonic(p[nested_fit], y[nested_fit])
    p_iso_nested = p.copy()
    p_iso_nested[nested_eval] = _apply_isotonic(iso_nested, p[nested_eval])
    _record("baseline_affine", p, nested_eval)
    _record("isotonic_global_nested", p_iso_nested, nested_eval)

    iso_pm_nested = _fit_per_channel(
        p[nested_fit], y[nested_fit], channel[nested_fit],
        min_n=max(100, args.min_channel_n // 5),
    )
    p_iso_pm_nested = p.copy()
    p_iso_pm_nested[nested_eval] = _apply_per_channel(
        iso_pm_nested, p[nested_eval], channel[nested_eval],
        fallback=iso_nested,
    )
    _record("isotonic_per_mode_nested", p_iso_pm_nested, nested_eval)

    # Delta vs baseline on main holdout eval.
    base_ll = float(rows[0]["logloss"])
    for r in rows[1:]:
        if r["slice"] != "all":
            continue
        if r["variant"] == "baseline_affine" and r["n"] == n_nested_eval:
            continue
        r["delta_logloss_vs_baseline"] = round(
            float(r["logloss"]) - base_ll, 6,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant", "slice", "n", "logloss", "delta_logloss_vs_baseline",
        "brier", "auc", "ece",
        "bias_0.5_0.6", "n_0.5_0.6",
        "bias_0.6_0.7", "n_0.6_0.7",
        "bias_0.7_0.8", "n_0.7_0.8",
        "elapsed_sec", "n_eval_total",
    ]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    best = min(
        (r for r in rows if r["n"] == n_eval and r["variant"] != "baseline_affine"),
        key=lambda r: float(r["logloss"]),
    )
    d_ll = float(best["logloss"]) - base_ll
    print(
        f"\n[summary] best on holdout: {best['variant']}  "
        f"Δlogloss={d_ll:+.6f}  (baseline_affine ll={base_ll:.6f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
