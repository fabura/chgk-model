#!/usr/bin/env python3
"""Summarise (b, log_a) geometry from a --results_npz trained with --no-freeze-log-a.

Prints correlation, PCA variance share on PC1 (standardised plane), linear residual
stats, and writes CSVs with extremes for manual inspection.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Repo root on PYTHONPATH when run as ``python scripts/...`` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_cached
from rating.io import load_results_npz


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--npz",
        type=Path,
        default=Path("results/seq_log_a.npz"),
        help="Rating results from training with --no-freeze-log-a",
    )
    ap.add_argument(
        "--cache-file",
        type=Path,
        default=Path("data.npz"),
        help="Observations cache used to count n_obs / n_taken per canonical question",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/log_a_diag"),
        help="Directory for CSV outputs",
    )
    args = ap.parse_args()

    if not args.npz.is_file():
        print(f"Missing {args.npz}", file=sys.stderr)
        return 1

    r = load_results_npz(args.npz)
    b_all = np.asarray(r.b, dtype=np.float64)
    a_all = np.asarray(r.a, dtype=np.float64)
    n_can = len(b_all)

    cq = r.canonical_q_idx
    rep_tid = -np.ones(n_can, dtype=np.int32)
    rep_qi = -np.ones(n_can, dtype=np.int32)
    if cq is None:
        rep_tid = np.asarray(r.question_tid, dtype=np.int32)[:n_can]
        rep_qi = np.asarray(r.question_qi, dtype=np.int32)[:n_can]
    else:
        qt = np.asarray(r.question_tid, dtype=np.int32)
        qq = np.asarray(r.question_qi, dtype=np.int32)
        for ri in range(int(cq.shape[0])):
            ci = int(cq[ri])
            if not (0 <= ci < n_can):
                continue
            if rep_tid[ci] < 0:
                rep_tid[ci] = int(qt[ri])
                rep_qi[ci] = int(qq[ri])

    arrays, maps = load_cached(str(args.cache_file))
    q_raw = np.asarray(arrays["q_idx"], dtype=np.int64)
    taken = np.asarray(arrays["taken"], dtype=np.float64)
    raw_to_can = np.asarray(maps.canonical_q_idx, dtype=np.int64) if getattr(maps, "canonical_q_idx", None) is not None else q_raw
    q_can = raw_to_can[q_raw]
    n_obs_can = np.bincount(q_can, minlength=n_can).astype(np.int64)
    n_taken_can = np.bincount(q_can, weights=taken, minlength=n_can).astype(np.int64)

    finite = np.isfinite(b_all) & np.isfinite(a_all) & (a_all > 0)
    has_obs = n_obs_can > 0
    nondegen = (n_taken_can > 0) & (n_taken_can < n_obs_can)
    n_grobs = int(np.sum(has_obs & (n_taken_can == 0)))
    n_buttons = int(np.sum(has_obs & (n_taken_can == n_obs_can) & (n_obs_can > 0)))
    print(
        f"canonical questions: {n_can}, with obs: {int(has_obs.sum())}, "
        f"гробы (0 takes): {n_grobs}, кнопки (100% takes): {n_buttons}"
    )

    ok = finite & has_obs & nondegen
    b = b_all[ok]
    a = a_all[ok]
    log_a = np.log(a)
    n = len(b)

    print(f"non-degenerate canonical: {n}")
    print(f"a: min={a.min():.4f} median={np.median(a):.4f} max={a.max():.4f}")
    print(f"b: min={b.min():.4f} median={np.median(b):.4f} max={b.max():.4f}")

    corr = float(np.corrcoef(b, log_a)[0, 1]) if n > 2 else float("nan")
    print(f"corr(b, log_a) = {corr:.4f}")

    Z = np.column_stack([b, log_a])
    Zs = (Z - Z.mean(axis=0)) / (Z.std(axis=0, ddof=0) + 1e-15)
    cov = np.cov(Zs.T, ddof=0)
    evals, _ = np.linalg.eigh(cov)
    evals = np.sort(evals)
    frac_pc1 = float(evals[-1] / (evals.sum() + 1e-15))
    print(f"PCA (standardised): frac variance on PC1 = {frac_pc1:.4f}")

    coef = np.polyfit(b, log_a, 1)
    slope, intercept = float(coef[0]), float(coef[1])
    pred = slope * b + intercept
    resid = log_a - pred
    print(
        f"OLS log_a ~ b: slope={slope:.6f} intercept={intercept:.6f} "
        f"R²={1.0 - np.var(resid) / (np.var(log_a) + 1e-15):.4f}"
    )
    print(
        f"residual (log_a pred): std={float(np.std(resid)):.6f} "
        f"p99 |r|={float(np.quantile(np.abs(resid), 0.99)):.6f}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    idx_ok = np.where(ok)[0]
    reps_tid = rep_tid[idx_ok]
    reps_qi = rep_qi[idx_ok]
    n_obs_sel = n_obs_can[idx_ok]
    n_taken_sel = n_taken_can[idx_ok]
    order = np.argsort(resid)

    header = [
        "canonical_idx", "tournament_id", "q_in_tournament",
        "n_obs", "n_taken", "p_take", "b", "a", "log_a", "residual",
    ]

    def write_side(name: str, sel: np.ndarray) -> None:
        path = args.out_dir / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for j in sel:
                ci = int(idx_ok[j])
                p_t = float(n_taken_sel[j]) / float(n_obs_sel[j])
                w.writerow(
                    [
                        ci,
                        int(reps_tid[j]) if reps_tid[j] >= 0 else "",
                        int(reps_qi[j]) if reps_qi[j] >= 0 else "",
                        int(n_obs_sel[j]),
                        int(n_taken_sel[j]),
                        round(p_t, 6),
                        round(float(b[j]), 6),
                        round(float(a[j]), 6),
                        round(float(log_a[j]), 6),
                        round(float(resid[j]), 6),
                    ]
                )
        print(f"Wrote {path}")

    k = min(30, n)
    # residual = log_a - OLS(b); negative ⇒ flatter a than typical at this b.
    write_side("flatter_than_ols_trend_top30.csv", order[:k])
    write_side("steeper_than_ols_trend_top30.csv", order[-k:][::-1])

    full = args.out_dir / "all_canonical.csv"
    with full.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        order_all = np.argsort(np.abs(resid))[::-1]
        for j in order_all:
            ci = int(idx_ok[j])
            p_t = float(n_taken_sel[j]) / float(n_obs_sel[j])
            w.writerow(
                [
                    ci,
                    int(reps_tid[j]) if reps_tid[j] >= 0 else "",
                    int(reps_qi[j]) if reps_qi[j] >= 0 else "",
                    int(n_obs_sel[j]),
                    int(n_taken_sel[j]),
                    round(p_t, 6),
                    round(float(b[j]), 6),
                    round(float(a[j]), 6),
                    round(float(log_a[j]), 6),
                    round(float(resid[j]), 6),
                ]
            )
    print(f"Wrote {full} (sorted by |residual| desc)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
