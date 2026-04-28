"""Upper-bound on what δ_size re-tuning can buy us.

Read the cached test predictions
(``results/test_predictions.npz``), then apply two oracle
calibrations to compare:

  (A) ``δ[size]``         — best-possible single shift per team size
  (B) ``δ[size, mode]``   — best-possible shift per (size, mode) cell

Each shift is the ML-fit logit offset that, applied uniformly inside
the cell, gives the observed take rate.  Equivalent to logistic
regression with cell-indicator features only.

Reports baseline / (A) / (B) logloss + how (A)/(B) compare.

If (A) ≈ baseline_recoverable and (B) ≈ (A): mode-independent →
just re-tune the global δ_size online.
If (B) ≪ (A): need ``δ_size[mode][size]``.

Caveat: this is over-optimistic (in-sample on the same test set we
calibrate on).  But the gap (B − A) is unbiased about whether mode
matters, since each cell is independently fit.
"""
from __future__ import annotations

import argparse

import numpy as np

from data import load_cached


def _logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logloss(p, y, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def _fit_cell_shift(p, y, mask, max_iter=80, tol=1e-9):
    """Find scalar logit shift δ minimising logloss in the cell."""
    if mask.sum() == 0:
        return 0.0
    p_cell = p[mask]
    y_cell = y[mask].astype(np.float64)
    z0 = _logit(p_cell)
    # Newton's method on δ: minimise mean -[y log σ(z+δ) + (1-y) log(1-σ(z+δ))]
    delta = 0.0
    for _ in range(max_iter):
        q = _sigmoid(z0 + delta)
        grad = float((q - y_cell).mean())
        hess = float((q * (1 - q)).mean())
        if hess < 1e-12:
            break
        step = grad / hess
        delta -= step
        if abs(step) < tol:
            break
    return delta


def _test_mask(pred_g, gdo):
    all_games = np.unique(pred_g)
    if gdo is not None:
        known = all_games[
            np.array([int(gdo[g]) >= 0 for g in all_games], dtype=bool)
        ]
        ordered = known[np.argsort(np.array([int(gdo[g]) for g in known]))]
    else:
        ordered = np.sort(all_games)
    n_test = max(1, int(len(ordered) * 0.2))
    test_games = set(int(g) for g in ordered[-n_test:])
    return np.fromiter(
        (int(g) in test_games for g in pred_g),
        count=len(pred_g),
        dtype=bool,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--cache_pred", default="results/test_predictions.npz")
    args = ap.parse_args()

    print(f"[load] preds   {args.cache_pred}")
    npz = np.load(args.cache_pred)
    pred_g = npz["pred_g"]
    pred_obs = npz["pred_obs"].astype(np.int64)
    pred_p = npz["pred_p"]
    actual_y = npz["actual_y"]

    print(f"[load] arrays  {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)

    test_mask = _test_mask(pred_g, getattr(maps, "game_date_ordinal", None))
    p = pred_p[test_mask]
    y = actual_y[test_mask].astype(np.float64)
    g = pred_g[test_mask]
    obs = pred_obs[test_mask]
    n = len(p)

    team_size = arrays["team_sizes"][obs]
    gtype_arr = getattr(maps, "game_type", None)
    if gtype_arr is None:
        modes = np.array(["offline"] * len(g))
    else:
        modes = np.array([str(gtype_arr[int(gi)]) for gi in g])

    base_ll = _logloss(p, y)
    print(f"\nBaseline      n={n:>9,}  logloss = {base_ll:.5f}")

    # ---------- (A) per-size shift ----------
    sizes = sorted(set(int(s) for s in team_size))
    p_a = p.copy()
    deltas_a = {}
    for s in sizes:
        mask = team_size == s
        d = _fit_cell_shift(p, y, mask)
        deltas_a[s] = d
        p_a[mask] = _sigmoid(_logit(p[mask]) + d)
    ll_a = _logloss(p_a, y)
    print(f"\n(A) δ[size]     logloss = {ll_a:.5f}   gain = {base_ll - ll_a:+.5f}")
    print("    optimal post-hoc shift per size:")
    for s in sizes:
        m = team_size == s
        print(
            f"      size {s:>2}  n={int(m.sum()):>9,}  "
            f"δ̂ = {deltas_a[s]:+.4f}  "
            f"(p̂ {p[m].mean():.3f} → {p_a[m].mean():.3f}, "
            f"y {y[m].mean():.3f})"
        )

    # ---------- (B) per (size, mode) shift ----------
    p_b = p.copy()
    deltas_b = {}
    print("\n(B) δ[size, mode]:")
    print(f"      {'size':<6}{'mode':<10}{'n':>9}{'δ̂':>10}")
    for s in sizes:
        for m_ in ("offline", "sync", "async"):
            mask = (team_size == s) & (modes == m_)
            n_cell = int(mask.sum())
            if n_cell < 50:
                d = 0.0
            else:
                d = _fit_cell_shift(p, y, mask)
            deltas_b[(s, m_)] = d
            p_b[mask] = _sigmoid(_logit(p[mask]) + d)
            if n_cell > 0:
                print(f"      {s:<6}{m_:<10}{n_cell:>9,}{d:>+10.4f}")
    ll_b = _logloss(p_b, y)
    print(
        f"\n(B) δ[size,mode] logloss = {ll_b:.5f}   "
        f"gain = {base_ll - ll_b:+.5f}"
    )

    print(
        f"\nSummary:  baseline = {base_ll:.5f}  "
        f"size-only = {ll_a:.5f} (Δ {ll_a - base_ll:+.5f})  "
        f"size×mode = {ll_b:.5f} (Δ {ll_b - base_ll:+.5f})"
    )
    print(
        f"Mode adds:  {ll_a - ll_b:.5f} extra logloss reduction "
        f"vs size-only "
        f"(< 0.0005 → not worth a per-mode δ_size)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
