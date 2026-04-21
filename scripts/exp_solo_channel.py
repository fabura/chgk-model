"""Sweep over the new solo-channel weights and report:

  * Backtest metrics (logloss/AUC, in total and per tournament type) so
    we know the change is not net-negative for the model.
  * Belov (player_id=2954) θ + rank, since that's the player the
    channel was introduced to fix.

Baseline = ``use_solo_channel=False`` (legacy behaviour).
Sweep    = ``use_solo_channel=True`` × ``w_solo ∈ {0.0, 0.1, 0.3, 0.5}``.

All other knobs (``w_solo_questions=0``, ``w_solo_log_a=0``,
``w_size_solo=1``, ``w_pos_solo=0``) stay at their Config defaults —
those are the "right" values per the design discussion; the only thing
worth sweeping right now is how strongly solo results should tug θ.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from data import load_cached
from rating.backtest import backtest as do_backtest
from rating.engine import Config


BELOV_PID = 2954


def _belov_summary(result, maps) -> tuple[float, int, int]:
    """Return (theta, games, rank) for Belov among `seen` players."""
    pidx = maps.player_id_to_idx.get(BELOV_PID)
    if pidx is None:
        return float("nan"), 0, -1
    ps = result.players
    seen_idx = np.where(ps.seen)[0]
    theta_seen = ps.theta[seen_idx]
    order = seen_idx[np.argsort(theta_seen)[::-1]]
    rank = int(np.where(order == pidx)[0][0]) + 1
    return float(ps.theta[pidx]), int(ps.games[pidx]), rank


def _make_cfg(use_solo: bool, w_solo: float) -> Config:
    cfg = Config()
    cfg.use_solo_channel = use_solo
    if use_solo:
        cfg.w_solo = w_solo
        # Other w_solo_* keep Config defaults (questions=0, log_a=0,
        # size=1, pos=0) — these were settled during the design.
    return cfg


def main(cache_path: str = "data.npz") -> int:
    print(f"Loading cache from {cache_path}...", flush=True)
    arrays, maps = load_cached(cache_path)

    runs: list[tuple[str, Config]] = [
        ("baseline (legacy)", _make_cfg(use_solo=False, w_solo=0.0)),
        ("solo w=0.0",        _make_cfg(use_solo=True,  w_solo=0.0)),
        ("solo w=0.1",        _make_cfg(use_solo=True,  w_solo=0.1)),
        ("solo w=0.3",        _make_cfg(use_solo=True,  w_solo=0.3)),
        ("solo w=0.5",        _make_cfg(use_solo=True,  w_solo=0.5)),
    ]

    rows = []
    for label, cfg in runs:
        print()
        print(f"========== {label} ==========", flush=True)
        t0 = time.time()
        metrics = do_backtest(arrays, maps, cfg, test_fraction=0.2, verbose=False)
        dt = time.time() - t0

        result = metrics.pop("result")
        theta_b, games_b, rank_b = _belov_summary(result, maps)
        n_seen = int(result.players.seen.sum())

        by_type = metrics.get("by_type", {})
        ll_off = by_type.get("offline", {}).get("logloss", float("nan"))
        ll_syn = by_type.get("sync",    {}).get("logloss", float("nan"))
        ll_asy = by_type.get("async",   {}).get("logloss", float("nan"))

        rows.append({
            "label":     label,
            "logloss":   metrics["logloss"],
            "auc":       metrics["auc"],
            "ll_off":    ll_off,
            "ll_syn":    ll_syn,
            "ll_asy":    ll_asy,
            "belov_th":  theta_b,
            "belov_g":   games_b,
            "belov_rk":  rank_b,
            "n_seen":    n_seen,
            "secs":      dt,
        })

        print(
            f"  test logloss = {metrics['logloss']:.4f}  AUC = {metrics['auc']:.4f}  "
            f"(off {ll_off:.4f} | syn {ll_syn:.4f} | asy {ll_asy:.4f})",
            flush=True,
        )
        print(
            f"  Belov: θ = {theta_b:+.4f}  games = {games_b}  rank = {rank_b}/{n_seen}",
            flush=True,
        )
        print(f"  ({dt:.1f}s)", flush=True)

    print()
    print("=" * 100)
    print(f"{'config':22s}  {'logloss':>8s}  {'AUC':>6s}  "
          f"{'ll_off':>7s}  {'ll_syn':>7s}  {'ll_asy':>7s}  "
          f"{'Belov θ':>9s}  {'games':>5s}  {'rank':>6s}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['label']:22s}  {r['logloss']:8.4f}  {r['auc']:6.4f}  "
            f"{r['ll_off']:7.4f}  {r['ll_syn']:7.4f}  {r['ll_asy']:7.4f}  "
            f"{r['belov_th']:+9.4f}  {r['belov_g']:5d}  "
            f"{r['belov_rk']:>6d}"
        )
    print("=" * 100)

    return 0


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else "data.npz"
    sys.exit(main(cache))
