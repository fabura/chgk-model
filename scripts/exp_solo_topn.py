"""Validate the solo-channel default before commit:

  1. Train baseline (use_solo_channel=False) and the proposed defaults
     (use_solo_channel=True, w_solo=0.1).
  2. For every player compute the fraction of their observations that
     came from solo (team_size==1) plays.
  3. Print top-30 by baseline θ side-by-side with their new θ + solo
     fraction.  Pure team players should barely move; soloists should
     visibly drop.
  4. Also print top-30 by NEW θ to make sure no fresh artefacts pop up.

No backtest here — pure in-sample θ comparison plus solo% diagnostic.
"""
from __future__ import annotations

import sys
import time

import duckdb
import numpy as np

from data import load_cached
from rating.engine import Config, run_sequential


DUCKDB_PATH = "website/data/chgk.duckdb"


def _name_lookup(pids: list[int]) -> dict[int, str]:
    if not pids:
        return {}
    try:
        conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    except Exception:
        return {}
    try:
        ph = ", ".join("?" for _ in pids)
        rows = conn.execute(
            f"SELECT player_id, last_name, first_name FROM players "
            f"WHERE player_id IN ({ph})",
            pids,
        ).fetchall()
        return {int(r[0]): f"{r[1]} {r[2]}" for r in rows}
    finally:
        conn.close()


def _solo_fraction_per_player(arrays, num_players: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_obs_per_player, n_solo_obs_per_player) — both length num_players.

    Counts each observation by every team member that participated.  A
    "solo obs" is one where team_size==1 (so each contributes 1 each).
    """
    team_sizes = arrays["team_sizes"]
    flat = arrays["player_indices_flat"]
    offsets = np.zeros(len(team_sizes) + 1, dtype=np.int64)
    np.cumsum(team_sizes, out=offsets[1:])

    n_obs = np.zeros(num_players, dtype=np.int64)
    n_solo = np.zeros(num_players, dtype=np.int64)
    for i in range(len(team_sizes)):
        s, e = int(offsets[i]), int(offsets[i + 1])
        is_solo = (e - s) == 1
        for pidx in flat[s:e]:
            p = int(pidx)
            n_obs[p] += 1
            if is_solo:
                n_solo[p] += 1
    return n_obs, n_solo


def _train(arrays, maps, *, use_solo: bool, w_solo: float) -> dict:
    cfg = Config()
    cfg.use_solo_channel = use_solo
    if use_solo:
        cfg.w_solo = w_solo
    t0 = time.time()
    res = run_sequential(arrays, maps, cfg, verbose=False, collect_predictions=False)
    dt = time.time() - t0
    seen = res.players.seen
    return {
        "theta": np.where(seen, res.players.theta, np.nan),
        "games": res.players.games.copy(),
        "seen": seen.copy(),
        "secs": dt,
    }


def main(cache_path: str = "data.npz") -> int:
    print(f"Loading cache from {cache_path}...", flush=True)
    arrays, maps = load_cached(cache_path)
    num_players = maps.num_players

    print("Computing per-player solo fraction (one pass over arrays)...", flush=True)
    t0 = time.time()
    n_obs, n_solo = _solo_fraction_per_player(arrays, num_players)
    print(f"  ({time.time()-t0:.1f}s)", flush=True)

    print()
    print("Training baseline (use_solo_channel=False)...", flush=True)
    base = _train(arrays, maps, use_solo=False, w_solo=0.0)
    print(f"  done in {base['secs']:.1f}s", flush=True)

    print()
    print("Training new defaults (use_solo_channel=True, w_solo=0.3)...", flush=True)
    new = _train(arrays, maps, use_solo=True, w_solo=0.3)
    print(f"  done in {new['secs']:.1f}s", flush=True)

    seen = base["seen"] & new["seen"]
    seen_idx = np.where(seen)[0]

    base_th = base["theta"]
    new_th = new["theta"]
    games = base["games"]

    base_order = seen_idx[np.argsort(-base_th[seen_idx], kind="stable")]
    new_order  = seen_idx[np.argsort(-new_th[seen_idx],  kind="stable")]
    base_rank = np.empty(num_players, dtype=np.int64)
    new_rank  = np.empty(num_players, dtype=np.int64)
    base_rank[base_order] = np.arange(1, len(base_order) + 1)
    new_rank[new_order]   = np.arange(1, len(new_order)  + 1)

    def _fmt_table(idxs: np.ndarray, title: str) -> None:
        pids = [int(maps.idx_to_player_id[int(i)]) for i in idxs]
        names = _name_lookup(pids)
        print()
        print("=" * 110)
        print(title)
        print(f"{'#':>3s}  {'pid':>7s}  {'name':25s}  "
              f"{'base θ':>8s}  {'new θ':>8s}  {'Δθ':>7s}  "
              f"{'base rk':>7s}  {'new rk':>7s}  "
              f"{'games':>5s}  {'solo%':>6s}")
        print("-" * 110)
        for n, idx in enumerate(idxs, 1):
            i = int(idx)
            pid = int(maps.idx_to_player_id[i])
            nm = names.get(pid, "?")[:25]
            solo_frac = (n_solo[i] / n_obs[i]) if n_obs[i] > 0 else 0.0
            print(
                f"{n:>3d}  {pid:>7d}  {nm:25s}  "
                f"{base_th[i]:+8.3f}  {new_th[i]:+8.3f}  "
                f"{new_th[i]-base_th[i]:+7.3f}  "
                f"{base_rank[i]:>7d}  {new_rank[i]:>7d}  "
                f"{int(games[i]):>5d}  {100*solo_frac:>5.1f}%"
            )

    _fmt_table(base_order[:30], "TOP 30 by BASELINE θ  (sorted by old rank)")
    _fmt_table(new_order[:30],  "TOP 30 by NEW θ       (sorted by new rank)")

    # Also: highest-solo% players among the top-1000 by baseline θ.
    # These are the ones the solo-channel was specifically introduced
    # to fix.
    top1000 = base_order[:1000]
    top1000_solo = (n_solo[top1000] / np.maximum(n_obs[top1000], 1))
    high_solo_idx = top1000[np.argsort(-top1000_solo, kind="stable")[:20]]
    _fmt_table(
        high_solo_idx,
        "TOP 20 by SOLO% within baseline top-1000 (the players the channel targets)",
    )

    return 0


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else "data.npz"
    sys.exit(main(cache))
