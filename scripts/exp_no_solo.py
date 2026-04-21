"""Counterfactual: retrain rating after dropping every team_size==1 sample globally.

Goal: see Belov (player_id=2954) θ when soloists are excluded from the corpus.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from data import load_cached
from rating.engine import Config, run_sequential


BELOV_PID = 2954


def main(cache_path: str = "data.npz") -> int:
    print(f"Loading cache from {cache_path}...", flush=True)
    arrays, maps = load_cached(cache_path)

    n_obs = len(arrays["q_idx"])
    team_sizes = arrays["team_sizes"]
    keep = team_sizes != 1
    n_kept = int(keep.sum())
    print(
        f"Filtering: keep {n_kept:,}/{n_obs:,} samples "
        f"({100*n_kept/n_obs:.2f}%); dropping {n_obs-n_kept:,} solo (size==1)",
        flush=True,
    )

    keep_pi = np.repeat(keep, team_sizes)
    new_arrays = {
        "q_idx": arrays["q_idx"][keep],
        "taken": arrays["taken"][keep],
        "team_sizes": team_sizes[keep],
        "player_indices_flat": arrays["player_indices_flat"][keep_pi],
    }
    if "game_idx" in arrays:
        new_arrays["game_idx"] = arrays["game_idx"][keep]
    if "team_strength" in arrays:
        new_arrays["team_strength"] = arrays["team_strength"][keep]

    cfg = Config()
    print(
        f"Config defaults: eta0={cfg.eta0}, w_online={cfg.w_online}, "
        f"team_size_max={cfg.team_size_max}, recenter_target={cfg.recenter_target}",
        flush=True,
    )

    t0 = time.time()
    result = run_sequential(new_arrays, maps, cfg, collect_history=False, collect_predictions=False)
    print(f"Training done in {time.time()-t0:.1f}s", flush=True)

    pidx = maps.player_id_to_idx.get(BELOV_PID)
    if pidx is None:
        print(f"Belov (id={BELOV_PID}) not in maps", flush=True)
        return 1

    ps = result.players
    theta_b = float(ps.theta[pidx])
    games_b = int(ps.games[pidx])

    seen = ps.seen
    seen_idx = np.where(seen)[0]
    theta_seen = ps.theta[seen_idx]
    order = seen_idx[np.argsort(theta_seen)[::-1]]
    rank_b = int(np.where(order == pidx)[0][0]) + 1

    print()
    print("=== BELOV (id=2954) WITHOUT SOLO SAMPLES ===")
    print(f"  θ      = {theta_b:.4f}")
    print(f"  games  = {games_b}")
    print(f"  rank   = {rank_b} / {len(seen_idx)}")

    print()
    print("=== TOP-15 by θ (no soloists) ===")
    for r, idx in enumerate(order[:15], 1):
        pid = maps.idx_to_player_id[int(idx)]
        marker = " <-- Belov" if int(idx) == pidx else ""
        print(f"  {r:2d}. pid={pid:>7d}  θ={float(ps.theta[idx]):+.4f}  games={int(ps.games[idx]):4d}{marker}")

    return 0


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else "data.npz"
    sys.exit(main(cache))
