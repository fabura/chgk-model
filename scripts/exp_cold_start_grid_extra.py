"""Two extra cells to confirm the boundary in the cold-start sweep."""
from __future__ import annotations

import json
import sys

import numpy as np

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential


CACHE_FILE = "data.npz"
GRID: list[tuple[float, float]] = [
    (-1.5, 0.25),
    (-2.0, 0.25),
]


def main() -> int:
    print(f"Loading {CACHE_FILE} ...", flush=True)
    arrays, maps = load_cached(CACHE_FILE)
    gdo = maps.game_date_ordinal
    n_games = len(maps.idx_to_game_id)
    known = [g for g in range(n_games) if g < len(gdo) and int(gdo[g]) >= 0]
    known_sorted = sorted(known, key=lambda g: int(gdo[g]))
    n_test = max(1, int(len(known_sorted) * 0.2))
    test_games = set(known_sorted[-n_test:])

    rows: list[dict] = []
    for theta_p, offset in GRID:
        print(f"\n===== θ_prior={theta_p:+.2f}  offset={offset:.2f} =====", flush=True)
        cfg = Config(
            cold_init_theta=theta_p,
            games_offset=offset,
        )
        full = run_sequential(arrays, maps, cfg, verbose=False, collect_predictions=True)
        preds = full.predictions or {}
        if not preds:
            continue
        mask = np.array([int(g) in test_games for g in preds["game_idx"]], dtype=bool)
        m = compute_metrics(preds["pred_p"][mask], preds["actual_y"][mask])
        seen = full.players.seen
        active = (full.players.games >= 30) & seen
        th_active = full.players.theta[active]
        rows.append({
            "cold_init_theta": theta_p,
            "games_offset": offset,
            "logloss": m["logloss"],
            "brier": m["brier"],
            "auc": m["auc"],
            "theta_active_mean": float(th_active.mean()),
            "theta_active_median": float(np.median(th_active)),
        })
        print(
            f"  logloss {rows[-1]['logloss']:.4f}  Brier {rows[-1]['brier']:.4f}  "
            f"AUC {rows[-1]['auc']:.4f}  | θ_active mean {rows[-1]['theta_active_mean']:+.3f}",
            flush=True,
        )

    with open("results/exp_cold_start_grid_extra.json", "w") as f:
        json.dump({"grid": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
