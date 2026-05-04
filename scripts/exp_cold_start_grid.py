"""Focused 2-D sweep over (cold_init_theta, games_offset).

Anchors at the previously chosen (θ=−0.3, offset=0.25) and varies one
axis at a time, plus a couple of diagonal points. Runs ~30 min.

Output: results/exp_cold_start_grid.json + readable table to stdout.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential


CACHE_FILE = "data.npz"

# 6 cells: θ-row at offset=0.25 (4 points) + offset-row at θ=-0.3 (2 extra).
GRID: list[tuple[float, float]] = [
    (-0.5, 0.25),
    (-0.4, 0.25),
    (-0.3, 0.25),  # current pick
    (-0.2, 0.25),
    (-0.3, 0.10),
    (-0.3, 0.50),
    # Round 2: confirm boundary (θ might still be improving past −0.5)
    # and try the cross of best-on-each-axis.
    (-0.6, 0.25),
    (-0.5, 0.10),
]


def main() -> int:
    print(f"Loading {CACHE_FILE} ...", flush=True)
    arrays, maps = load_cached(CACHE_FILE)
    print(
        f"Data: {len(arrays['q_idx']):,} obs, {maps.num_players:,} players, "
        f"{maps.num_questions:,} questions",
        flush=True,
    )

    gdo = maps.game_date_ordinal
    n_games = len(maps.idx_to_game_id)
    known = [g for g in range(n_games) if g < len(gdo) and int(gdo[g]) >= 0]
    known_sorted = sorted(known, key=lambda g: int(gdo[g]))
    n_test = max(1, int(len(known_sorted) * 0.2))
    test_games = set(known_sorted[-n_test:])
    print(
        f"Backtest split: {len(known_sorted)-n_test} train games, "
        f"{n_test} test games (last 20% by date)",
        flush=True,
    )

    rows: list[dict] = []
    for theta_p, offset in GRID:
        print(
            f"\n===== θ_prior={theta_p:+.2f}  offset={offset:.2f} =====",
            flush=True,
        )
        cfg = Config(
            cold_init_theta=theta_p,
            games_offset=offset,
        )
        full = run_sequential(
            arrays, maps, cfg,
            verbose=False,
            collect_history=False,  # skip — we don't need drift here
            collect_predictions=True,
        )
        preds = full.predictions or {}
        if not preds:
            continue
        mask = np.array(
            [int(g) in test_games for g in preds["game_idx"]],
            dtype=bool,
        )
        m = compute_metrics(preds["pred_p"][mask], preds["actual_y"][mask])

        seen = full.players.seen
        active = (full.players.games >= 30) & seen
        th_active = full.players.theta[active]
        row = {
            "cold_init_theta": theta_p,
            "games_offset": offset,
            "logloss": m["logloss"],
            "brier": m["brier"],
            "auc": m["auc"],
            "n_test_obs": int(mask.sum()),
            "theta_active_mean": float(th_active.mean()) if th_active.size else float("nan"),
            "theta_active_median": float(np.median(th_active)) if th_active.size else float("nan"),
        }
        rows.append(row)
        print(
            f"  logloss {row['logloss']:.4f}  Brier {row['brier']:.4f}  "
            f"AUC {row['auc']:.4f}  | θ_active mean {row['theta_active_mean']:+.3f}",
            flush=True,
        )

    # Sort by logloss
    rows_sorted = sorted(rows, key=lambda r: r["logloss"])
    print("\n" + "=" * 86)
    print(
        f"{'θ_prior':>8} {'offset':>7} {'logloss':>9} {'Brier':>8} "
        f"{'AUC':>7} {'θ_act mean':>11} {'θ_act med':>11}"
    )
    print("-" * 86)
    for r in rows_sorted:
        print(
            f"{r['cold_init_theta']:>+8.2f} {r['games_offset']:>7.2f} "
            f"{r['logloss']:>9.4f} {r['brier']:>8.4f} {r['auc']:>7.4f} "
            f"{r['theta_active_mean']:>+11.3f} {r['theta_active_median']:>+11.3f}"
        )
    print("=" * 86)

    out = "results/exp_cold_start_grid.json"
    with open(out, "w") as f:
        json.dump({"grid": rows}, f, indent=2)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
