"""
Cold-start experiment: compare three regimes for initialising new players
and the adaptive learning-rate offset.

NOTE (2026-05): this script predates the 2026-05 Config cleanup and
references ``cold_init_use_team_mean``, which was removed when the
fixed-prior cold-start became the only behaviour.  Saved results are
in ``results/exp_cold_start.json``.  Restore from git history before
2026-05 to re-run.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date

import numpy as np

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential


CACHE_FILE = "data.npz"


VARIANTS: list[tuple[str, dict]] = [
    ("baseline (team-mean inherit)", dict()),
    (
        "fixed prior θ=0, no inherit",
        dict(cold_init_use_team_mean=False, cold_init_theta=0.0),
    ),
    (
        "fixed prior θ=0, no inherit + rookie boost (offset=0.25)",
        dict(
            cold_init_use_team_mean=False,
            cold_init_theta=0.0,
            games_offset=0.25,
        ),
    ),
    (
        "fixed prior θ=-0.3, no inherit + rookie boost (offset=0.25)",
        dict(
            cold_init_use_team_mean=False,
            cold_init_theta=-0.3,
            games_offset=0.25,
        ),
    ),
]


def population_drift(result, maps) -> dict[int, dict[str, float]]:
    """For each year, take the top-1000 players by games-so-far at the
    end of the year and report median/mean of their θ-at-year-end."""
    if result.history is None:
        return {}
    pid_to_idx = {pid: i for i, pid in enumerate(maps.idx_to_player_id)}
    gid_to_idx = {gid: g for g, gid in enumerate(maps.idx_to_game_id)}
    gdo = maps.game_date_ordinal

    last_theta_by_year: dict[int, dict[int, float]] = defaultdict(dict)
    games_by_year: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for pid, gid, theta in result.history:
        g = gid_to_idx.get(gid)
        if g is None or g >= len(gdo):
            continue
        ordn = int(gdo[g])
        if ordn < 0:
            continue
        year = date.fromordinal(ordn).year
        pidx = pid_to_idx.get(pid)
        if pidx is None:
            continue
        last_theta_by_year[year][pidx] = float(theta)
        games_by_year[year][pidx] += 1

    cum_games: dict[int, int] = defaultdict(int)
    out: dict[int, dict[str, float]] = {}
    for year in sorted(games_by_year):
        for pidx, n in games_by_year[year].items():
            cum_games[pidx] += n
        if not cum_games:
            continue
        order = sorted(cum_games.items(), key=lambda kv: -kv[1])[:1000]
        thetas = [
            last_theta_by_year[year].get(pidx)
            for pidx, _ in order
            if pidx in last_theta_by_year[year]
        ]
        if not thetas:
            continue
        arr = np.array(thetas, dtype=np.float64)
        out[year] = {
            "n": int(len(arr)),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "p10": float(np.quantile(arr, 0.10)),
            "p90": float(np.quantile(arr, 0.90)),
        }
    return out


def main() -> int:
    sys.exit(
        "scripts/exp_cold_start.py is historical (pre-2026-05 Config "
        "cleanup) and no longer runs against the current Config. "
        "See results/exp_cold_start.json for the original output."
    )
    print(f"Loading {CACHE_FILE} ...", flush=True)
    arrays, maps = load_cached(CACHE_FILE)
    print(
        f"Data: {len(arrays['q_idx']):,} obs, "
        f"{maps.num_players:,} players, {maps.num_questions:,} questions",
        flush=True,
    )

    # Pre-compute test-game set (last 20% by date) on the maps, shared
    # across variants so all configs are scored on the same observations.
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

    summary: list[dict] = []
    drifts: dict[str, dict] = {}
    for name, overrides in VARIANTS:
        print(f"\n===== {name} =====", flush=True)
        cfg = Config(**overrides)
        full = run_sequential(
            arrays, maps, cfg,
            verbose=False, collect_history=True, collect_predictions=True,
        )
        preds = full.predictions or {}
        if preds:
            mask = np.array(
                [int(g) in test_games for g in preds["game_idx"]],
                dtype=bool,
            )
            m = compute_metrics(preds["pred_p"][mask], preds["actual_y"][mask])
            n_test_obs = int(mask.sum())
        else:
            m = {"logloss": float("nan"), "brier": float("nan"), "auc": float("nan")}
            n_test_obs = 0

        drift = population_drift(full, maps)
        drifts[name] = drift

        seen = full.players.seen
        th = full.players.theta[seen]
        active = (full.players.games >= 30) & seen
        th_active = full.players.theta[active]
        row = {
            "name": name,
            "logloss": m["logloss"],
            "brier": m["brier"],
            "auc": m["auc"],
            "n_test_obs": n_test_obs,
            "n_seen_players": int(seen.sum()),
            "theta_all_mean": float(th.mean()),
            "theta_all_std": float(th.std()),
            "n_active_30plus": int(active.sum()),
            "theta_active_mean": float(th_active.mean()) if th_active.size else float("nan"),
            "theta_active_median": float(np.median(th_active)) if th_active.size else float("nan"),
        }
        summary.append(row)
        print(
            f"  logloss {row['logloss']:.4f}  Brier {row['brier']:.4f}  "
            f"AUC {row['auc']:.4f}  | active30+ {row['n_active_30plus']:,}  "
            f"θ active mean {row['theta_active_mean']:+.3f}  median {row['theta_active_median']:+.3f}",
            flush=True,
        )

    # ---- comparative report -----------------------------------------
    print("\n" + "=" * 110)
    print(
        f"{'variant':<60} {'logloss':>9} {'Brier':>8} {'AUC':>7} "
        f"{'θ_act mean':>11} {'θ_act med':>11} {'#active':>9}"
    )
    print("-" * 110)
    for r in summary:
        print(
            f"{r['name']:<60} {r['logloss']:>9.4f} {r['brier']:>8.4f} "
            f"{r['auc']:>7.4f} {r['theta_active_mean']:>+11.3f} "
            f"{r['theta_active_median']:>+11.3f} {r['n_active_30plus']:>9,}"
        )
    print("=" * 110)

    print("\nTop-1000 (by cumulative games) median θ at end of each year:\n")
    years = sorted({y for d in drifts.values() for y in d})
    header = f"{'year':>5} " + " ".join(f"{n[:30]:>32}" for n, _ in VARIANTS)
    print(header)
    for y in years:
        cells = []
        for name, _ in VARIANTS:
            d = drifts[name].get(y)
            cells.append(f"{d['median']:>+8.3f} (n={d['n']:>4})" if d else " " * 18)
        print(f"{y:>5} " + " ".join(f"{c:>32}" for c in cells))

    out_path = "results/exp_cold_start.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "drifts": drifts}, f, indent=2)
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
