"""Sweep w_solo (+ optional async-solo lapse init) for async/solo/newbie slices.

Honest cell-holdout (default fraction=0.10, seed=42).  Reports overall,
async, async+solo, and async+solo+newbie (mean roster games < 15 at
prediction time) logloss and calibration bias (actual − predicted).

Outputs ``results/exp_async_solo_newbies.csv``.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential


def _build_offsets(team_sizes: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(team_sizes.astype(np.int64))])


def _bucket_mode(g: str) -> str:
    if "async" in g:
        return "async"
    if "sync" in g:
        return "sync"
    return "offline"


def _mean_games_at_prediction(
    arrays: dict[str, np.ndarray],
    maps,
    pred: dict,
    holdout_mask: np.ndarray,
) -> np.ndarray:
    """Per-obs mean pre-tournament game count on the predicting roster."""
    n_obs = len(arrays["q_idx"])
    n_players = maps.num_players
    offsets = _build_offsets(arrays["team_sizes"])
    player_flat = arrays["player_indices_flat"]
    games_count = np.zeros(n_players, dtype=np.int32)

    gdo = getattr(maps, "game_date_ordinal", None)
    all_games = np.unique(pred["game_idx"])
    if gdo is not None:
        known = all_games[
            np.array([gdo[g] >= 0 for g in all_games], dtype=bool)
        ]
        game_order = (
            known[np.argsort([gdo[g] for g in known])]
            if len(known) > 0
            else all_games
        )
    else:
        game_order = np.sort(all_games)

    obs_by_game: dict[int, list[int]] = {}
    for oi in range(n_obs):
        obs_by_game.setdefault(int(arrays["game_idx"][oi]), []).append(oi)

    mean_games = np.zeros(n_obs, dtype=np.float32)
    for g in game_order:
        for i in obs_by_game.get(int(g), []):
            s, e = int(offsets[i]), int(offsets[i + 1])
            if e > s:
                mean_games[i] = float(games_count[player_flat[s:e]].mean())
        seen: set[int] = set()
        for i in obs_by_game.get(int(g), []):
            s, e = int(offsets[i]), int(offsets[i + 1])
            for pi in player_flat[s:e]:
                seen.add(int(pi))
        for pi in seen:
            games_count[pi] += 1

    obs_h = pred["obs_idx"][holdout_mask]
    return mean_games[obs_h]


@dataclass(frozen=True)
class Trial:
    label: str
    w_solo: float = 0.7
    lapse_init_async_solo: float = 0.03


def _trial_configs(trials: str) -> list[Trial]:
    if trials == "w_solo":
        return [Trial(label=f"w_solo={v:.2f}", w_solo=v) for v in (0.3, 0.5, 0.7, 0.9)]
    if trials == "lapse_async_solo":
        return [
            Trial(label="baseline", w_solo=0.7, lapse_init_async_solo=0.03),
            Trial(label="lapse_asy_solo=0.01", w_solo=0.7, lapse_init_async_solo=0.01),
            Trial(label="lapse_asy_solo=0.05", w_solo=0.7, lapse_init_async_solo=0.05),
            Trial(label="lapse_asy_solo=0.08", w_solo=0.7, lapse_init_async_solo=0.08),
        ]
    if trials == "all":
        out: list[Trial] = []
        for v in (0.3, 0.5, 0.7, 0.9):
            out.append(Trial(label=f"w_solo={v:.2f}", w_solo=v))
        for lv in (0.01, 0.05, 0.08):
            out.append(
                Trial(
                    label=f"w_solo=0.70,lapse_asy_solo={lv:.2f}",
                    w_solo=0.7,
                    lapse_init_async_solo=lv,
                )
            )
        return out
    raise ValueError(f"unknown trials preset: {trials!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_async_solo_newbies.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--trials",
        choices=["w_solo", "lapse_async_solo", "all"],
        default="w_solo",
        help="w_solo only (4 runs), lapse async-solo init (4 runs), or both (7 runs)",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="n_extra_epochs=0 for quicker comparative sweeps",
    )
    args = ap.parse_args()

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)
    team_sizes = arrays["team_sizes"].astype(np.int32)
    gtype = maps.game_type

    trials = _trial_configs(args.trials)
    rows: list[dict] = []
    mean_games_h: np.ndarray | None = None

    for trial in trials:
        print(f"\n=== {trial.label} ===", flush=True)
        cfg = Config(
            w_solo=trial.w_solo,
            lapse_init_async_solo=trial.lapse_init_async_solo,
            holdout_obs_fraction=args.holdout,
            holdout_seed=args.seed,
            n_extra_epochs=0 if args.fast else 1,
        )
        t0 = time.time()
        result = run_sequential(
            arrays, maps, cfg, verbose=False, collect_predictions=True
        )
        elapsed = time.time() - t0

        pred = result.predictions
        if pred is None:
            raise RuntimeError("no predictions collected")
        mask = pred["is_holdout"].astype(bool)
        p = pred["pred_p"][mask]
        y = pred["actual_y"][mask].astype(np.float64)
        obs = pred["obs_idx"][mask]
        games = pred["game_idx"][mask]
        ts = team_sizes[obs]

        if mean_games_h is None:
            mean_games_h = _mean_games_at_prediction(arrays, maps, pred, mask)

        modes = np.array(
            [
                _bucket_mode(gtype[g] if g < len(gtype) else "offline")
                for g in games
            ],
            dtype=object,
        )
        newbie = mean_games_h < 15.0

        slice_masks = {
            "all": np.ones(len(p), dtype=bool),
            "async": modes == "async",
            "async_solo": (modes == "async") & (ts == 1),
            "async_solo_newbie": (modes == "async") & (ts == 1) & newbie,
        }

        for slice_name, sl in slice_masks.items():
            if not sl.any():
                continue
            m = compute_metrics(p[sl], y[sl].astype(int))
            bias = float((y[sl] - p[sl]).mean())
            row = {
                "label": trial.label,
                "w_solo": trial.w_solo,
                "lapse_init_async_solo": trial.lapse_init_async_solo,
                "slice": slice_name,
                "n": int(sl.sum()),
                "logloss": round(float(m["logloss"]), 6),
                "bias_pp": round(bias * 100.0, 4),
                "brier": round(float(m["brier"]), 6),
                "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
                "elapsed_sec": round(elapsed, 1) if slice_name == "all" else "",
            }
            rows.append(row)

        m_all = compute_metrics(p, y.astype(int))
        bias_all = float((y - p).mean())
        print(
            f"  overall       : ll={m_all['logloss']:.4f}  bias={bias_all*100:+.2f}pp"
            f"  ({elapsed:.1f}s)",
            flush=True,
        )
        for slice_name in ("async", "async_solo", "async_solo_newbie"):
            sl = slice_masks[slice_name]
            if not sl.any():
                continue
            ms = compute_metrics(p[sl], y[sl].astype(int))
            bs = float((y[sl] - p[sl]).mean())
            print(
                f"  {slice_name:16s}: ll={ms['logloss']:.4f}  "
                f"bias={bs*100:+.2f}pp  n={int(sl.sum())}",
                flush=True,
            )

        if result.lapse is not None:
            print(
                f"  learned lapse[async,solo]={float(result.lapse[2, 1]):.4f}",
                flush=True,
            )
        if result.recal is not None:
            a, b = result.recal[2, 1]
            print(f"  learned recal[async,solo] α={a:.3f} β={b:.3f}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label", "w_solo", "lapse_init_async_solo", "slice", "n",
        "logloss", "bias_pp", "brier", "auc", "elapsed_sec",
    ]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n=== Ranked by overall logloss ===")
    overall = sorted(
        [r for r in rows if r["slice"] == "all"],
        key=lambda r: r["logloss"],
    )
    for r in overall:
        marker = "  ★" if r is overall[0] else ""
        print(
            f"  {r['label']:30s}  ll={r['logloss']:.4f}  "
            f"bias={r['bias_pp']:+.2f}pp{marker}"
        )

    print("\n=== Ranked by async_solo_newbie logloss ===")
    target = sorted(
        [r for r in rows if r["slice"] == "async_solo_newbie"],
        key=lambda r: r["logloss"],
    )
    for r in target:
        marker = "  ★" if r is target[0] else ""
        print(
            f"  {r['label']:30s}  ll={r['logloss']:.4f}  "
            f"bias={r['bias_pp']:+.2f}pp{marker}"
        )

    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
