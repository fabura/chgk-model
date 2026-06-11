"""Sweep difficulty-weighted loss under honest cell-holdout.

Per-observation gradient scale (forward unchanged):

  miss (y=0): w = (1 − p) ** diff_w_miss_power
  take (y=1): w = 1 + diff_w_take_boost · (1 − p)

Also runs a convergence check: pass-1 only (``n_extra_epochs=0``) vs
default two-pass training, reporting train avg log-likelihood.

Outputs ``results/exp_difficulty_weights_sweep.csv``.

Usage::

    python scripts/exp_difficulty_weights_sweep.py --cache_file data.npz

Cost: ~7 trials × ~10 min ≈ 70 min.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential

CASE_STUDY = {
    28751: "Semushin",
    27403: "Russo",
    34909: "Chernukha",
}


def _bucket_type(g: str) -> str:
    if "async" in g:
        return "async"
    if "sync" in g:
        return "sync"
    return "offline"


def _player_thetas(result, maps, pids: dict[int, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    pid_to_idx = {int(pid): i for i, pid in enumerate(maps.idx_to_player_id)}
    for db_id, label in pids.items():
        idx = pid_to_idx.get(int(db_id))
        out[f"theta_{label}"] = (
            round(float(result.players.theta[idx]), 4) if idx is not None else float("nan")
        )
    return out


def _solo_holdout_metrics(pred, arrays, mask: np.ndarray) -> dict[str, float]:
    """Holdout logloss on observations with team_size == 1."""
    obs_idx = pred["obs_idx"][mask]
    team_sizes = arrays["team_sizes"]
    solo = team_sizes[obs_idx] == 1
    if not np.any(solo):
        return {"solo_holdout_ll": float("nan"), "solo_holdout_n": 0}
    p = pred["pred_p"][mask][solo]
    y = pred["actual_y"][mask][solo]
    m = compute_metrics(p, y)
    return {
        "solo_holdout_ll": round(float(m["logloss"]), 6),
        "solo_holdout_n": int(solo.sum()),
    }


def _run_trial(
    arrays,
    maps,
    *,
    label: str,
    miss_power: float,
    take_boost: float,
    solo_only: bool,
    n_extra_epochs: int,
    holdout: float,
    seed: int,
    case_ids: dict[int, str],
) -> list[dict]:
    print(
        f"\n=== {label} "
        f"(miss={miss_power}, take={take_boost}, solo_only={solo_only}, "
        f"n_extra={n_extra_epochs}) ===",
        flush=True,
    )
    cfg = Config(
        diff_w_miss_power=miss_power,
        diff_w_take_boost=take_boost,
        diff_w_solo_only=solo_only,
        n_extra_epochs=n_extra_epochs,
        holdout_obs_fraction=holdout,
        holdout_seed=seed,
    )
    t0 = time.time()
    result = run_sequential(arrays, maps, cfg, verbose=False, collect_predictions=True)
    elapsed = time.time() - t0

    train_avg_ll = result.total_loglik / max(result.total_obs, 1)
    pred = result.predictions
    mask = pred["is_holdout"].astype(bool)
    p, y, g = pred["pred_p"][mask], pred["actual_y"][mask], pred["game_idx"][mask]
    m = compute_metrics(p, y)
    solo_m = _solo_holdout_metrics(pred, arrays, mask)

    row = {
        "config": label,
        "diff_w_miss_power": miss_power,
        "diff_w_take_boost": take_boost,
        "diff_w_solo_only": solo_only,
        "n_extra_epochs": n_extra_epochs,
        "slice": "all",
        "n_holdout": int(mask.sum()),
        "train_avg_ll": round(float(train_avg_ll), 6),
        "logloss": round(float(m["logloss"]), 6),
        "brier": round(float(m["brier"]), 6),
        "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
        "elapsed_sec": round(elapsed, 1),
        **solo_m,
    }
    row.update(_player_thetas(result, maps, case_ids))
    rows = [row]

    print(
        f"  train_ll={row['train_avg_ll']:.4f}  holdout_ll={row['logloss']:.4f}  "
        f"solo_ll={row.get('solo_holdout_ll', float('nan')):.4f}  "
        f"AUC={row['auc']}  ({elapsed:.0f}s)",
        flush=True,
    )
    for lbl in case_ids.values():
        k = f"theta_{lbl}"
        if k in row:
            print(f"  {k}={row[k]:+.4f}", flush=True)

    gtype = getattr(maps, "game_type", None)
    if gtype is not None:
        types = np.array([_bucket_type(str(gtype[gi])) for gi in g], dtype=object)
        for t in ("offline", "sync", "async"):
            tm = types == t
            if not tm.any():
                continue
            ms = compute_metrics(p[tm], y[tm])
            sr = {
                "config": label,
                "diff_w_miss_power": miss_power,
                "diff_w_take_boost": take_boost,
                "diff_w_solo_only": solo_only,
                "n_extra_epochs": n_extra_epochs,
                "slice": t,
                "n_holdout": int(tm.sum()),
                "train_avg_ll": row["train_avg_ll"],
                "logloss": round(float(ms["logloss"]), 6),
                "brier": round(float(ms["brier"]), 6),
                "auc": round(float(ms["auc"]), 6) if not np.isnan(ms["auc"]) else "",
                "elapsed_sec": row["elapsed_sec"],
                "solo_holdout_ll": "",
                "solo_holdout_n": "",
            }
            rows.append(sr)

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_difficulty_weights_sweep.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="skip convergence (1-pass) re-runs",
    )
    args = ap.parse_args()

    trials = [
        ("baseline", 0.0, 0.0, True),
        ("miss_a03", 0.3, 0.0, True),
        ("miss_a05", 0.5, 0.0, True),
        ("miss_a10", 1.0, 0.0, True),
        ("take_b05", 0.0, 0.5, True),
        ("take_b10", 0.0, 1.0, True),
        ("both_a05_b10", 0.5, 1.0, True),
        ("both_a10_b10_all", 1.0, 1.0, False),
    ]

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)
    all_pids = {int(p) for p in maps.idx_to_player_id}
    case_ids = {k: v for k, v in CASE_STUDY.items() if k in all_pids}

    rows: list[dict] = []
    for label, miss_p, take_b, solo_only in trials:
        rows.extend(
            _run_trial(
                arrays,
                maps,
                label=label,
                miss_power=miss_p,
                take_boost=take_b,
                solo_only=solo_only,
                n_extra_epochs=1,
                holdout=args.holdout,
                seed=args.seed,
                case_ids=case_ids,
            )
        )

    if not args.quick:
        print("\n[convergence] pass-1 only (n_extra_epochs=0) for baseline + best solo", flush=True)
        for label, miss_p, take_b, solo_only in [
            ("baseline_1pass", 0.0, 0.0, True),
            ("both_a05_b10_1pass", 0.5, 1.0, True),
        ]:
            rows.extend(
                _run_trial(
                    arrays,
                    maps,
                    label=label,
                    miss_power=miss_p,
                    take_boost=take_b,
                    solo_only=solo_only,
                    n_extra_epochs=0,
                    holdout=args.holdout,
                    seed=args.seed,
                    case_ids=case_ids,
                )
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[wrote] {out_path} ({len(rows)} rows)", flush=True)
    best = min(
        (r for r in rows if r.get("slice") == "all" and r.get("n_extra_epochs") == 1),
        key=lambda r: float(r["logloss"]),
    )
    base = next(
        r for r in rows
        if r["config"] == "baseline" and r.get("slice") == "all" and r.get("n_extra_epochs") == 1
    )
    print(
        f"\nBest holdout: {best['config']} ll={best['logloss']:.4f} "
        f"(Δ vs baseline {float(best['logloss']) - float(base['logloss']):+.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
