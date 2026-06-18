"""Sweep δ_size knobs for small-team (1–3) bias, especially offline.

Honest cell-holdout (default 10 %, seed 42).  Reports logloss and
calibration bias (mean actual − predicted) on team sizes 1, 2, 3, 4, 6
and the offline∩size≤3 cross-slice.

Outputs ``results/exp_small_teams_offline.csv``.

Usage::

    python scripts/exp_small_teams_offline.py --cache_file data.npz

Cost: ~10 trials × ~7 min ≈ 70 min.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential

SIZE_SLICES = (1, 2, 3, 4, 6)
EXTRA_SLICES = ("offline_small_1_3",)


def _bucket_type(g: str) -> str:
    if "async" in g:
        return "async"
    if "sync" in g:
        return "sync"
    return "offline"


def _bias(p: np.ndarray, y: np.ndarray) -> float:
    """Mean(actual − predicted) in probability units."""
    return float((y.astype(np.float64) - p).mean())


def _variant_specs() -> list[tuple[str, dict]]:
    """Pragmatic grid: baseline + single-knob moves + one combo + anchor."""
    return [
        ("baseline", {}),
        ("eta_size_0.003", {"eta_size": 0.003}),
        ("eta_size_0.005", {"eta_size": 0.005}),
        ("eta_size_0.01", {"eta_size": 0.01}),
        ("w_size_offline_2.0", {"w_size_offline": 2.0}),
        ("w_size_offline_3.0", {"w_size_offline": 3.0}),
        ("reg_size_0.05", {"reg_size": 0.05}),
        ("reg_size_0.10", {"reg_size": 0.10}),
        (
            "eta0.005_w_off2",
            {"eta_size": 0.005, "w_size_offline": 2.0},
        ),
        ("anchor_4", {"team_size_anchor": 4}),
        (
            "delta_seed_small",
            {
                "eta_size": 0.005,
                "delta_size_init": {1: -0.25, 2: -0.10, 3: -0.05},
            },
        ),
    ]


def _run_variant(
    arrays: dict,
    maps,
    team_sizes: np.ndarray,
    *,
    holdout: float,
    seed: int,
    name: str,
    overrides: dict,
) -> tuple[list[dict], float]:
    cfg_kw = {
        "holdout_obs_fraction": holdout,
        "holdout_seed": seed,
        **overrides,
    }
    cfg = Config(**cfg_kw)

    t0 = time.time()
    result = run_sequential(
        arrays, maps, cfg, verbose=False, collect_predictions=True
    )
    elapsed = time.time() - t0

    pred = result.predictions
    mask = pred["is_holdout"].astype(bool)
    p = pred["pred_p"][mask]
    y = pred["actual_y"][mask]
    g = pred["game_idx"][mask]
    obs_idx = pred["obs_idx"][mask]
    ts = team_sizes[obs_idx]

    gtype = getattr(maps, "game_type", None)
    if gtype is not None:
        types = np.array(
            [_bucket_type(str(gtype[int(gi)])) for gi in g], dtype=object
        )
    else:
        types = np.array(["offline"] * len(g), dtype=object)

    ds = (
        np.asarray(result.delta_size).round(4).tolist()
        if result.delta_size is not None
        else []
    )
    base_row = {
        "variant": name,
        "eta_size": cfg.eta_size,
        "w_size_offline": cfg.w_size_offline,
        "reg_size": cfg.reg_size,
        "team_size_anchor": cfg.team_size_anchor,
        "delta_size_init": json.dumps(cfg.delta_size_init or ""),
        "delta_size": json.dumps(ds),
        "elapsed_sec": round(elapsed, 1),
    }

    rows: list[dict] = []

    def _append(slice_name: str, sl: np.ndarray) -> None:
        if not sl.any():
            return
        m = compute_metrics(p[sl], y[sl])
        rows.append({
            **base_row,
            "slice": slice_name,
            "n": int(sl.sum()),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "bias_pp": round(100.0 * _bias(p[sl], y[sl]), 4),
            "auc": (
                round(float(m["auc"]), 6)
                if not np.isnan(m["auc"]) else ""
            ),
        })

    _append("all", np.ones(len(p), dtype=bool))
    for n in SIZE_SLICES:
        _append(f"size_{n}", ts == n)
    offline_small = (types == "offline") & (ts <= 3)
    _append("offline_small_1_3", offline_small)
    for t in ("offline", "sync", "async"):
        _append(t, types == t)

    return rows, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_small_teams_offline.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)
    team_sizes = arrays["team_sizes"].astype(np.int32)

    all_rows: list[dict] = []
    for name, overrides in _variant_specs():
        ov = dict(overrides)
        print(f"\n=== {name}  {ov} ===", flush=True)
        rows, elapsed = _run_variant(
            arrays,
            maps,
            team_sizes,
            holdout=args.holdout,
            seed=args.seed,
            name=name,
            overrides=ov,
        )
        all_rows.extend(rows)
        overall = next(r for r in rows if r["slice"] == "all")
        print(
            f"  overall: ll={overall['logloss']:.4f}  "
            f"bias={overall['bias_pp']:+.2f}pp  ({elapsed:.1f}s)",
            flush=True,
        )
        for n in SIZE_SLICES:
            sub = [r for r in rows if r["slice"] == f"size_{n}"]
            if sub:
                r = sub[0]
                print(
                    f"  size {n}: ll={r['logloss']:.4f}  "
                    f"bias={r['bias_pp']:+.2f}pp",
                    flush=True,
                )
        off_sm = [r for r in rows if r["slice"] == "offline_small_1_3"]
        if off_sm:
            r = off_sm[0]
            print(
                f"  offline 1-3: ll={r['logloss']:.4f}  "
                f"bias={r['bias_pp']:+.2f}pp",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in all_rows for k in r})
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print("\n=== Ranked overall logloss ===", flush=True)
    overall_rows = [r for r in all_rows if r["slice"] == "all"]
    overall_rows.sort(key=lambda r: r["logloss"])
    for r in overall_rows:
        star = "  ★" if r is overall_rows[0] else ""
        print(
            f"  {r['variant']:22s} ll={r['logloss']:.4f}  "
            f"bias={r['bias_pp']:+.2f}pp{star}",
            flush=True,
        )

    print("\n=== offline_small_1_3 bias (pp) ===", flush=True)
    off_rows = [r for r in all_rows if r["slice"] == "offline_small_1_3"]
    off_rows.sort(key=lambda r: abs(r["bias_pp"]))
    for r in off_rows:
        star = "  ★" if r is off_rows[0] else ""
        print(
            f"  {r['variant']:22s} bias={r['bias_pp']:+.2f}pp  "
            f"ll={r['logloss']:.4f}{star}",
            flush=True,
        )

    print(f"\n[ok] {len(all_rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
