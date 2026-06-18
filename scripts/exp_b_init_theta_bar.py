"""θ̄-aware vs noisy-OR-only b_init on honest cell hold-out.

Mid-b quintile (q3) is the suspected pain point: θ̄-aware init may be
less informative when field strength is near population average.

Variants (all use ``holdout_obs_fraction=0.10``, ``holdout_seed=42``):

* ``baseline``           — production defaults
* ``no_theta_bar``       — noisy-OR init only (Round 1)
* ``theta_bar_g10``      — θ̄ from players with ≥10 games
* ``theta_bar_g20``      — θ̄ from players with ≥20 games
* ``hybrid_teams5``      — θ̄ only when ≥5 teams contribute
* ``hybrid_teams10``     — θ̄ only when ≥10 teams contribute

Outputs ``results/exp_b_init_theta_bar.csv`` with overall, per-mode,
per-b-quintile, and offline-strong-field (hardness q4) slices.

Usage::

    .venv/bin/python scripts/exp_b_init_theta_bar.py
    .venv/bin/python scripts/exp_b_init_theta_bar.py baseline no_theta_bar
    .venv/bin/python scripts/exp_b_init_theta_bar.py --list
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential

CACHE = ROOT / "data.npz"
OUT = ROOT / "results" / "exp_b_init_theta_bar.csv"


def _base(holdout: float, seed: int, **overrides) -> Config:
    cfg = Config(holdout_obs_fraction=holdout, holdout_seed=seed)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _build_configs(holdout: float, seed: int) -> dict[str, Config]:
    return {
        "baseline": _base(holdout, seed),
        "no_theta_bar": _base(holdout, seed, theta_bar_init=False),
        "theta_bar_g10": _base(holdout, seed, theta_bar_min_games=10),
        "theta_bar_g20": _base(holdout, seed, theta_bar_min_games=20),
        "hybrid_teams5": _base(holdout, seed, theta_bar_min_teams=5),
        "hybrid_teams10": _base(holdout, seed, theta_bar_min_teams=10),
    }


def _bucket_type(g: str) -> str:
    if "async" in g:
        return "async"
    if "sync" in g:
        return "sync"
    return "offline"


def _b_quintile_labels(b: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cuts = np.unique(np.quantile(b, [0.2, 0.4, 0.6, 0.8]))
    bucket = np.searchsorted(cuts, b, side="right")
    return bucket, [float(c) for c in cuts]


def _hardness_quartile(
    thbar: np.ndarray, game_idx: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    unique_g, inv = np.unique(game_idx, return_inverse=True)
    sums = np.zeros(len(unique_g), dtype=np.float64)
    cnts = np.zeros(len(unique_g), dtype=np.int64)
    np.add.at(sums, inv, thbar)
    np.add.at(cnts, inv, 1)
    per_g = sums / np.maximum(cnts, 1)
    cuts = np.unique(np.quantile(per_g, [0.25, 0.5, 0.75]))
    g_to_q = np.searchsorted(cuts, per_g, side="right") + 1
    return g_to_q[inv], [float(c) for c in cuts]


def _slice_metrics(
    p: np.ndarray, y: np.ndarray, mask: np.ndarray,
) -> dict[str, float | int]:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "logloss": float("nan"), "brier": float("nan"), "auc": float("nan")}
    m = compute_metrics(p[mask], y[mask])
    return {
        "n": n,
        "logloss": round(float(m["logloss"]), 6),
        "brier": round(float(m["brier"]), 6),
        "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else float("nan"),
    }


FIELDNAMES = [
    "variant", "slice", "n", "logloss", "brier", "auc", "elapsed_sec",
    "b_cuts", "hardness_cuts",
]


def run_variant(
    name: str,
    cfg: Config,
    arrays: dict[str, np.ndarray],
    maps,
) -> list[dict]:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    result = run_sequential(
        arrays, maps, cfg, verbose=False, collect_predictions=True,
    )
    elapsed = time.time() - t0

    pred = result.predictions
    if pred is None:
        raise RuntimeError(f"{name}: no predictions collected")

    holdout = pred["is_holdout"].astype(bool)
    p = pred["pred_p"][holdout]
    y = pred["actual_y"][holdout]
    g = pred["game_idx"][holdout]
    thbar = pred.get("team_theta_mean")
    thbar_h = thbar[holdout] if thbar is not None else None

    # b at canonical question index (post-train; diagnostic bucketing)
    q_obs = arrays["q_idx"][pred["obs_idx"][holdout]]
    cq = (
        maps.canonical_q_idx
        if maps.canonical_q_idx is not None
        else np.arange(maps.num_questions, dtype=np.int32)
    )
    b_h = result.questions.b[cq[q_obs]]

    rows: list[dict] = []
    b_cuts: list[float] = []
    h_cuts: list[float] = []

    def _row(slice_name: str, mask: np.ndarray) -> dict:
        m = _slice_metrics(p, y, mask)
        return {
            "variant": name,
            "slice": slice_name,
            "n": m["n"],
            "logloss": m["logloss"],
            "brier": m["brier"],
            "auc": m["auc"],
            "elapsed_sec": round(elapsed, 1) if slice_name == "all" else "",
            "b_cuts": " | ".join(f"{c:+.3f}" for c in b_cuts) if b_cuts else "",
            "hardness_cuts": (
                " | ".join(f"{c:+.3f}" for c in h_cuts) if h_cuts else ""
            ),
        }

    all_mask = np.ones(len(p), dtype=bool)
    rows.append(_row("all", all_mask))
    print(
        f"  all     : n={rows[-1]['n']:>9d}  ll={rows[-1]['logloss']:.4f}  "
        f"AUC={rows[-1]['auc']:.4f}  ({elapsed:.1f}s)",
        flush=True,
    )

    gtype = getattr(maps, "game_type", None)
    if gtype is not None:
        types = np.array(
            [_bucket_type(str(gtype[gi])) for gi in g], dtype=object,
        )
        for t in ("offline", "sync", "async"):
            tm = types == t
            if not tm.any():
                continue
            rows.append(_row(t, tm))
            print(
                f"  {t:7s}: n={rows[-1]['n']:>9d}  ll={rows[-1]['logloss']:.4f}",
                flush=True,
            )

    b_bucket, b_cuts = _b_quintile_labels(b_h)
    for q in range(5):
        mask = b_bucket == q
        if not mask.any():
            continue
        rows.append(_row(f"b_q{q + 1}", mask))
        print(
            f"  b_q{q + 1}   : n={rows[-1]['n']:>9d}  ll={rows[-1]['logloss']:.4f}  "
            f"(b cuts: {' | '.join(f'{c:+.2f}' for c in b_cuts)})",
            flush=True,
        )

    if thbar_h is not None:
        h_q, h_cuts = _hardness_quartile(thbar_h, g)
        for q in (1, 2, 3, 4):
            mask = h_q == q
            if not mask.any():
                continue
            rows.append(_row(f"field_q{q}", mask))
        # offline × strongest field (q4)
        if gtype is not None:
            offline = types == "offline"
            strong = h_q == 4
            mask = offline & strong
            if mask.any():
                rows.append(_row("offline_field_q4", mask))
                print(
                    f"  off_q4  : n={rows[-1]['n']:>9d}  "
                    f"ll={rows[-1]['logloss']:.4f}",
                    flush=True,
                )

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_file", default=str(CACHE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "variants", nargs="*",
        help="subset of variant names; default = all",
    )
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    configs = _build_configs(args.holdout, args.seed)
    if args.list:
        for n in configs:
            print(n)
        return 0

    wanted = args.variants or list(configs)
    unknown = [v for v in wanted if v not in configs]
    if unknown:
        print(f"unknown variants: {unknown}.  Use --list.")
        return 2
    configs = {k: v for k, v in configs.items() if k in wanted}

    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)
    print(
        f"[holdout] fraction={args.holdout}, seed={args.seed}, "
        f"variants={list(configs)}",
        flush=True,
    )

    all_rows: list[dict] = []
    for name, cfg in configs.items():
        all_rows.extend(run_variant(name, cfg, arrays, maps))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[ok] {len(all_rows)} rows → {out_path}", flush=True)

    if "baseline" in {r["variant"] for r in all_rows}:
        print("\n=== Δ logloss vs baseline (positive = worse) ===", flush=True)
        base = {
            (r["variant"], r["slice"]): float(r["logloss"])
            for r in all_rows if r["variant"] == "baseline"
        }
        print(f"{'variant':>18s}  {'slice':>18s}  {'logloss':>9s}  {'Δll':>9s}", flush=True)
        for r in all_rows:
            key = ("baseline", r["slice"])
            if key not in base or r["variant"] == "baseline":
                continue
            d = float(r["logloss"]) - base[key]
            print(
                f"{r['variant']:>18s}  {r['slice']:>18s}  "
                f"{float(r['logloss']):>9.4f}  {d:>+9.4f}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
