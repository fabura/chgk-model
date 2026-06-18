#!/usr/bin/env python3
"""Honest cell-holdout sweep for ``Config.min_eta`` (η floor for veterans).

η_k = max(min_eta, η0 / √(games_offset + games_k))

Usage:
    python scripts/exp_min_eta_step.py
    python scripts/exp_min_eta_step.py --values 0 0.005 0.01 0.02
    python scripts/exp_min_eta_step.py --quick   # baseline + 0.01 only
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import load_cached
from rating.backtest import backtest
from rating.engine import Config, run_sequential

CACHE = os.path.join(ROOT, "data.npz")
OUT = os.path.join(ROOT, "results", "exp_min_eta_step.csv")

WATCH_PIDS = {
    131922: "Васильев Дмитрий",
    23737: "Островский Андрей",
}


def _player_ranks(theta, maps) -> dict[int, int]:
    order = sorted(
        range(maps.num_players),
        key=lambda i: (-float(theta[i]), -int(maps.idx_to_player_id[i])),
    )
    return {
        int(maps.idx_to_player_id[i]): r + 1
        for r, i in enumerate(order)
        if i < len(maps.idx_to_player_id)
    }


def watch_player_stats(arrays, maps, min_eta: float) -> dict[str, object]:
    cfg = Config(
        holdout_obs_fraction=0.10,
        holdout_seed=42,
        min_eta=min_eta,
    )
    result = run_sequential(arrays, maps, cfg, verbose=False)
    ranks = _player_ranks(result.players.theta, maps)
    out: dict[str, object] = {}
    for pid, label in WATCH_PIDS.items():
        pidx = maps.player_id_to_idx.get(pid)
        if pidx is None:
            continue
        out[f"theta_{pid}"] = round(float(result.players.theta[pidx]), 6)
        out[f"rank_{pid}"] = ranks.get(pid)
        out[f"name_{pid}"] = label
    return out


def run_variant(min_eta: float, arrays, maps) -> dict:
    cfg = Config(
        holdout_obs_fraction=0.10,
        holdout_seed=42,
        min_eta=min_eta,
    )
    t0 = time.time()
    metrics = backtest(arrays, maps, cfg, verbose=False)
    elapsed = time.time() - t0
    row = {
        "min_eta": min_eta,
        "logloss": round(metrics["logloss"], 6),
        "brier": round(metrics["brier"], 6),
        "auc": round(metrics["auc"], 6),
        "n_test_obs": metrics.get("n_test_obs", 0),
        "elapsed_sec": round(elapsed, 1),
    }
    by_type = metrics.get("by_type", {})
    for t in ("offline", "sync", "async"):
        m = by_type.get(t, {})
        if m:
            row[f"logloss_{t}"] = round(m["logloss"], 6)
            row[f"auc_{t}"] = round(m.get("auc", float("nan")), 6)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--values",
        nargs="+",
        type=float,
        default=[0.0, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02],
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument(
        "--watch-players",
        action="store_true",
        help="After the sweep, export final θ/rank for watch-list players "
        "(baseline + highest min_eta only — two extra full passes)",
    )
    args = ap.parse_args()

    values = [0.0, 0.01] if args.quick else args.values
    print(f"Loading {CACHE} …", flush=True)
    arrays, maps = load_cached(CACHE)
    print(
        f"  {len(arrays['q_idx'])} obs, {maps.num_players} players",
        flush=True,
    )

    rows: list[dict] = []
    for i, v in enumerate(values):
        print(f"\n[{i+1}/{len(values)}] min_eta={v}", flush=True)
        row = run_variant(v, arrays, maps)
        rows.append(row)
        print(
            f"  logloss={row['logloss']:.4f}  auc={row['auc']:.4f}  "
            f"({row['elapsed_sec']}s)",
            flush=True,
        )

    if args.watch_players and values:
        for v in (values[0], values[-1]):
            print(f"\nwatch-players pass min_eta={v} …", flush=True)
            extra = watch_player_stats(arrays, maps, v)
            target = rows[0] if v == values[0] else rows[-1]
            target.update(extra)
            for pid, label in WATCH_PIDS.items():
                th = target.get(f"theta_{pid}")
                rk = target.get(f"rank_{pid}")
                if th is not None:
                    print(f"    {label}: θ={th:+.4f} rank={rk}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}", flush=True)

    base = rows[0]
    print("\n=== vs baseline (min_eta=0) ===")
    for r in rows[1:]:
        dll = r["logloss"] - base["logloss"]
        dauc = r["auc"] - base["auc"]
        print(
            f"  min_eta={r['min_eta']:.4f}: Δlogloss={dll:+.6f}  Δauc={dauc:+.6f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
