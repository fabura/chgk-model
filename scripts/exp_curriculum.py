"""Curriculum filter sweep: skip SGD on "lottery" questions.

Hypothesis (docs/error_structure_2026-04.md): questions with very few
takes (or very few misses) supply rare, high-magnitude gradient kicks
to θ that are mostly luck-driven, polluting θ estimates and
indirectly worsening predictions on mid-difficulty questions.

For each ``curriculum_min_events`` value N ∈ {0, 1, 3, 5, 10} we run
a backtest where any observation whose canonical question has
``min(takes_q, n_q − takes_q) < N`` is excluded from SGD updates
(but kept for init and for prediction).

Output: results/exp_curriculum.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from data import load_cached
from rating.backtest import backtest
from rating.engine import Config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out_csv", default="results/exp_curriculum.csv")
    ap.add_argument("--values", default="0,1,3,5,10")
    args = ap.parse_args()

    values = [int(v) for v in args.values.split(",")]

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for n in values:
        cfg = Config(curriculum_min_events=n)
        print(f"\n=== curriculum_min_events = {n} ===")
        t0 = time.time()
        m = backtest(arrays, maps, cfg, verbose=False)
        elapsed = time.time() - t0

        row = {
            "curriculum_min_events": n,
            "logloss": float(m["logloss"]),
            "brier": float(m["brier"]),
            "auc": float(m["auc"]),
            "elapsed_sec": round(elapsed, 1),
            "n_test_obs": int(m["n_test_obs"]),
        }
        for t in ("offline", "sync", "async"):
            sub = m.get("by_type", {}).get(t, {})
            row[f"ll_{t}"] = sub.get("logloss", float("nan"))
            row[f"auc_{t}"] = sub.get("auc", float("nan"))
        for q in (1, 2, 3, 4):
            sub = m.get("by_hardness", {}).get(f"q{q}", {})
            row[f"ll_q{q}"] = sub.get("logloss", float("nan"))

        # Bonus diagnostic: the model's overall final b/θ stats.
        result = m["result"]
        b = result.questions.b[result.questions.initialized]
        th = result.players.theta[result.players.seen]
        row["b_mean"] = float(b.mean())
        row["b_std"] = float(b.std())
        row["theta_mean"] = float(th.mean())
        row["theta_std"] = float(th.std())

        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] → {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
