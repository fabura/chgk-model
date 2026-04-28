"""Compare init-from-take-rate variants on the backtest split.

Sweeps a few combinations of:

* ``init_laplace_alpha`` ∈ {0.0, 0.5, 1.0}    — Beta(α,α) shrinkage
                                                of observed take rate
* ``(init_b_clip_lo, init_b_clip_hi)``        — hard clamp on init b

Variants tried:

    baseline   : α=0, clip=(-10, +10)        # current production behaviour
    lap1       : α=1, clip=(-10, +10)        # classical Laplace only
    clip_tight : α=0, clip=(-3,  +6)         # hard-clip only
    both       : α=1, clip=(-3,  +6)         # combined defensive init
    lap0.5     : α=0.5, clip=(-10, +10)      # half-strength Laplace
    both0.5    : α=0.5, clip=(-3,  +6)

For each variant we run one full backtest, capturing global, per-mode
and per-hardness metrics. Output to ``results/exp_init_robust.csv``.

Heads-up: at K=0 these all run as one chronological pass (≈ 8 min on
data.npz, single CPU). 6 variants ≈ 50 min wall total.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from data import load_cached
from rating.backtest import backtest
from rating.engine import Config


VARIANTS = [
    ("baseline",   {"init_laplace_alpha": 0.0, "init_b_clip_lo": -10.0, "init_b_clip_hi": 10.0}),
    ("lap1",       {"init_laplace_alpha": 1.0, "init_b_clip_lo": -10.0, "init_b_clip_hi": 10.0}),
    ("clip_tight", {"init_laplace_alpha": 0.0, "init_b_clip_lo": -3.0,  "init_b_clip_hi": 6.0}),
    ("both",       {"init_laplace_alpha": 1.0, "init_b_clip_lo": -3.0,  "init_b_clip_hi": 6.0}),
    ("lap0.5",     {"init_laplace_alpha": 0.5, "init_b_clip_lo": -10.0, "init_b_clip_hi": 10.0}),
    ("both0.5",    {"init_laplace_alpha": 0.5, "init_b_clip_lo": -3.0,  "init_b_clip_hi": 6.0}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out_csv", default="results/exp_init_robust.csv")
    ap.add_argument("--variants", default=",".join(v[0] for v in VARIANTS),
                    help="comma-separated subset of variant names to run")
    args = ap.parse_args()

    selected = set(args.variants.split(","))
    todo = [(n, kw) for (n, kw) in VARIANTS if n in selected]
    if not todo:
        raise SystemExit(f"no variants matched {selected!r}")

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, kw in todo:
        cfg = Config(**kw)
        print(f"\n=== {name}  cfg={kw} ===")
        t0 = time.time()
        m = backtest(arrays, maps, cfg, verbose=False)
        elapsed = time.time() - t0

        row = {
            "variant": name,
            **kw,
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
            row[f"n_{t}"] = sub.get("n_obs", 0)
        for q in (1, 2, 3, 4):
            sub = m.get("by_hardness", {}).get(f"q{q}", {})
            row[f"ll_q{q}"] = sub.get("logloss", float("nan"))
            row[f"thbar_q{q}"] = sub.get("mean_team_theta", float("nan"))

        # Parameter diagnostics: how many init-clip events?
        result = m["result"]
        b = result.questions.b
        init_q = result.questions.initialized
        b_init = b[init_q]
        n_q = int(init_q.sum())
        b_at_hi = int((b_init >= float(kw["init_b_clip_hi"]) - 1e-6).sum())
        b_at_lo = int((b_init <= float(kw["init_b_clip_lo"]) + 1e-6).sum())
        row["n_init_q"] = n_q
        row["b_at_hi_clip"] = b_at_hi
        row["b_at_lo_clip"] = b_at_lo
        row["b_mean"] = float(b_init.mean())
        row["b_std"] = float(b_init.std())

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
