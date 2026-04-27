"""Multi-epoch warm-start refit experiment.

For each value of ``n_extra_epochs ∈ N`` (default ``0,1,2,4``), runs
``backtest()`` with ``Config(n_extra_epochs=N)`` from scratch and
records:

* overall and per-format backtest metrics on the 20 % time tail;
* snapshots of θ, b, log a, δ_size, δ_pos.

Finally, prints a parameter-drift summary: how far each set of params
travelled from the baseline (n_extra=0) state, and the top-K players
and questions by |Δθ|, |Δb|.

Outputs:
    results/exp_multi_epoch.csv
    results/exp_multi_epoch_drift.csv
    results/exp_multi_epoch_top_player_drifts.csv
    results/exp_multi_epoch_top_question_drifts.csv

Each backtest is one independent training run from cold init, so the
total wall time is roughly ``Σ(1 + N) × single_pass_time``.  With
single pass ≈ 4 min on cache, the ``0,1,2,4`` budget is ≈ 32 min.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from data import load_cached
from rating.backtest import backtest
from rating.engine import Config


def _try_open_duckdb(path: str):
    try:
        import duckdb

        return duckdb.connect(path, read_only=True)
    except Exception as e:
        print(f"  (no DuckDB: {e})")
        return None


def _player_names(con, ids: list[int]) -> dict[int, str]:
    if con is None or not ids:
        return {}
    rows = con.execute(
        "SELECT player_id, COALESCE(last_name,'') || ' ' || "
        "COALESCE(first_name,'') AS name "
        "FROM players WHERE player_id = ANY(?)",
        [ids],
    ).fetchall()
    return {r[0]: r[1].strip() for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument(
        "--epochs", default="0,1,2,4",
        help="comma-separated list of n_extra_epochs values to evaluate",
    )
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    args = ap.parse_args()

    epochs_list = [int(x) for x in args.epochs.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)

    # Snapshots keyed by n_extra value.
    snapshots: dict[int, dict] = {}
    rows: list[dict] = []

    for n_extra in epochs_list:
        cfg = Config(n_extra_epochs=n_extra)
        print(f"\n=== n_extra_epochs = {n_extra} ===")
        t0 = time.time()
        m = backtest(arrays, maps, cfg, verbose=False)
        elapsed = time.time() - t0

        result = m["result"]
        # Snapshot parameters.
        snapshots[n_extra] = {
            "theta": result.players.theta.copy(),
            "games": result.players.games.copy(),
            "b": result.questions.b.copy(),
            "log_a": result.questions.log_a.copy(),
            "delta_size": (
                result.delta_size.copy()
                if result.delta_size is not None else None
            ),
            "delta_pos": (
                result.delta_pos.copy()
                if result.delta_pos is not None else None
            ),
            "q_initialized": result.questions.initialized.copy(),
            "p_seen": result.players.seen.copy(),
        }

        row = {
            "n_extra_epochs": n_extra,
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
        rows.append(row)

        print(json.dumps(row, ensure_ascii=False, indent=2))

    out_csv = out / "exp_multi_epoch.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] metrics → {out_csv}")

    # === Parameter drift relative to n_extra=0 baseline =============
    if 0 not in snapshots:
        print("[drift] no n_extra=0 baseline; skipping drift analysis")
        return 0

    base = snapshots[0]
    drift_rows = []
    for n_extra in epochs_list:
        s = snapshots[n_extra]
        # Compare on parameters that exist in both (player seen by
        # base and question initialized by base).
        seen_p = base["p_seen"] & s["p_seen"]
        init_q = base["q_initialized"] & s["q_initialized"]
        d_theta = s["theta"][seen_p] - base["theta"][seen_p]
        d_b = s["b"][init_q] - base["b"][init_q]
        d_la = s["log_a"][init_q] - base["log_a"][init_q]
        drift_rows.append({
            "n_extra_epochs": n_extra,
            "n_players": int(seen_p.sum()),
            "n_questions": int(init_q.sum()),
            "theta_rms": float(np.sqrt(np.mean(d_theta ** 2))),
            "theta_max_abs": float(np.max(np.abs(d_theta))) if len(d_theta) else 0.0,
            "theta_mean": float(d_theta.mean()) if len(d_theta) else 0.0,
            "b_rms": float(np.sqrt(np.mean(d_b ** 2))),
            "b_max_abs": float(np.max(np.abs(d_b))) if len(d_b) else 0.0,
            "log_a_rms": float(np.sqrt(np.mean(d_la ** 2))),
            "log_a_max_abs": float(np.max(np.abs(d_la))) if len(d_la) else 0.0,
        })
    drift_csv = out / "exp_multi_epoch_drift.csv"
    with open(drift_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=drift_rows[0].keys())
        w.writeheader()
        w.writerows(drift_rows)
    print(f"[ok] drift → {drift_csv}")

    # === Top-K player and question drifts (largest n_extra vs base) ==
    largest = max(epochs_list)
    if largest > 0:
        s = snapshots[largest]
        seen_p = base["p_seen"] & s["p_seen"]
        d_theta = s["theta"] - base["theta"]
        d_theta[~seen_p] = 0.0

        idx_sorted = np.argsort(-np.abs(d_theta))
        top_idx = idx_sorted[:args.top_k * 2]  # pad in case some lack ids
        ids = [
            int(maps.idx_to_player_id[i])
            if i < len(maps.idx_to_player_id) else int(i)
            for i in top_idx
        ]
        con = _try_open_duckdb(args.duckdb)
        names = _player_names(con, ids)
        rows_pl = []
        for i, pid in zip(top_idx, ids):
            if not seen_p[i]:
                continue
            rows_pl.append({
                "player_id": pid,
                "name": names.get(pid, ""),
                "games": int(base["games"][i]),
                "theta_base": float(base["theta"][i]),
                "theta_new": float(s["theta"][i]),
                "delta": float(d_theta[i]),
            })
            if len(rows_pl) >= args.top_k:
                break
        ppath = out / "exp_multi_epoch_top_player_drifts.csv"
        with open(ppath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows_pl[0].keys())
            w.writeheader()
            w.writerows(rows_pl)
        print(f"[ok] top player drifts → {ppath}")

        # Top-K question drifts.
        init_q = base["q_initialized"] & s["q_initialized"]
        d_b = s["b"] - base["b"]
        d_b[~init_q] = 0.0
        idx_q = np.argsort(-np.abs(d_b))[:args.top_k]
        rows_q = []
        for i in idx_q:
            if not init_q[i]:
                continue
            rows_q.append({
                "canonical_q_idx": int(i),
                "b_base": float(base["b"][i]),
                "b_new": float(s["b"][i]),
                "delta_b": float(d_b[i]),
                "log_a_base": float(base["log_a"][i]),
                "log_a_new": float(s["log_a"][i]),
            })
        qpath = out / "exp_multi_epoch_top_question_drifts.csv"
        with open(qpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows_q[0].keys())
            w.writeheader()
            w.writerows(rows_q)
        print(f"[ok] top question drifts → {qpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
