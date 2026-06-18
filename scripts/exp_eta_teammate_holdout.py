"""Sweep ``eta_teammate`` on honest cell-holdout (extended range).

Re-validates the 2026-05 bump (0.005→0.02) and tests higher values
(0.05–0.20) after the June 2026 floor-player cycle rejected other
fixes.  Records overall / offline / stable-roster (Jaccard≥0.85) slices
and final θ for three case-study floor players.

Outputs ``results/exp_eta_teammate_holdout.csv``.

Usage::

    python scripts/exp_eta_teammate_holdout.py --cache_file data.npz

Cost: ~7 trials × ~7 min ≈ 50 min.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data import load_cached
from rating.backtest import compute_metrics
from rating.engine import Config, run_sequential

CASE_STUDY = {
    34909: "Chernukha",
    26818: "Rekshinskaya",
    158668: "Monina",
}

DEFAULT_VALUES = "0.0,0.02,0.05,0.08,0.10,0.15,0.20"


def _build_offsets(team_sizes: np.ndarray) -> np.ndarray:
    off = np.empty(len(team_sizes) + 1, dtype=np.int64)
    off[0] = 0
    np.cumsum(team_sizes.astype(np.int64), out=off[1:])
    return off


def _holdout_mask(n_obs: int, fraction: float, seed: int) -> np.ndarray:
    """Same RNG draw as ``run_sequential`` cell-holdout."""
    if fraction <= 0.0:
        return np.zeros(n_obs, dtype=bool)
    rng = np.random.default_rng(seed)
    return rng.random(n_obs) < fraction


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
        if idx is None:
            out[f"theta_{label}"] = float("nan")
        else:
            out[f"theta_{label}"] = round(float(result.players.theta[idx]), 4)
    return out


def _load_roster_stability(duckdb_path: Path) -> tuple[dict, dict]:
    """Return (roster_team, team_jaccard) from baked DuckDB."""
    roster_team: dict[tuple[int, tuple[int, ...]], int] = {}
    team_jacc: dict[int, float] = {}
    if not duckdb_path.is_file():
        print(f"[warn] DuckDB not found: {duckdb_path}", flush=True)
        return roster_team, team_jacc
    try:
        import duckdb

        con = duckdb.connect(str(duckdb_path), read_only=True)
        tr = con.execute(
            """
            SELECT pg.tournament_id, pg.team_id,
                   list(pg.player_id ORDER BY pg.player_id) AS pids
            FROM player_games pg
            GROUP BY pg.tournament_id, pg.team_id
            """
        ).fetchall()
        for tid, team_id, pids in tr:
            roster_team[(int(tid), tuple(int(x) for x in pids))] = int(team_id)

        hist = con.execute(
            """
            SELECT pg.team_id, pg.tournament_id, t.start_date,
                   list(pg.player_id ORDER BY pg.player_id) AS pids
            FROM player_games pg
            JOIN tournaments t ON t.tournament_id = pg.tournament_id
            WHERE pg.team_id IS NOT NULL
            GROUP BY pg.team_id, pg.tournament_id, t.start_date
            ORDER BY pg.team_id, t.start_date
            """
        ).fetchall()
        prev: dict[int, set[int]] = {}
        sums: dict[int, list[float]] = defaultdict(list)
        for team_id, _tid, _dt, pids in hist:
            cur = set(int(x) for x in pids)
            if team_id in prev:
                inter = len(prev[team_id] & cur)
                union = len(prev[team_id] | cur)
                if union > 0:
                    sums[int(team_id)].append(inter / union)
            prev[int(team_id)] = cur
        for tid, vals in sums.items():
            team_jacc[tid] = float(np.mean(vals)) if vals else 1.0
        con.close()
        print(
            f"[aux] roster_team={len(roster_team):,}  "
            f"team_jaccard={len(team_jacc):,}",
            flush=True,
        )
    except Exception as exc:
        print(f"[warn] DuckDB roster stability unavailable: {exc}", flush=True)
    return roster_team, team_jacc


def _stable_roster_mask_for_holdout(
    maps,
    arrays: dict[str, np.ndarray],
    roster_team: dict[tuple[int, tuple[int, ...]], int],
    team_jacc: dict[int, float],
    holdout_mask: np.ndarray,
    *,
    jaccard_min: float = 0.85,
) -> np.ndarray | None:
    """Boolean mask over holdout observations (aligned with pred[holdout])."""
    if not roster_team or not team_jacc:
        return None

    offsets = _build_offsets(arrays["team_sizes"])
    player_flat = arrays["player_indices_flat"]
    idx_to_tid = maps.idx_to_game_id
    game_idx = arrays["game_idx"]
    pid_to_idx = maps.player_id_to_idx

    roster_team_idx: dict[tuple[int, tuple[int, ...]], int] = {}
    for (tid, pids), team_id in roster_team.items():
        idxs = tuple(sorted(pid_to_idx.get(pid, -1) for pid in pids))
        if idxs and idxs[0] >= 0:
            roster_team_idx[(tid, idxs)] = team_id

    holdout_obs = np.flatnonzero(holdout_mask)
    stable = np.zeros(len(holdout_obs), dtype=bool)

    group_obs: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for k, oi in enumerate(holdout_obs):
        tid = int(idx_to_tid[int(game_idx[oi])])
        s, e = int(offsets[oi]), int(offsets[oi + 1])
        key = (tid, tuple(sorted(int(pi) for pi in player_flat[s:e])))
        group_obs[key].append(k)

    for key, ks in group_obs.items():
        team_id = roster_team_idx.get(key)
        if team_id is None:
            continue
        j = team_jacc.get(team_id)
        if j is not None and j >= jaccard_min:
            stable[ks] = True

    return stable


def _slice_metrics(
    p: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int] | None:
    if not mask.any():
        return None
    m = compute_metrics(p[mask], y[mask])
    bias = float((y[mask].astype(np.float64) - p[mask]).mean())
    return {
        "n": int(mask.sum()),
        "logloss": round(float(m["logloss"]), 6),
        "brier": round(float(m["brier"]), 6),
        "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else float("nan"),
        "bias_pp": round(100.0 * bias, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_eta_teammate_holdout.csv")
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--values",
        default=DEFAULT_VALUES,
        help="comma-separated eta_teammate values",
    )
    ap.add_argument(
        "--jaccard-min",
        type=float,
        default=0.85,
        dest="jaccard_min",
        help="mean consecutive-roster Jaccard threshold for stable slice",
    )
    args = ap.parse_args()

    eta_values = [float(x) for x in args.values.split(",")]
    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)

    all_pids = {int(p) for p in maps.idx_to_player_id}
    case_ids = {k: v for k, v in CASE_STUDY.items() if k in all_pids}
    missing = set(CASE_STUDY) - set(case_ids)
    if missing:
        print(f"[warn] case-study ids not in cache: {missing}", flush=True)

    roster_team, team_jacc = _load_roster_stability(Path(args.duckdb))

    n_obs = len(arrays["q_idx"])
    holdout_mask = _holdout_mask(n_obs, args.holdout, args.seed)
    stable_mask = _stable_roster_mask_for_holdout(
        maps,
        arrays,
        roster_team,
        team_jacc,
        holdout_mask,
        jaccard_min=args.jaccard_min,
    )
    if stable_mask is not None:
        print(
            f"[aux] stable-roster holdout obs: "
            f"{int(stable_mask.sum()):,} / {int(holdout_mask.sum()):,}",
            flush=True,
        )

    rows: list[dict] = []

    for eta in eta_values:
        print(f"\n=== eta_teammate={eta:.4f} ===", flush=True)
        cfg = Config(
            eta_teammate=eta,
            holdout_obs_fraction=args.holdout,
            holdout_seed=args.seed,
        )
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

        m = compute_metrics(p, y)
        row: dict = {
            "eta_teammate": eta,
            "slice": "all",
            "n": int(mask.sum()),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
            "bias_pp": round(100.0 * float((y.astype(np.float64) - p).mean()), 4),
            "elapsed_sec": round(elapsed, 1),
        }
        row.update(_player_thetas(result, maps, case_ids))
        rows.append(row)
        print(
            f"  overall : ll={row['logloss']:.4f}  bias={row['bias_pp']:+.2f}pp  "
            f"AUC={row['auc']}  ({elapsed:.1f}s)",
            flush=True,
        )
        for label in CASE_STUDY.values():
            key = f"theta_{label}"
            if key in row and row[key] != "":
                print(f"  {key}={row[key]:+.4f}", flush=True)

        gtype = getattr(maps, "game_type", None)
        if gtype is not None:
            types = np.array(
                [_bucket_type(str(gtype[gi])) for gi in g], dtype=object
            )
            for t in ("offline", "sync", "async"):
                tm = types == t
                sm = _slice_metrics(p, y, tm)
                if sm is None:
                    continue
                slice_row = {
                    "eta_teammate": eta,
                    "slice": t,
                    "elapsed_sec": "",
                    **sm,
                }
                if np.isnan(slice_row.get("auc", 0.0)):
                    slice_row["auc"] = ""
                rows.append(slice_row)
                print(
                    f"  {t:7s}: ll={sm['logloss']:.4f}  "
                    f"bias={sm['bias_pp']:+.2f}pp  AUC={sm['auc']:.4f}",
                    flush=True,
                )

        if stable_mask is not None:
            sm = _slice_metrics(p, y, stable_mask)
            if sm is not None:
                slice_row = {
                    "eta_teammate": eta,
                    "slice": f"stable_jacc_ge{args.jaccard_min:.2f}",
                    "elapsed_sec": "",
                    **sm,
                }
                if np.isnan(slice_row.get("auc", 0.0)):
                    slice_row["auc"] = ""
                rows.append(slice_row)
                print(
                    f"  stable : ll={sm['logloss']:.4f}  "
                    f"bias={sm['bias_pp']:+.2f}pp  n={sm['n']:,}",
                    flush=True,
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    overall = [r for r in rows if r["slice"] == "all"]
    overall.sort(key=lambda r: r["logloss"])
    print("\n=== Ranked overall (best → worst) ===")
    for r in overall:
        marker = "  ★" if r is overall[0] else ""
        thetas = "  ".join(
            f"{k.replace('theta_', '')}={r[k]:+.3f}"
            for k in sorted(r)
            if k.startswith("theta_") and r[k] != ""
        )
        print(
            f"  eta={r['eta_teammate']:.4f}  ll={r['logloss']:.4f}  "
            f"bias={r['bias_pp']:+.2f}pp  {thetas}{marker}"
        )

    baseline = next((r for r in overall if r["eta_teammate"] == 0.02), None)
    if baseline:
        print("\n=== vs baseline eta=0.02 ===")
        for r in overall:
            d_ll = r["logloss"] - baseline["logloss"]
            d_ch = ""
            if "theta_Chernukha" in r and "theta_Chernukha" in baseline:
                d_ch = f"  ΔChernukha={r['theta_Chernukha'] - baseline['theta_Chernukha']:+.3f}"
            print(
                f"  eta={r['eta_teammate']:.4f}  Δll={d_ll:+.6f}{d_ch}"
            )

    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
