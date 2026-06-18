"""Sweep offline-only ``eta_teammate`` under honest cell-holdout.

Keeps sync/async at ``eta_teammate=0.02`` (production default) and varies
``eta_teammate_offline``.  Use ``None`` / ``baseline`` for the current
uniform 0.02 everywhere.

Outputs ``results/exp_eta_teammate_offline.csv``.

Usage::

    python scripts/exp_eta_teammate_offline.py --cache_file data.npz

Cost: ~4 trials × ~7 min ≈ 28 min.
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
    34909: "Chernukha",
    26818: "Rekshinskaya",
    158668: "Monina",
    131922: "Vasiliev",
    23737: "Ostrovsky",
}

CHR_2026_TID = 12826
WATCH_CHR_PID = 131922  # Vasiliev


def _bucket_type(g: str) -> str:
    if "async" in g:
        return "async"
    if "sync" in g:
        return "sync"
    return "offline"


def _bucket_team_size(n: int) -> str:
    if n <= 1:
        return "solo"
    if n <= 3:
        return "small_2_3"
    if n <= 6:
        return "mid_4_6"
    return "large_7plus"


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


def _tournament_delta_theta(
    history: list[tuple[int, int, float]] | None,
    player_id: int,
    tournament_id: int,
) -> float | None:
    if not history:
        return None
    theta_before: float | None = None
    theta_after: float | None = None
    for pid, gid, theta in history:
        if pid != player_id:
            continue
        if gid == tournament_id:
            theta_after = theta
            break
        theta_before = theta
    if theta_before is None or theta_after is None:
        return None
    return theta_after - theta_before


def _parse_offline_values(raw: str) -> list[float | None]:
    out: list[float | None] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if part in ("baseline", "none", "null", ""):
            out.append(None)
        else:
            out.append(float(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--out", default="results/exp_eta_teammate_offline.csv")
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--values",
        default="baseline,0.01,0.005,0.0",
        help="comma-separated eta_teammate_offline values (baseline=None)",
    )
    ap.add_argument(
        "--eta-teammate",
        type=float,
        default=0.02,
        dest="eta_teammate",
        help="sync/async eta_teammate (fixed during sweep)",
    )
    args = ap.parse_args()

    offline_values = _parse_offline_values(args.values)
    print(f"[load] {args.cache_file}", flush=True)
    arrays, maps = load_cached(args.cache_file)
    team_sizes = arrays["team_sizes"].astype(np.int32)

    all_pids = {int(p) for p in maps.idx_to_player_id}
    case_ids = {k: v for k, v in CASE_STUDY.items() if k in all_pids}
    missing = set(CASE_STUDY) - set(case_ids)
    if missing:
        print(f"[warn] case-study ids not in cache: {missing}", flush=True)

    chr_in_cache = CHR_2026_TID in maps.idx_to_game_id
    if not chr_in_cache:
        print(
            f"[warn] tournament {CHR_2026_TID} not in cache — "
            "Δθ on II ЧР skipped",
            flush=True,
        )

    rows: list[dict] = []
    for eta_off in offline_values:
        label = "baseline" if eta_off is None else f"{eta_off:.4f}"
        print(
            f"\n=== eta_teammate_offline={label} "
            f"(sync/async={args.eta_teammate:.4f}) ===",
            flush=True,
        )
        cfg_kw: dict = {
            "eta_teammate": args.eta_teammate,
            "holdout_obs_fraction": args.holdout,
            "holdout_seed": args.seed,
        }
        if eta_off is not None:
            cfg_kw["eta_teammate_offline"] = eta_off
        cfg = Config(**cfg_kw)

        t0 = time.time()
        result = run_sequential(
            arrays,
            maps,
            cfg,
            verbose=False,
            collect_predictions=True,
            collect_history=chr_in_cache,
        )
        elapsed = time.time() - t0

        pred = result.predictions
        mask = pred["is_holdout"].astype(bool)
        p = pred["pred_p"][mask]
        y = pred["actual_y"][mask]
        g = pred["game_idx"][mask]
        obs_idx = pred["obs_idx"][mask]
        ts = team_sizes[obs_idx]

        m = compute_metrics(p, y)
        row: dict = {
            "eta_teammate": args.eta_teammate,
            "eta_teammate_offline": "" if eta_off is None else eta_off,
            "slice": "all",
            "n": int(mask.sum()),
            "logloss": round(float(m["logloss"]), 6),
            "brier": round(float(m["brier"]), 6),
            "auc": round(float(m["auc"]), 6) if not np.isnan(m["auc"]) else "",
            "elapsed_sec": round(elapsed, 1),
        }
        row.update(_player_thetas(result, maps, case_ids))
        if chr_in_cache:
            dth = _tournament_delta_theta(
                result.history, WATCH_CHR_PID, CHR_2026_TID
            )
            row["vasiliev_dtheta_chr12826"] = (
                round(dth, 6) if dth is not None else ""
            )
        rows.append(row)
        print(
            f"  overall : ll={row['logloss']:.4f}  brier={row['brier']:.4f} "
            f"AUC={row['auc']}  ({elapsed:.1f}s)",
            flush=True,
        )
        if "vasiliev_dtheta_chr12826" in row and row["vasiliev_dtheta_chr12826"] != "":
            print(
                f"  Vasiliev Δθ on #{CHR_2026_TID}: "
                f"{row['vasiliev_dtheta_chr12826']:+.4f}",
                flush=True,
            )
        for label_name in CASE_STUDY.values():
            key = f"theta_{label_name}"
            if key in row and row[key] != "":
                print(f"  {key}={row[key]:+.4f}", flush=True)

        gtype = getattr(maps, "game_type", None)
        if gtype is not None:
            types = np.array(
                [_bucket_type(str(gtype[gi])) for gi in g], dtype=object
            )
            for t in ("offline", "sync", "async"):
                tm = types == t
                if not tm.any():
                    continue
                ms = compute_metrics(p[tm], y[tm])
                slice_row = {
                    "eta_teammate": args.eta_teammate,
                    "eta_teammate_offline": "" if eta_off is None else eta_off,
                    "slice": t,
                    "n": int(tm.sum()),
                    "logloss": round(float(ms["logloss"]), 6),
                    "brier": round(float(ms["brier"]), 6),
                    "auc": (
                        round(float(ms["auc"]), 6)
                        if not np.isnan(ms["auc"]) else ""
                    ),
                    "elapsed_sec": "",
                }
                rows.append(slice_row)
                print(
                    f"  {t:7s}: ll={ms['logloss']:.4f} AUC={ms['auc']:.4f}",
                    flush=True,
                )

        size_buckets = np.array([_bucket_team_size(int(n)) for n in ts], dtype=object)
        for sb in ("solo", "small_2_3", "mid_4_6", "large_7plus"):
            sm = size_buckets == sb
            if not sm.any():
                continue
            ms = compute_metrics(p[sm], y[sm])
            slice_row = {
                "eta_teammate": args.eta_teammate,
                "eta_teammate_offline": "" if eta_off is None else eta_off,
                "slice": f"size_{sb}",
                "n": int(sm.sum()),
                "logloss": round(float(ms["logloss"]), 6),
                "brier": round(float(ms["brier"]), 6),
                "auc": (
                    round(float(ms["auc"]), 6)
                    if not np.isnan(ms["auc"]) else ""
                ),
                "elapsed_sec": "",
            }
            rows.append(slice_row)

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
        offline = r["eta_teammate_offline"]
        offline_s = "baseline" if offline == "" else f"{float(offline):.4f}"
        thetas = "  ".join(
            f"{k.replace('theta_', '')}={r[k]:+.3f}"
            for k in sorted(r)
            if k.startswith("theta_") and r[k] != ""
        )
        dchr = r.get("vasiliev_dtheta_chr12826", "")
        dchr_s = f"  VasilievΔCHR={dchr}" if dchr != "" else ""
        print(
            f"  offline={offline_s}  ll={r['logloss']:.4f}  "
            f"AUC={r['auc']}  {thetas}{dchr_s}{marker}"
        )

    print(f"\n[ok] {len(rows)} rows → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
