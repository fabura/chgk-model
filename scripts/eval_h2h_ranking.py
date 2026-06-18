"""Evaluate model θ against decisive head-to-head outcomes on shared questions.

Usage::

    python scripts/eval_h2h_ranking.py \\
      --cache_file data.npz \\
      --results_npz results/seq.npz \\
      --duckdb website/data/chgk.duckdb \\
      --min-games 200 --min-shared 50 --min-decisive 20 \\
      --out_dir results/h2h_eval
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import load_cached
from rating.h2h import (
    PairStat,
    build_pair_stats_from_arrays,
    compute_duel_scores,
    compute_pairwise_concordance,
    context_explains,
    fit_duel_elo,
    pair_outcome,
    pair_stat_to_dict,
)
from rating.io import load_results_npz

try:
    from scipy.stats import kendalltau, spearmanr

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

_TYPE_LABELS = ("offline", "sync", "async")


def _load_player_names(duckdb_path: Path | None) -> dict[int, str]:
    if duckdb_path is None or not duckdb_path.is_file():
        return {}
    import duckdb

    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT player_id, first_name, last_name FROM players"
        ).fetchall()
    finally:
        con.close()
    return {
        int(r[0]): f"{r[1] or ''} {r[2] or ''}".strip() for r in rows
    }


def _theta_by_player_idx(results, maps) -> np.ndarray:
    theta = np.full(maps.num_players, np.nan, dtype=np.float64)
    pid_to_idx = maps.player_id_to_idx
    for i, pid in enumerate(results.player_id):
        pidx = pid_to_idx.get(int(pid))
        if pidx is not None:
            theta[pidx] = float(results.theta[i])
    return theta


def _games_by_player_idx(results, maps) -> np.ndarray:
    games = np.zeros(maps.num_players, dtype=np.int32)
    pid_to_idx = maps.player_id_to_idx
    for i, pid in enumerate(results.player_id):
        pidx = pid_to_idx.get(int(pid))
        if pidx is not None:
            games[pidx] = int(results.games[i])
    return games


def _eligible_mask(games: np.ndarray, min_games: int) -> np.ndarray:
    return games >= min_games


def _correlation_metrics(
    theta: np.ndarray,
    scores: dict[int, float],
    eligible: np.ndarray,
) -> dict[str, float]:
    idx = np.where(eligible & np.isfinite(theta))[0]
    if len(idx) < 3:
        return {"spearman_rho": float("nan"), "kendall_tau": float("nan"), "n_players": len(idx)}
    th = theta[idx]
    duel = np.array([scores.get(int(p), 0.0) for p in idx], dtype=np.float64)
    out: dict[str, float] = {"n_players": float(len(idx))}
    if _HAVE_SCIPY:
        rho, _ = spearmanr(th, duel)
        tau, _ = kendalltau(th, duel)
        out["spearman_rho"] = float(rho)
        out["kendall_tau"] = float(tau)
    return out


def _filter_pairs(
    pair_stats: dict[tuple[int, int], PairStat],
    *,
    min_shared: int,
    min_decisive: int,
) -> dict[tuple[int, int], PairStat]:
    return {
        k: v
        for k, v in pair_stats.items()
        if v.n_shared >= min_shared and v.n_decisive >= min_decisive
    }


def _concordance_by_mode(
    pair_stats: dict[tuple[int, int], PairStat],
    arrays: dict,
    maps,
    theta: np.ndarray,
    *,
    min_shared: int,
    min_decisive: int,
) -> list[dict]:
    if maps.game_type is None or "game_idx" not in arrays:
        return []

    # Rebuild per-pair mode counts from tourney_decisive keys is hard;
    # use overall concordance per mode via slot rebuild is expensive.
    # Simpler: slice pair_stats by checking if majority of tourney ids are of mode X.
    tid_to_mode: dict[int, str] = {}
    for gidx, gid in enumerate(maps.idx_to_game_id):
        if gidx < len(maps.game_type):
            tid_to_mode[int(gid)] = str(maps.game_type[gidx])

    mode_pairs: dict[str, list[PairStat]] = {m: [] for m in _TYPE_LABELS}
    for st in pair_stats.values():
        if st.n_shared < min_shared or st.n_decisive < min_decisive:
            continue
        if not st.tourney_decisive:
            continue
        mode_counts: dict[str, int] = {}
        for tid in st.tourney_decisive:
            m = tid_to_mode.get(tid, "offline")
            mode_counts[m] = mode_counts.get(m, 0) + 1
        dominant = max(mode_counts, key=mode_counts.get)
        mode_pairs[dominant].append(st)

    rows = []
    for mode, stats_list in mode_pairs.items():
        if not stats_list:
            continue
        sub = {(s.p_lo, s.p_hi): s for s in stats_list}
        m = compute_pairwise_concordance(
            sub, theta, min_shared=0, min_decisive=min_decisive
        )
        rows.append({"slice": f"mode_{mode}", **m})
    return rows


def find_outlier_pairs(
    pair_stats: dict[tuple[int, int], PairStat],
    theta: np.ndarray,
    maps,
    names: dict[int, str],
    *,
    min_shared: int,
    min_decisive: int,
    delta_theta_min: float,
    delta_decisive_rate_min: float,
) -> list[dict]:
    rows: list[dict] = []
    for st in pair_stats.values():
        if st.n_shared < min_shared or st.n_decisive < min_decisive:
            continue
        nd = st.n_decisive
        if abs(st.only_lo - st.only_hi) / nd < delta_decisive_rate_min:
            continue
        d_theta = float(theta[st.p_lo] - theta[st.p_hi])
        if abs(d_theta) < delta_theta_min:
            continue
        fact_lo = st.only_lo > st.only_hi
        pred_lo = d_theta > 0
        if pred_lo == fact_lo:
            continue
        row = pair_stat_to_dict(st, theta=theta, maps=maps, delta_theta_min=delta_theta_min)
        row["name_a"] = names.get(row["player_id_a"], "")
        row["name_b"] = names.get(row["player_id_b"], "")
        row["top_tournaments"] = ";".join(
            f"{tid}:{delta}" for tid, delta in row.pop("top_tourney_deltas")
        )
        rows.append(row)
    rows.sort(
        key=lambda r: (
            abs(r["delta_theta"]) * r["decisive_rate_delta"],
            r["n_decisive"],
        ),
        reverse=True,
    )
    return rows


def find_outlier_players(
    pair_stats: dict[tuple[int, int], PairStat],
    theta: np.ndarray,
    maps,
    names: dict[int, str],
    eligible: np.ndarray,
    *,
    min_shared: int,
    min_decisive: int,
    delta_theta_min: float,
) -> list[dict]:
    upset_wins: dict[int, float] = {}
    upset_losses: dict[int, float] = {}
    ctx_adj: dict[int, float] = {}

    def _bump(d: dict[int, float], pidx: int, w: float) -> None:
        d[pidx] = d.get(pidx, 0.0) + w

    for st in pair_stats.values():
        if st.n_shared < min_shared or st.n_decisive < min_decisive:
            continue
        if st.only_lo == st.only_hi:
            continue
        p_lo, p_hi = st.p_lo, st.p_hi
        d_th = float(theta[p_lo] - theta[p_hi])
        if abs(d_th) < delta_theta_min:
            continue
        fact_lo_wins = st.only_lo > st.only_hi
        pred_lo_wins = d_th > 0
        w = float(st.n_decisive)
        explained = context_explains(st, delta_theta=delta_theta_min)
        weight = w * 0.25 if explained else w

        if fact_lo_wins and not pred_lo_wins:
            _bump(upset_wins, p_lo, weight)
            _bump(upset_losses, p_hi, weight)
            if explained:
                _bump(ctx_adj, p_lo, w * 0.25)
        elif not fact_lo_wins and pred_lo_wins:
            _bump(upset_wins, p_hi, weight)
            _bump(upset_losses, p_lo, weight)
            if explained:
                _bump(ctx_adj, p_hi, w * 0.25)

    rows: list[dict] = []
    for pidx in np.where(eligible)[0]:
        pidx = int(pidx)
        uw = upset_wins.get(pidx, 0.0)
        ul = upset_losses.get(pidx, 0.0)
        if uw + ul == 0:
            continue
        pid = maps.idx_to_player_id[pidx]
        rows.append(
            {
                "player_id": pid,
                "name": names.get(pid, ""),
                "theta": float(theta[pidx]),
                "games": 0,
                "upset_wins": uw,
                "upset_losses": ul,
                "net_upset": uw - ul,
                "context_adjusted_net_upset": ctx_adj.get(pidx, 0.0) - ul * 0.25,
            }
        )
    rows.sort(key=lambda r: r["net_upset"], reverse=True)
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_report(
    path: Path,
    summary_rows: list[dict],
    outlier_pairs: list[dict],
    outlier_players: list[dict],
    *,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# H2H evaluation report",
        "",
        "## Settings",
        f"- min_games: {args.min_games}",
        f"- min_shared: {args.min_shared}",
        f"- min_decisive: {args.min_decisive}",
        f"- delta_theta_min: {args.delta_theta_min}",
        f"- delta_decisive_rate_min: {args.delta_decisive_rate_min}",
        "",
        "## Summary metrics",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- **{row['slice']}**: pairs={row.get('n_pairs', 'n/a')}, "
            f"accuracy={row.get('accuracy', 'n/a')}, "
            f"weighted_accuracy={row.get('weighted_accuracy', 'n/a')}, "
            f"spearman={row.get('spearman_rho', 'n/a')}, "
            f"kendall={row.get('kendall_tau', 'n/a')}"
        )
    lines.extend(["", "## Top outlier pairs", ""])
    for r in outlier_pairs[:20]:
        lines.append(
            f"- {r.get('name_a', r['player_id_a'])} ({r['theta_a']:+.2f}) vs "
            f"{r.get('name_b', r['player_id_b'])} ({r['theta_b']:+.2f}): "
            f"decisive {r['only_a']}-{r['only_b']}/{r['n_decisive']}, "
            f"context_explains={r['context_explains']}"
        )
    lines.extend(["", "## Top outlier players", ""])
    for r in outlier_players[:20]:
        lines.append(
            f"- {r.get('name', r['player_id'])} (θ={r['theta']:+.2f}): "
            f"net_upset={r['net_upset']:.1f}, "
            f"context_adj={r['context_adjusted_net_upset']:.1f}"
        )
    lines.extend([
        "",
        "## Caveats",
        "- Descriptive metric: final θ was trained on the same questions.",
        "- `taken` is team-level, not individual credit.",
        "- Outliers with `context_explains=True` may reflect stronger teammates.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="H2H decisive duel evaluation vs θ")
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--results_npz", default="results/seq.npz")
    ap.add_argument("--duckdb", default="website/data/chgk.duckdb")
    ap.add_argument("--out_dir", default="results/h2h_eval")
    ap.add_argument("--min-games", type=int, default=200)
    ap.add_argument("--min-shared", type=int, default=50)
    ap.add_argument("--min-decisive", type=int, default=20)
    ap.add_argument("--min-opponents", type=int, default=10)
    ap.add_argument("--delta-theta-min", type=float, default=0.3)
    ap.add_argument("--delta-decisive-rate-min", type=float, default=0.10)
    ap.add_argument("--fit-elo", action="store_true")
    ap.add_argument(
        "--with-context",
        action="store_true",
        help="Accumulate team/teammate theta context (slower)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    arrays, maps = load_cached(args.cache_file)
    results = load_results_npz(args.results_npz)
    names = _load_player_names(Path(args.duckdb) if args.duckdb else None)

    theta = _theta_by_player_idx(results, maps)
    games = _games_by_player_idx(results, maps)
    eligible = _eligible_mask(games, args.min_games)
    n_eligible = int(eligible.sum())
    print(f"  eligible players (games>={args.min_games}): {n_eligible}", flush=True)

    print("Building pair stats...", flush=True)
    pair_stats = build_pair_stats_from_arrays(
        arrays,
        maps,
        theta=theta,
        eligible=eligible,
        exclude_same_team=True,
        collect_context=args.with_context,
    )
    filtered = _filter_pairs(
        pair_stats,
        min_shared=args.min_shared,
        min_decisive=args.min_decisive,
    )
    print(f"  total pairs: {len(pair_stats)}, filtered: {len(filtered)}", flush=True)

    eligible_pidx = [int(p) for p in np.where(eligible)[0]]
    duel_scores = compute_duel_scores(
        pair_stats, eligible_pidx, min_decisive=args.min_decisive
    )

    concordance = compute_pairwise_concordance(
        pair_stats,
        theta,
        min_shared=args.min_shared,
        min_decisive=args.min_decisive,
    )
    corr = _correlation_metrics(theta, duel_scores, eligible)

    summary_rows: list[dict] = [
        {
            "slice": "overall",
            **concordance,
            **corr,
        }
    ]
    summary_rows.extend(
        _concordance_by_mode(
            pair_stats,
            arrays,
            maps,
            theta,
            min_shared=args.min_shared,
            min_decisive=args.min_decisive,
        )
    )

    if args.fit_elo:
        elo = fit_duel_elo(
            pair_stats, eligible_pidx, min_decisive=args.min_decisive
        )
        elo_arr = np.array([elo.get(p, 0.0) for p in eligible_pidx])
        th_arr = theta[eligible_pidx]
        if _HAVE_SCIPY and len(eligible_pidx) > 2:
            rho, _ = spearmanr(th_arr, elo_arr)
            tau, _ = kendalltau(th_arr, elo_arr)
            summary_rows.append(
                {
                    "slice": "elo_correlation",
                    "spearman_rho": float(rho),
                    "kendall_tau": float(tau),
                    "n_players": float(len(eligible_pidx)),
                }
            )

    for mg in (200, 500, 1000):
        if mg < args.min_games:
            continue
        sub_elig = games >= mg
        sub_scores = compute_duel_scores(
            pair_stats,
            [int(p) for p in np.where(sub_elig)[0]],
            min_decisive=args.min_decisive,
        )
        sub_corr = _correlation_metrics(theta, sub_scores, sub_elig)
        summary_rows.append({"slice": f"games_ge_{mg}", **sub_corr})

    outlier_pairs = find_outlier_pairs(
        pair_stats,
        theta,
        maps,
        names,
        min_shared=args.min_shared,
        min_decisive=args.min_decisive,
        delta_theta_min=args.delta_theta_min,
        delta_decisive_rate_min=args.delta_decisive_rate_min,
    )
    outlier_players = find_outlier_players(
        pair_stats,
        theta,
        maps,
        names,
        eligible,
        min_shared=args.min_shared,
        min_decisive=args.min_decisive,
        delta_theta_min=args.delta_theta_min,
    )
    for row in outlier_players:
        pid = row["player_id"]
        pidx = maps.player_id_to_idx.get(pid)
        row["games"] = int(games[pidx]) if pidx is not None else 0

    pair_rows = []
    for st in filtered.values():
        row = pair_stat_to_dict(
            st, theta=theta, maps=maps, delta_theta_min=args.delta_theta_min
        )
        row["name_a"] = names.get(row["player_id_a"], "")
        row["name_b"] = names.get(row["player_id_b"], "")
        row.pop("top_tourney_deltas", None)
        pair_rows.append(row)

    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_csv(out_dir / "pair_stats.csv", pair_rows)
    _write_csv(out_dir / "outlier_pairs.csv", outlier_pairs)
    _write_csv(out_dir / "outlier_players.csv", outlier_players)
    _write_report(
        out_dir / "h2h_eval_report.md",
        summary_rows,
        outlier_pairs,
        outlier_players,
        args=args,
    )

    print(f"\nOverall concordance: {concordance}", flush=True)
    print(f"Correlation: {corr}", flush=True)
    print(f"Outlier pairs: {len(outlier_pairs)}, players: {len(outlier_players)}", flush=True)
    print(f"Wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
