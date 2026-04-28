"""
CLI entry-point:  python -m rating

Examples
--------
    python -m rating --mode synthetic
    python -m rating --mode cached --cache_file data.pkl
    python -m rating --mode cached --cache_file data.pkl --backtest
    python -m rating --mode db --players_out results/seq_players.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ChGK Sequential Online Rating System"
    )

    src = parser.add_argument_group("data source")
    src.add_argument(
        "--mode",
        choices=["synthetic", "db", "cached"],
        default="cached",
    )
    src.add_argument("--cache_file", type=str, default=None)
    src.add_argument("--max_tournaments", type=int, default=None)
    src.add_argument(
        "--min_tournament_date", type=str, default="2015-01-01"
    )

    hp = parser.add_argument_group("hyperparameters")
    hp.add_argument(
        "--eta0", type=float, default=0.15, help="Base learning rate"
    )
    hp.add_argument(
        "--rho", type=float, default=0.9995, help="Rating decay"
    )
    hp.add_argument(
        "--w_online",
        type=float,
        default=0.5,
        help="Async/online weight for player updates",
    )
    hp.add_argument(
        "--w_online_questions",
        type=float,
        default=0.15,
        help="Async/online weight for question difficulty updates",
    )
    hp.add_argument(
        "--w_online_log_a",
        type=float,
        default=0.05,
        help="Async/online weight for question discrimination updates",
    )
    hp.add_argument(
        "--reg_theta",
        type=float,
        default=0.0,
        help="L2-style shrinkage for player theta (per-step pull to 0)",
    )
    hp.add_argument(
        "--reg_b",
        type=float,
        default=0.0,
        help="L2-style shrinkage for question difficulty b",
    )
    hp.add_argument(
        "--reg_log_a",
        type=float,
        default=0.0,
        help="L2-style shrinkage for log_a (question discrimination)",
    )
    hp.add_argument(
        "--use-calendar-decay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-player calendar-based decay (default: on). "
        "Disable with --no-use-calendar-decay to fall back to the legacy "
        "global per-tournament decay (rho).",
    )
    hp.add_argument(
        "--rho_calendar",
        type=float,
        default=1.0,
        help="Calendar decay factor per period (1.0 = no decay; default per-week unit)",
    )
    hp.add_argument(
        "--decay_period_days",
        type=float,
        default=7.0,
        help="Length of one decay period in days (default 7)",
    )
    hp.add_argument(
        "--cold_init_factor",
        type=float,
        default=1.0,
        help="Shrink team-mean θ when initialising a new player (1.0 = inherit fully, 0.5 = half)",
    )
    hp.add_argument(
        "--cold_init_theta",
        type=float,
        default=0.0,
        help="Prior θ for first-time players (used when team mean unavailable, "
        "or always if --no-cold-init-team-mean).",
    )
    hp.add_argument(
        "--cold-init-team-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inherit (blended) team-mean θ for new players. "
        "Disable with --no-cold-init-team-mean to use cold_init_theta uniformly.",
    )
    hp.add_argument(
        "--games_offset",
        type=float,
        default=1.0,
        help="Adaptive lr offset: η_k = η0/√(games_offset + games_k). "
        "Values < 1 give a chess-Elo-style rookie boost (e.g. 0.25 → first-game lr is 2× η0).",
    )
    hp.add_argument(
        "--use-team-size-effect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn a per-team-size shift on tournament difficulty (default on). "
        "Disable with --no-use-team-size-effect.",
    )
    hp.add_argument(
        "--team_size_max",
        type=int,
        default=8,
        help="Cap team size for the size-effect (sizes above are clipped to this).",
    )
    hp.add_argument(
        "--team_size_anchor",
        type=int,
        default=6,
        help="Team size whose δ is fixed at 0 (identifiability anchor).",
    )
    hp.add_argument(
        "--eta_size",
        type=float,
        default=0.001,
        help="Learning rate for delta_size (per-team-size effect).",
    )
    hp.add_argument(
        "--reg_size",
        type=float,
        default=0.0,
        help="L2-style shrinkage for delta_size (default 0.0; "
             "raised values pull δ_size toward 0 — see "
             "docs/error_structure_2026-04.md §3).",
    )
    hp.add_argument(
        "--use-pos-effect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn a per-position-in-tour shift on tournament difficulty (default on). "
        "Disable with --no-use-pos-effect.",
    )
    hp.add_argument(
        "--tour_len",
        type=int,
        default=12,
        help="Standard tour length (questions are bucketed by index%%tour_len).",
    )
    hp.add_argument(
        "--pos_anchor",
        type=int,
        default=0,
        help="Position whose δ is fixed at 0 (identifiability anchor; "
        "default 0 = easiest position in the tour).",
    )
    hp.add_argument(
        "--eta_pos",
        type=float,
        default=0.001,
        help="Learning rate for delta_pos (per-position-in-tour effect).",
    )
    hp.add_argument(
        "--reg_pos",
        type=float,
        default=0.10,
        help="L2-style shrinkage for delta_pos.",
    )
    hp.add_argument(
        "--recenter_period_days",
        type=float,
        default=365.0,
        help="Period (days) between gauge re-centerings; 0 to disable.",
    )
    hp.add_argument(
        "--recenter_target",
        type=float,
        default=-0.70,
        help="Target median θ of active veterans after each re-centering.",
    )
    hp.add_argument(
        "--recenter_min_games",
        type=int,
        default=200,
        help="Min games to count a player as 'veteran' for re-centering.",
    )
    hp.add_argument(
        "--recenter_active_days",
        type=int,
        default=365,
        help="Days since last game to count a veteran as 'active'.",
    )
    hp.add_argument(
        "--eta_teammate",
        type=float,
        default=0.005,
        help="Per-tournament L2 pull of teammate θ toward team mean",
    )
    hp.add_argument(
        "--noisy-or-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialise b accounting for noisy-OR composition over team "
        "size: b = log(n) - log(-log(1-p)) instead of legacy b = -log(p). "
        "Disable with --no-noisy-or-init for baseline ablation.",
    )
    hp.add_argument(
        "--theta-bar-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extend noisy-OR init with average pre-tournament team θ: "
        "b = log(n) + θ̄ - log(-log(1-p)).  θ̄ averages over mature "
        "players only (--theta-bar-min-games).  Disable for ablation.",
    )
    hp.add_argument(
        "--theta-bar-min-games",
        type=int,
        default=3,
        help="Minimum games for a player to count toward θ̄ in init "
        "(default: 3; rookies are excluded because their θ is dominated "
        "by cold_init_theta).",
    )

    ev = parser.add_argument_group("evaluation")
    ev.add_argument(
        "--backtest", action="store_true", help="Run time-based backtesting"
    )
    ev.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (grid search over eta0, rho, w_online)",
    )
    ev.add_argument(
        "--tune-trials",
        type=int,
        default=None,
        help="Random search trials (default: grid search)",
    )
    ev.add_argument(
        "--tune-output",
        type=str,
        default=None,
        help="Save tune results to CSV",
    )
    ev.add_argument("--test_fraction", type=float, default=0.2)

    out = parser.add_argument_group("output")
    out.add_argument(
        "--results_npz", type=str, default=None,
        help="Save all results to one compressed .npz (players, questions, history)"
    )
    out.add_argument(
        "--players_out", type=str, default=None, help="Player ratings CSV"
    )
    out.add_argument(
        "--shrink_K",
        type=float,
        default=20.0,
        help="Shrinkage constant K for theta_shrunk = theta * games / (games + K)",
    )
    out.add_argument(
        "--questions_out", type=str, default=None, help="Question params CSV"
    )
    out.add_argument(
        "--history_out", type=str, default=None, help="Rating history CSV"
    )

    args = parser.parse_args()

    from rating.backtest import backtest as do_backtest
    from rating.engine import Config, run_sequential

    cfg = Config(
        eta0=args.eta0,
        rho=args.rho,
        w_online=args.w_online,
        w_online_questions=args.w_online_questions,
        w_online_log_a=args.w_online_log_a,
        reg_theta=args.reg_theta,
        reg_b=args.reg_b,
        reg_log_a=args.reg_log_a,
        use_calendar_decay=args.use_calendar_decay,
        rho_calendar=args.rho_calendar,
        decay_period_days=args.decay_period_days,
        cold_init_factor=args.cold_init_factor,
        cold_init_theta=args.cold_init_theta,
        cold_init_use_team_mean=args.cold_init_team_mean,
        games_offset=args.games_offset,
        use_team_size_effect=args.use_team_size_effect,
        team_size_max=args.team_size_max,
        team_size_anchor=args.team_size_anchor,
        eta_size=args.eta_size,
        reg_size=args.reg_size,
        use_pos_effect=args.use_pos_effect,
        tour_len=args.tour_len,
        pos_anchor=args.pos_anchor,
        eta_pos=args.eta_pos,
        reg_pos=args.reg_pos,
        eta_teammate=args.eta_teammate,
        recenter_period_days=args.recenter_period_days,
        recenter_target=args.recenter_target,
        recenter_min_games=args.recenter_min_games,
        recenter_active_days=args.recenter_active_days,
        noisy_or_init=args.noisy_or_init,
        theta_bar_init=args.theta_bar_init,
        theta_bar_min_games=args.theta_bar_min_games,
    )

    # --- load data ---
    arrays, maps = _load_data(args, parser)

    print(
        f"Data: {len(arrays['q_idx'])} observations, "
        f"{maps.num_players} players, "
        f"{maps.num_questions} questions"
    )

    if args.backtest:
        do_backtest(
            arrays, maps, cfg, test_fraction=args.test_fraction
        )
        return 0

    if args.tune:
        from rating.tune import tune

        results = tune(
            arrays,
            maps,
            grid=args.tune_trials is None,
            n_trials=args.tune_trials or 20,
            test_fraction=args.test_fraction,
            metric="logloss",
            verbose=True,
        )
        best = results[0]
        print(f"\n{'=' * 60}")
        print("BEST CONFIG (by logloss):")
        print(
            f"  eta0={best.config.eta0:.4f} "
            f"rho={best.config.rho:.4f} "
            f"w_online={best.config.w_online:.2f}"
        )
        print(
            f"  logloss={best.logloss:.4f} "
            f"Brier={best.brier:.4f} "
            f"AUC={best.auc:.4f}"
        )
        print(f"\nTo run with best config:")
        print(
            f"  --eta0 {best.config.eta0:.4f} --rho {best.config.rho:.4f} "
            f"--w_online {best.config.w_online:.2f}"
        )
        if args.tune_output:
            os.makedirs(os.path.dirname(args.tune_output) or ".", exist_ok=True)
            with open(args.tune_output, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    ["eta0", "rho", "w_online", "logloss", "brier", "auc"]
                )
                for r in results:
                    w.writerow([
                        r.config.eta0,
                        r.config.rho,
                        r.config.w_online,
                        round(r.logloss, 6),
                        round(r.brier, 6),
                        round(r.auc, 6),
                    ])
            print(f"\nResults saved to {args.tune_output}")
        print(f"{'=' * 60}")
        return 0

    collect_hist = args.history_out is not None or args.results_npz is not None
    result = run_sequential(
        arrays,
        maps,
        cfg,
        collect_history=collect_hist,
        collect_predictions=True,
    )

    if args.results_npz:
        _export_results_npz(args.results_npz, result, maps)
    if args.players_out:
        _export_players(args.players_out, result, maps, shrink_K=args.shrink_K)
    if args.questions_out:
        _export_questions(args.questions_out, result, maps)
    if args.history_out and result.history:
        _export_history(args.history_out, result.history)

    return 0


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def _load_data(args, parser):
    if args.mode == "synthetic":
        from data import generate_synthetic_two_populations, samples_to_arrays

        samples, maps = generate_synthetic_two_populations()
        return samples_to_arrays(samples), maps

    if args.mode == "cached":
        if not args.cache_file:
            parser.error("--cache_file is required when --mode=cached")
        from data import load_cached

        return load_cached(args.cache_file)

    # mode == "db"
    from data import load_from_db, samples_to_arrays, save_cached

    samples, maps = load_from_db(
        max_tournaments=args.max_tournaments,
        min_tournament_date=args.min_tournament_date or None,
    )
    print("Building arrays...", flush=True)
    arrays = samples_to_arrays(samples)
    if args.cache_file:
        print(f"Saving cache to {args.cache_file}...", flush=True)
        save_cached(samples, maps, args.cache_file)
        print("Cache saved.", flush=True)
    return arrays, maps


# ------------------------------------------------------------------
# Compact NPZ export
# ------------------------------------------------------------------

def _export_results_npz(path: str, result, maps) -> None:
    """Save all results to one compressed .npz file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ps = result.players
    qs = result.questions
    cq = result.canonical_q_map

    # Players: full arrays (unseen: theta=0, games=0)
    player_id = np.array(maps.idx_to_player_id, dtype=np.int64)
    theta = ps.theta.astype(np.float32)
    games = ps.games.astype(np.int32)

    # Questions: canonical b, a; map raw->canonical
    qids = maps.idx_to_question_id
    q_is_tuple = qids and isinstance(qids[0], tuple)
    if q_is_tuple:
        q_tid = np.array([q[0] for q in qids], dtype=np.int32)
        q_qi = np.array([q[1] for q in qids], dtype=np.int32)
    else:
        q_tid = np.array(qids, dtype=np.int32)
        q_qi = np.zeros(len(qids), dtype=np.int32)

    kw = {
        "version": np.array([1], dtype=np.int32),
        "player_id": player_id,
        "theta": theta,
        "games": games,
        "question_tid": q_tid,
        "question_qi": q_qi,
        "question_is_tuple": np.array([1 if q_is_tuple else 0], dtype=np.int8),
        "b": qs.b.astype(np.float32),
        "a": qs.a.astype(np.float32),
    }
    if cq is not None:
        kw["canonical_q_idx"] = cq.astype(np.int32)

    # Team-size effect: indexed 0..team_size_max, anchored at team_size_anchor.
    if result.delta_size is not None:
        kw["delta_size"] = np.asarray(result.delta_size, dtype=np.float32)
        kw["team_size_anchor"] = np.array([int(result.team_size_anchor)], dtype=np.int32)

    # Position-in-tour effect: indexed 0..tour_len-1, anchored at pos_anchor.
    if result.delta_pos is not None:
        kw["delta_pos"] = np.asarray(result.delta_pos, dtype=np.float32)
        kw["pos_anchor"] = np.array([int(result.pos_anchor)], dtype=np.int32)

    if result.history:
        h = result.history
        kw["history_player_id"] = np.array([x[0] for x in h], dtype=np.int64)
        kw["history_game_id"] = np.array([x[1] for x in h], dtype=np.int64)
        kw["history_theta"] = np.array([x[2] for x in h], dtype=np.float32)

    rc_events = getattr(result, "recenter_events", None)
    if rc_events:
        kw["recenter_ord"] = np.array([e[0] for e in rc_events], dtype=np.int64)
        kw["recenter_delta"] = np.array([e[2] for e in rc_events], dtype=np.float32)

    np.savez_compressed(path, **kw)
    print(f"Results saved to {path}")


# ------------------------------------------------------------------
# CSV exports
# ------------------------------------------------------------------

def _export_players(path: str, result, maps, *, shrink_K: float = 20.0) -> None:
    """Export player ratings as CSV.

    Adds two extra columns over the raw θ:

    * ``theta_shrunk`` — empirical-Bayes-flavoured shrinkage toward
      zero based on the number of games:
      ``theta_shrunk = theta * games / (games + shrink_K)``.
      This protects the top of the table from low-experience extremes
      without retraining the model.
    * ``rank_shrunk`` — rank order by ``theta_shrunk``.

    Rows are still ordered by raw ``theta`` (descending) for backwards
    compatibility.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ps = result.players
    seen_mask = ps.seen
    games = ps.games.astype(np.float64)
    theta = ps.theta.astype(np.float64)
    theta_shrunk = theta * (games / (games + max(shrink_K, 1e-9)))

    seen_idx = np.where(seen_mask)[0]
    shrunk_order = seen_idx[np.argsort(theta_shrunk[seen_idx])[::-1]]
    rank_shrunk = {int(idx): rank + 1 for rank, idx in enumerate(shrunk_order)}

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "theta", "theta_shrunk", "rank_shrunk", "games"])
        order = np.argsort(theta)[::-1]
        for idx in order:
            if not seen_mask[idx]:
                continue
            pid = (
                maps.idx_to_player_id[idx]
                if idx < len(maps.idx_to_player_id)
                else idx
            )
            w.writerow([
                pid,
                round(float(theta[idx]), 6),
                round(float(theta_shrunk[idx]), 6),
                rank_shrunk.get(int(idx), -1),
                int(ps.games[idx]),
            ])
    print(f"Players saved to {path}")


def _export_questions(path: str, result, maps) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    qs = result.questions
    cq = result.canonical_q_map
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "b", "a"])
        for raw_idx in range(maps.num_questions):
            qi = int(cq[raw_idx]) if cq is not None else raw_idx
            if qi >= qs.num_questions or not qs.initialized[qi]:
                continue
            qid = (
                maps.idx_to_question_id[raw_idx]
                if raw_idx < len(maps.idx_to_question_id)
                else raw_idx
            )
            w.writerow([
                str(qid),
                round(float(qs.b[qi]), 6),
                round(float(qs.a[qi]), 6),
            ])
    print(f"Questions saved to {path}")


def _export_history(path: str, history: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "game_id", "theta"])
        for pid, gid, theta in history:
            w.writerow([pid, gid, round(theta, 6)])
    print(f"History saved to {path}")


if __name__ == "__main__":
    sys.exit(main())
