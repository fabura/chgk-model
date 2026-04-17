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
        "--eta0", type=float, default=0.10, help="Base learning rate"
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
        default=0.45,
        help="Async/online weight for question difficulty updates",
    )
    hp.add_argument(
        "--w_online_log_a",
        type=float,
        default=0.05,
        help="Async/online weight for question discrimination updates",
    )
    hp.add_argument(
        "--w_async_mode",
        type=float,
        default=0.3,
        help="Async/online weight for global mode-offset updates",
    )
    hp.add_argument(
        "--w_async_residual",
        type=float,
        default=0.6,
        help="Async/online weight for tournament residual-offset updates",
    )
    hp.add_argument(
        "--eta_mu",
        type=float,
        default=0.005,
        help="Learning rate for tournament mode offsets",
    )
    hp.add_argument(
        "--eta_eps",
        type=float,
        default=0.03,
        help="Learning rate for per-tournament residual offsets",
    )
    hp.add_argument(
        "--reg_mu_type",
        type=float,
        default=0.10,
        help="L2-style shrinkage for sync/async mode offsets",
    )
    hp.add_argument(
        "--reg_eps",
        type=float,
        default=0.20,
        help="L2-style shrinkage for per-tournament residual offsets",
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
        action="store_true",
        help="Use per-player calendar-based decay instead of global per-tournament decay",
    )
    hp.add_argument(
        "--rho_calendar",
        type=float,
        default=0.99,
        help="Calendar decay factor per period (default: per-week)",
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
        "--no-tournament-delta",
        action="store_true",
        help="Disable tournament difficulty offsets",
    )
    hp.add_argument(
        "--delta-type-prior",
        action="store_true",
        help="Initialize δ from type (offline=0, sync=-0.1, online=-0.2)",
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
        w_async_mode=args.w_async_mode,
        w_async_residual=args.w_async_residual,
        eta_mu=args.eta_mu,
        eta_eps=args.eta_eps,
        reg_mu_type=args.reg_mu_type,
        reg_eps=args.reg_eps,
        reg_theta=args.reg_theta,
        reg_b=args.reg_b,
        reg_log_a=args.reg_log_a,
        use_calendar_decay=args.use_calendar_decay,
        rho_calendar=args.rho_calendar,
        decay_period_days=args.decay_period_days,
        cold_init_factor=args.cold_init_factor,
        use_tournament_delta=not args.no_tournament_delta,
        use_delta_type_prior=args.delta_type_prior,
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

    if result.history:
        h = result.history
        kw["history_player_id"] = np.array([x[0] for x in h], dtype=np.int64)
        kw["history_game_id"] = np.array([x[1] for x in h], dtype=np.int64)
        kw["history_theta"] = np.array([x[2] for x in h], dtype=np.float32)

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
