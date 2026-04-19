"""Compare the trained-model backtest metrics to simple baselines on the
same prequential test split.

Baselines (all evaluated on the same test observations as the model):

    1. constant 0.5                       — pure chance
    2. train-set global mean take rate    — "knows nothing"
    3. tournament-type mean (train)       — "knows offline / sync / async"
    4. team-size mean (train)             — "knows team size only"
    5. test-tournament question marginal  — oracle that knows the question
       outcome but NOT the team (per-tournament per-question take rate);
       this is an UPPER bound on what any team-blind predictor can do
    6. test-tournament team marginal      — symmetric oracle: knows the
       team's average take rate in this tournament but not the question

Run:  python scripts/compare_to_baselines.py
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

import numpy as np

# Allow `python scripts/compare_to_baselines.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_cached
from rating.engine import Config, run_sequential
from rating.backtest import compute_metrics


def main(cache: str = "data.npz", test_fraction: float = 0.2) -> None:
    print(f"Loading {cache}...")
    arrays, maps = load_cached(cache)

    print("Running model (with predictions) ...")
    cfg = Config()  # current defaults — calendar decay, t6, δ_size, δ_pos
    result = run_sequential(arrays, maps, cfg, verbose=False, collect_predictions=True)

    pred = result.predictions
    pred_p = pred["pred_p"]
    actual_y = pred["actual_y"]
    pred_game = pred["game_idx"]

    # ---- determine test split (same logic as backtest.py) ----
    gdo = maps.game_date_ordinal
    all_games = np.unique(pred_game)
    known = all_games[np.array([gdo[g] >= 0 for g in all_games], dtype=bool)]
    ordered = known[np.argsort(np.array([gdo[g] for g in known]))]
    n_test = max(1, int(len(ordered) * test_fraction))
    test_games = set(int(g) for g in ordered[-n_test:])
    test_mask = np.array([int(g) in test_games for g in pred_game], dtype=bool)
    train_mask = ~test_mask

    print(f"Train obs: {train_mask.sum():,}   Test obs: {test_mask.sum():,}")
    print(f"Train games: {len(ordered) - n_test}   Test games: {n_test}")

    p_model = pred_p[test_mask]
    y_test = actual_y[test_mask]

    # ---- Build per-test-obs auxiliary indices ----
    # We need, for each *prediction row*, the (game_idx, q_idx, team_size).
    # These live alongside pred_p but only game_idx is in `pred`.  We'll
    # rebuild them by re-scanning the dataset in the same order the
    # predictions were collected.  Easier: re-derive by joining on game.
    #
    # Cheaper trick: predictions are collected in dataset order, so we can
    # match them back by index.  But the engine only collects predictions
    # for known-date games (≥0 ordinal).  Use the prequential-collect
    # invariant: pred arrays are in the same order as the original obs,
    # filtered to known-date games.
    n_obs_total = len(arrays["q_idx"])
    is_known_obs = (gdo[arrays["game_idx"]] >= 0)
    assert is_known_obs.sum() == len(pred_p), (
        f"Mismatch: {is_known_obs.sum()} known obs vs {len(pred_p)} preds"
    )
    q_idx_pred = arrays["q_idx"][is_known_obs]
    game_idx_pred = arrays["game_idx"][is_known_obs]
    team_size_pred = arrays["team_sizes"][is_known_obs]
    # canonical question id for sharing across paired tournaments
    cq_pred = maps.canonical_q_idx[q_idx_pred] if hasattr(maps, "canonical_q_idx") else q_idx_pred

    y_train = actual_y[train_mask]
    p_results: dict[str, np.ndarray] = {}

    # ---- 1. Constant 0.5 ----
    p_results["constant 0.5"] = np.full_like(p_model, 0.5)

    # ---- 2. Global train-set mean ----
    mean_train = float(y_train.mean())
    print(f"Train-set mean take rate: {mean_train:.4f}")
    p_results[f"train mean ({mean_train:.3f})"] = np.full_like(p_model, mean_train)

    # ---- 3. Tournament-type mean (train) ----
    gtype_arr = maps.game_type
    train_types = np.array(
        [str(gtype_arr[g]) for g in game_idx_pred[train_mask]], dtype=object
    )

    def bucket(g: str) -> str:
        if "async" in g:
            return "async"
        if "sync" in g:
            return "sync"
        return "offline"

    train_buckets = np.array([bucket(t) for t in train_types], dtype=object)
    type_mean: dict[str, float] = {}
    for name in ("offline", "sync", "async"):
        sel = train_buckets == name
        if sel.any():
            type_mean[name] = float(y_train[sel].mean())
        else:
            type_mean[name] = mean_train
    print(f"Per-type train means: {type_mean}")

    test_types = np.array(
        [str(gtype_arr[g]) for g in game_idx_pred[test_mask]], dtype=object
    )
    test_buckets = np.array([bucket(t) for t in test_types], dtype=object)
    p_results["per-type mean"] = np.array(
        [type_mean[b] for b in test_buckets], dtype=np.float64
    )

    # ---- 4. Team-size mean (train) ----
    ts_train = team_size_pred[train_mask]
    ts_test = team_size_pred[test_mask]
    size_mean: dict[int, float] = {}
    for n in range(1, int(ts_train.max()) + 1):
        sel = ts_train == n
        if sel.sum() >= 100:
            size_mean[n] = float(y_train[sel].mean())
    fallback = mean_train
    p_results["per-team-size mean"] = np.array(
        [size_mean.get(int(n), fallback) for n in ts_test], dtype=np.float64
    )

    # ---- 5. Question-marginal oracle (per-test-tournament, per-question) ----
    # For each (game, q) pair in the test set, compute the empirical take
    # rate within that tournament for that question.  This is what an
    # "oracle that knows the question difficulty in this tournament but
    # not the team" could predict.  Smoothed slightly with a Beta(1,1)
    # prior to avoid log(0).
    test_q = q_idx_pred[test_mask]
    test_g = game_idx_pred[test_mask]
    sums: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for q, g, y in zip(test_q, test_g, y_test):
        s = sums[(int(g), int(q))]
        s[0] += int(y)
        s[1] += 1
    p_q_oracle = np.empty_like(p_model)
    for i, (q, g) in enumerate(zip(test_q, test_g)):
        s = sums[(int(g), int(q))]
        # Beta(1,1) Laplace smoothing: (k+1)/(n+2)
        p_q_oracle[i] = (s[0] + 1.0) / (s[1] + 2.0)
    p_results["oracle: per-tournament question marginal"] = p_q_oracle

    # ---- 6. Team-marginal oracle (per-test-tournament, per-team) ----
    # Each "team" is a unique roster; identify by sorted player tuple.
    offsets = np.cumsum(np.concatenate([[0], arrays["team_sizes"]]))
    pflat = arrays["player_indices_flat"]
    # Build per-known-obs roster identifier
    test_indices = np.where(is_known_obs)[0][test_mask]
    sums_team: dict[tuple[int, tuple], list[int]] = defaultdict(lambda: [0, 0])
    rosters_test = []
    for idx_in_obs, y in zip(test_indices, y_test):
        s_, e_ = int(offsets[idx_in_obs]), int(offsets[idx_in_obs + 1])
        roster = tuple(sorted(int(p) for p in pflat[s_:e_]))
        g = int(arrays["game_idx"][idx_in_obs])
        rosters_test.append((g, roster))
        rec = sums_team[(g, roster)]
        rec[0] += int(y)
        rec[1] += 1
    p_t_oracle = np.empty_like(p_model)
    for i, key in enumerate(rosters_test):
        rec = sums_team[key]
        p_t_oracle[i] = (rec[0] + 1.0) / (rec[1] + 2.0)
    p_results["oracle: per-tournament team marginal"] = p_t_oracle

    # ---- Print results table ----
    print()
    print(f"{'predictor':<48} {'logloss':>9} {'Brier':>9} {'AUC':>9}")
    print("-" * 78)
    for name, p in p_results.items():
        m = compute_metrics(p, y_test)
        print(f"{name:<48} {m['logloss']:>9.4f} {m['brier']:>9.4f} {m['auc']:>9.4f}")
    m_model = compute_metrics(p_model, y_test)
    print("-" * 78)
    print(
        f"{'TRAINED MODEL (calendar decay + δ_size + δ_pos)':<48} "
        f"{m_model['logloss']:>9.4f} {m_model['brier']:>9.4f} {m_model['auc']:>9.4f}"
    )

    # Also report skill scores vs the strongest naive baseline (per-type)
    base = compute_metrics(p_results["per-type mean"], y_test)
    skill_ll = 1.0 - m_model["logloss"] / base["logloss"]
    skill_br = 1.0 - m_model["brier"] / base["brier"]
    print()
    print(f"Skill vs per-type-mean baseline:  logloss {skill_ll:+.1%},  Brier {skill_br:+.1%}")

    # Distance to the per-question-oracle (irreducible-loss proxy)
    oracle = compute_metrics(p_results["oracle: per-tournament question marginal"], y_test)
    closed = 1.0 - (m_model["logloss"] - oracle["logloss"]) / (base["logloss"] - oracle["logloss"])
    print(
        f"Closed-gap to question-oracle:    logloss {closed:.1%}  "
        f"(model {m_model['logloss']:.4f}, baseline {base['logloss']:.4f}, "
        f"oracle {oracle['logloss']:.4f})"
    )


if __name__ == "__main__":
    main()
