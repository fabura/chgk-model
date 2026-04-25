"""
Time-based backtesting.

Split tournaments chronologically: the last *test_fraction* are evaluated
with predictions made *before* the model sees them (prequential evaluation).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from rating.engine import Config, SequentialResult, run_sequential


def compute_metrics(
    pred_p: np.ndarray,
    actual_y: np.ndarray,
) -> dict[str, float]:
    """Logloss, Brier score, AUC from parallel arrays."""
    eps = 1e-15
    p = np.clip(pred_p, eps, 1.0 - eps)
    y = actual_y.astype(np.float64)

    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    logloss = float(ll.mean())
    brier = float(np.mean((p - y) ** 2))

    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y, pred_p))
    except ImportError:
        auc = float("nan")

    return {"logloss": logloss, "brier": brier, "auc": auc}


def backtest(
    arrays: dict[str, np.ndarray],
    maps,
    cfg: Config = Config(),
    *,
    test_fraction: float = 0.2,
    verbose: bool = True,
) -> dict:
    """Run sequential training with time-based out-of-sample evaluation.

    The last *test_fraction* of tournaments (by date) form the test set.
    Predictions are recorded *before* the model updates on each
    tournament, so test-set predictions are genuinely out-of-sample
    with respect to player ratings.

    Returns a dict with logloss / brier / auc and supporting counts.
    """
    result = run_sequential(
        arrays,
        maps,
        cfg,
        verbose=verbose,
        collect_predictions=True,
    )

    if result.predictions is None or len(result.predictions["pred_p"]) == 0:
        empty = {
            "logloss": float("nan"),
            "brier": float("nan"),
            "auc": float("nan"),
        }
        if verbose:
            print("No predictions collected — cannot compute metrics.")
        return empty

    pred_p = result.predictions["pred_p"]
    actual_y = result.predictions["actual_y"]
    pred_game = result.predictions["game_idx"]
    pred_thbar = result.predictions.get("team_theta_mean")

    # --- determine test games (last test_fraction by date) ---
    gdo = getattr(maps, "game_date_ordinal", None)
    all_games = np.unique(pred_game)

    if gdo is not None:
        known = all_games[
            np.array([gdo[g] >= 0 for g in all_games], dtype=bool)
        ]
    else:
        known = np.array([], dtype=np.int32)

    if len(known) >= 2:
        ordered = known[np.argsort(np.array([gdo[g] for g in known]))]
    else:
        ordered = np.sort(all_games)

    n_test = max(1, int(len(ordered) * test_fraction))
    test_games = set(int(g) for g in ordered[-n_test:])

    test_mask = np.array(
        [int(g) in test_games for g in pred_game], dtype=bool
    )

    if not test_mask.any():
        return {
            "logloss": float("nan"),
            "brier": float("nan"),
            "auc": float("nan"),
        }

    p_test = pred_p[test_mask]
    y_test = actual_y[test_mask]

    metrics = compute_metrics(p_test, y_test)
    metrics["n_test_obs"] = int(test_mask.sum())
    metrics["n_train_obs"] = int((~test_mask).sum())
    metrics["n_test_games"] = len(test_games)
    metrics["n_total_games"] = len(all_games)
    metrics["result"] = result

    # ---- per-tournament-type metrics on the test set ------------------
    # Useful diagnostic: see whether the global score moves because of
    # async calibration only, or because of genuine sync/offline
    # improvements as well.
    by_type: dict[str, dict[str, float]] = {}
    gtype_arr = getattr(maps, "game_type", None)
    if gtype_arr is not None:
        test_pred_game = pred_game[test_mask]
        test_types = np.array(
            [str(gtype_arr[g]) if g < len(gtype_arr) else "offline" for g in test_pred_game],
            dtype=object,
        )

        def _bucket(g: str) -> str:
            if "async" in g:
                return "async"
            if "sync" in g:
                return "sync"
            return "offline"

        buckets = np.array([_bucket(g) for g in test_types], dtype=object)
        for name in ("offline", "sync", "async"):
            mask = buckets == name
            n = int(mask.sum())
            if n == 0:
                by_type[name] = {
                    "n_obs": 0,
                    "logloss": float("nan"),
                    "brier": float("nan"),
                    "auc": float("nan"),
                }
                continue
            sub = compute_metrics(p_test[mask], y_test[mask])
            sub["n_obs"] = n
            by_type[name] = sub
        metrics["by_type"] = by_type

    # ---- per-hardness-quartile metrics on the test set ----------------
    # Bucket test tournaments by mean pre-tournament θ of the rosters
    # that played them.  q1 = weakest-roster quartile,
    # q4 = strongest-roster (Vyshka, Гран-при и т.п.).  Quartile cuts
    # are done over tournaments (each weighted equally), then metrics
    # are computed on observations within the bucket.
    by_hardness: dict[str, dict[str, float]] = {}
    if pred_thbar is not None and len(pred_thbar) == len(pred_p):
        thbar_test = pred_thbar[test_mask]
        game_test = pred_game[test_mask]
        # Per-tournament average team θ.
        unique_g, inv = np.unique(game_test, return_inverse=True)
        sums = np.zeros(len(unique_g), dtype=np.float64)
        cnts = np.zeros(len(unique_g), dtype=np.int64)
        np.add.at(sums, inv, thbar_test)
        np.add.at(cnts, inv, 1)
        per_g_thbar = sums / np.maximum(cnts, 1)
        # Quartile cuts (avoid duplicate edges).
        cuts = np.unique(np.quantile(per_g_thbar, [0.25, 0.5, 0.75]))
        # Map each test obs's tournament to its quartile (1..4).
        g_to_q = np.searchsorted(cuts, per_g_thbar, side="right") + 1
        obs_q = g_to_q[inv]
        # Cut summary (mean θ̄ in each bucket) for printout.
        q_edges = list(np.round(cuts, 3))
        for q in (1, 2, 3, 4):
            mask = obs_q == q
            n = int(mask.sum())
            if n == 0:
                by_hardness[f"q{q}"] = {
                    "n_obs": 0,
                    "n_games": 0,
                    "logloss": float("nan"),
                    "brier": float("nan"),
                    "auc": float("nan"),
                    "mean_team_theta": float("nan"),
                }
                continue
            sub = compute_metrics(p_test[mask], y_test[mask])
            sub["n_obs"] = n
            sub["n_games"] = int((g_to_q == q).sum())
            sub["mean_team_theta"] = float(per_g_thbar[g_to_q == q].mean())
            by_hardness[f"q{q}"] = sub
        metrics["by_hardness"] = by_hardness
        metrics["hardness_cuts"] = q_edges

    if verbose:
        print(f"\n{'=' * 50}")
        print(
            f"BACKTEST  "
            f"({len(test_games)} test tournaments, "
            f"{test_mask.sum()} test obs)"
        )
        print(f"  Logloss : {metrics['logloss']:.4f}")
        print(f"  Brier   : {metrics['brier']:.4f}")
        print(f"  AUC     : {metrics['auc']:.4f}")
        if by_type:
            print(f"  ----- by tournament type -----")
            for name in ("offline", "sync", "async"):
                m = by_type[name]
                if m["n_obs"] == 0:
                    continue
                print(
                    f"  {name:7s}: n={m['n_obs']:>9d}  "
                    f"logloss={m['logloss']:.4f}  "
                    f"Brier={m['brier']:.4f}  "
                    f"AUC={m['auc']:.4f}"
                )
        if by_hardness:
            cuts_str = " | ".join(
                f"{c:+.2f}" for c in metrics.get("hardness_cuts", [])
            )
            print(
                f"  ----- by roster strength quartile (θ̄ cuts: "
                f"{cuts_str}) -----"
            )
            for q in (1, 2, 3, 4):
                m = by_hardness.get(f"q{q}")
                if not m or m["n_obs"] == 0:
                    continue
                print(
                    f"  q{q} (θ̄≈{m['mean_team_theta']:+.3f}): "
                    f"n={m['n_obs']:>9d} games={m['n_games']:>4d}  "
                    f"logloss={m['logloss']:.4f}  "
                    f"Brier={m['brier']:.4f}  "
                    f"AUC={m['auc']:.4f}"
                )
        print(f"{'=' * 50}")

    return metrics
