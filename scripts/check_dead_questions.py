"""Are the high-b worst-questions actual graveyards (0 takes everywhere)?

Loads the cache, identifies:

* "true graveyards" — canonical questions with 0 takes across the
  ENTIRE dataset (train + test).  Their gradient is identically zero
  in noisy-OR; including or excluding them in training has no effect
  on θ or b for other questions.  In test predictions p̂ ≈ 0 and
  y = 0, so they contribute logloss ≈ 0.
* "train-only graveyards" — 0 takes in train, but ≥ 1 take in test.
  These are the actual source of the b ≈ +9.6 clamp pathology and
  CANNOT be excluded; they need to be learned, often via shared
  ``canonical_q_idx`` from a paired sync↔async pack where one side
  had no takes.

Then it cross-checks the top question drifts from the multi-epoch
experiment to confirm which bucket they fall into.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import load_cached


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_file", default="data.npz")
    ap.add_argument("--test_fraction", type=float, default=0.2)
    ap.add_argument(
        "--top_drifts_csv",
        default="results/exp_multi_epoch_top_question_drifts.csv",
    )
    ap.add_argument(
        "--worst_questions_csv",
        default="results/error_analysis/worst_questions.csv",
    )
    args = ap.parse_args()

    print(f"[load] {args.cache_file}")
    arrays, maps = load_cached(args.cache_file)
    q_idx = arrays["q_idx"]
    taken = arrays["taken"]
    game_idx = arrays["game_idx"]

    cq_arr = (
        maps.canonical_q_idx if maps.canonical_q_idx is not None
        else np.arange(maps.num_questions, dtype=np.int32)
    )
    n_cq = int(cq_arr.max()) + 1

    cq_obs = cq_arr[q_idx]

    # Train/test split by date — same rule as backtest().
    gdo = maps.game_date_ordinal
    all_games = np.unique(game_idx)
    if gdo is not None:
        known = all_games[
            np.array([int(gdo[g]) >= 0 for g in all_games], dtype=bool)
        ]
        ordered = known[
            np.argsort(np.array([int(gdo[g]) for g in known]))
        ]
    else:
        ordered = np.sort(all_games)
    n_test = max(1, int(len(ordered) * args.test_fraction))
    test_games = set(int(g) for g in ordered[-n_test:])
    test_mask = np.fromiter(
        (int(g) in test_games for g in game_idx),
        count=len(game_idx),
        dtype=bool,
    )
    train_mask = ~test_mask

    print(
        f"[split] {len(test_games)}/{len(all_games)} test games "
        f"({test_mask.sum():,} test obs vs "
        f"{train_mask.sum():,} train obs)"
    )

    # Per-canonical-q stats.
    n_obs_total = np.bincount(cq_obs, minlength=n_cq)
    n_takes_total = np.bincount(cq_obs, weights=taken, minlength=n_cq)
    n_obs_train = np.bincount(
        cq_obs[train_mask], minlength=n_cq
    )
    n_takes_train = np.bincount(
        cq_obs[train_mask], weights=taken[train_mask], minlength=n_cq
    )
    n_obs_test = np.bincount(
        cq_obs[test_mask], minlength=n_cq
    )
    n_takes_test = np.bincount(
        cq_obs[test_mask], weights=taken[test_mask], minlength=n_cq
    )

    initialized = n_obs_total > 0  # canonical q ever observed
    seen_in_train = n_obs_train > 0
    seen_in_test = n_obs_test > 0

    # === True graveyards ===========================================
    true_grave = (n_takes_total == 0) & initialized
    n_grave = int(true_grave.sum())
    grave_obs = int(n_obs_total[true_grave].sum())
    grave_train_obs = int(n_obs_train[true_grave].sum())
    grave_test_obs = int(n_obs_test[true_grave].sum())

    print("\n=== True graveyards (0 takes everywhere) ===")
    print(f"  canonical questions: {n_grave:,} / {int(initialized.sum()):,} "
          f"({100*n_grave/max(int(initialized.sum()),1):.2f}%)")
    print(f"  total observations:  {grave_obs:,} / {int(initialized.sum() and len(q_idx)):,} "
          f"({100*grave_obs/max(len(q_idx),1):.3f}% of all obs)")
    print(f"  train obs (will be discarded if filtered): {grave_train_obs:,}")
    print(f"  test obs  (would be excluded from metrics): {grave_test_obs:,} "
          f"({100*grave_test_obs/max(int(test_mask.sum()),1):.3f}% of test)")

    # === Train-only graveyards (the actual problem) ================
    train_only_grave = (
        seen_in_train & seen_in_test
        & (n_takes_train == 0) & (n_takes_test > 0)
    )
    n_tog = int(train_only_grave.sum())
    tog_test_obs = int(n_obs_test[train_only_grave].sum())
    tog_test_takes = int(n_takes_test[train_only_grave].sum())
    print("\n=== Train-only graveyards (0 in train, ≥1 in test) ===")
    print(f"  canonical questions: {n_tog:,}")
    print(f"  test observations:   {tog_test_obs:,}")
    print(f"  test takes:          {tog_test_takes:,} "
          f"(test take rate = {tog_test_takes/max(tog_test_obs,1):.2%})")

    # === Cross-check vs top question drifts ========================
    drift_csv = Path(args.top_drifts_csv)
    if drift_csv.exists():
        print(f"\n=== Top {drift_csv.name}: where do they fall? ===")
        with drift_csv.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        n_true = n_train_only = n_normal = 0
        for r in rows:
            cq = int(r["canonical_q_idx"])
            if not (0 <= cq < n_cq):
                continue
            if true_grave[cq]:
                bucket = "TRUE-grave"
                n_true += 1
            elif n_takes_train[cq] == 0 and n_takes_test[cq] > 0:
                bucket = "train-only-grave"
                n_train_only += 1
            else:
                bucket = "normal"
                n_normal += 1
            print(
                f"  cq={cq:>7d} {bucket:>16s} "
                f"train: n={int(n_obs_train[cq]):>4d} takes={int(n_takes_train[cq]):>4d} | "
                f"test: n={int(n_obs_test[cq]):>3d} takes={int(n_takes_test[cq]):>3d} "
                f"(b_base={float(r['b_base']):+.2f} → b_K4={float(r['b_new']):+.2f})"
            )
        print(
            f"  ── summary: {n_true} true-grave, "
            f"{n_train_only} train-only-grave, "
            f"{n_normal} other"
        )

    # === Cross-check vs worst questions (Part 1) ===================
    wq_csv = Path(args.worst_questions_csv)
    if wq_csv.exists():
        print(f"\n=== Top {wq_csv.name}: where do they fall? ===")
        with wq_csv.open() as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        n_true = n_train_only = n_normal = 0
        for r in rows[:30]:
            cq = int(r["cq"])
            if not (0 <= cq < n_cq):
                continue
            if true_grave[cq]:
                bucket = "TRUE-grave"
                n_true += 1
            elif n_takes_train[cq] == 0 and n_takes_test[cq] > 0:
                bucket = "train-only-grave"
                n_train_only += 1
            else:
                bucket = "normal"
                n_normal += 1
            print(
                f"  cq={cq:>7d} {bucket:>16s} "
                f"train: n={int(n_obs_train[cq]):>4d} takes={int(n_takes_train[cq]):>4d} | "
                f"test: n={int(n_obs_test[cq]):>3d} takes={int(n_takes_test[cq]):>3d}"
            )
        print(
            f"  ── summary: {n_true} true-grave, "
            f"{n_train_only} train-only-grave, "
            f"{n_normal} other"
        )

    # === Histograms ================================================
    print("\n=== Take-rate buckets across all canonical questions ===")
    init_idx = np.where(initialized)[0]
    rates = n_takes_total[init_idx] / np.maximum(n_obs_total[init_idx], 1)
    edges = [0, 1e-9, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0001]
    labels = [
        "= 0%", "(0,1%]", "(1,5%]", "(5,10%]", "(10,25%]",
        "(25,50%]", "(50,75%]", "(75,95%]", "(95,100%]",
    ]
    counts, _ = np.histogram(rates, bins=edges)
    for lab, c in zip(labels, counts):
        print(f"  {lab:>10s}: {int(c):>8d}")

    # === Effect on metrics ========================================
    # Estimate the change in test logloss if we excluded true graveyards
    # from the test set (predicting 0% take).  Since they have y=0 and
    # p̂≈0, contribution to logloss is ≈ −log(1 − ε) per obs ≈ 0.
    print("\n=== If we excluded TRUE graveyards from training... ===")
    print(
        "  - they contribute zero gradient (every obs has y=0 and "
        "p̂≈0, so ∂L/∂· ≈ 0)"
    )
    print(
        "  - removing them frees no other parameter — they're "
        "isolated canonical_q_idx slots; nothing else changes"
    )
    print(
        "  - test obs of true graveyards have logloss ≈ "
        "−log(1 − p̂) ≈ {:.4f} per obs, "
        "so removing them from the test denominator would "
        "marginally NUDGE test logloss UP (by leaving the harder "
        "obs in)".format(-np.log(1 - 1e-4))
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
