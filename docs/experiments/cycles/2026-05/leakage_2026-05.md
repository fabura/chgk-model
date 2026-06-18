# Test-set leakage in the time-split backtest, and the cell-holdout fix

**Status:** discovered 2026-05 after a subscriber pointed it out
(thread on the model release post). The legacy
`backtest()`/`run_sequential` pipeline computes its hold-out logloss
on observations that have already leaked into the question
parameters, so reported metrics — and many ablation comparisons
collected before this date — are optimistic.

## What leaks, exactly

`run_sequential` processes tournaments in chronological order and,
for each tournament, performs:

1. **Decay** of player θ.
2. **Initialise unseen players** from teammates (cold start).
3. **Initialise unseen questions** from the empirical take rate of
   the very observations being predicted:

   ```
   p_take = sum(taken_i) / count(taken_i)   over obs in this tournament
   b_init = log(n_avg) + θ̄ - log(-log(1 - p_take))    # θ̄-aware init
   ```

4. **Record predictions** — predictions for every observation are
   stored *before* the SGD update in step 5.
5. **SGD updates** of θ, b, log_a, δ_size, δ_pos.

Because predictions are recorded between steps 3 and 5, the SGD
updates of step 5 do not contaminate the prediction loss for
observations in the same tournament. *But* the `b_init` of step 3
*does*: the very `taken[i]` we are about to predict is averaged into
`p_take` and pushed into `b_init`, which is then used to compute the
prediction at step 4.

For ChGK questions that appear in only one tournament — the vast
majority, since each pack is played once, the only exception being
sync+async pairs that share a `canonical_q_idx` — `b_init` is
calibrated **exactly** to the test tournament's average take rate.
The model then predicts each team using that calibrated `b` plus
the team's pre-tournament θ. Per-team variance still tests θ, but
the question's base rate is essentially handed to the model for
free.

## Magnitude of the leakage

Measured by training the same configuration twice — once normally,
once with `holdout_obs_fraction=0.10` — and computing logloss on
the same 2.7 M held-out observations under both regimes:

| Configuration | logloss | Brier | AUC |
|---|---:|---:|---:|
| Leaky baseline (current `Config()`) | 0.4850 | 0.1595 | 0.8455 |
| Cell hold-out (same obs, leakage-free) | 0.5083 | 0.1643 | 0.8359 |
| **Δ (leakage)** | **+0.0233 (+4.8 %)** | +0.0048 | −0.0096 |

Per tournament type, the leakage is sharply non-uniform:

| Type | Leaky logloss | Clean logloss | Δ |
|---|---:|---:|---:|
| **offline** | 0.4501 | 0.5213 | **+0.0712 (+15.8 %)** |
| sync | 0.4773 | 0.4893 | +0.0120 (+2.5 %) |
| async | 0.5269 | 0.5468 | +0.0200 (+3.8 %) |

Offline takes the heaviest hit because every offline question is
unique to a single tournament — `b_init` is fitted on that
tournament's teams alone, and there are no paired sync/async
observations to dilute the leak. Sync and async benefit from the
shared `canonical_q_idx` — when the sync half is in train, the
async half evaluates against an honestly trained `b`.

## The cell-holdout fix

`Config.holdout_obs_fraction` (added 2026-05) randomly marks a
fraction of observations as held-out, and `run_sequential` then
excludes those observations from:

* question initialisation (`q_takes`, `q_team_sizes`,
  `q_theta_bars`) — so `b_init` does not see them;
* player initialisation (an obs is hidden from
  `players.initialize_new` so a roster-only appearance doesn't
  count);
* SGD batches in `process_batch_nb` (no θ / b / log_a / δ_size /
  δ_pos updates from held-out cells);
* `tourn_players` — so games and per-player calendar decay aren't
  bumped by hidden cells;
* the teammate-θ shrinkage step.

Predictions are still recorded for every observation (including
held-out ones) using the state at the moment of the tournament,
and `predictions["is_holdout"]` flags which rows were hidden from
training. `backtest()` automatically switches to evaluating on the
held-out subset when `holdout_obs_fraction > 0`.

CLI:

```bash
python -m rating --mode cached --cache_file data.npz \
    --backtest --holdout 0.10 --holdout-seed 42
```

The default remains `0.0` (legacy time-split) so existing scripts
continue to print the historical numbers; the new mode is opt-in.

## Implications for past ablation results

Ablation gains reported before 2026-05 — especially the
2026-04 "noisy-OR init" (`-0.0088 logloss`) and the follow-up
"θ̄-aware init" (`-0.0305 logloss`) — were measured under the
leaky time-split. Both interventions modify `b_init`, i.e. the
exact leakage channel, so the apparent gains may be partly or
mostly improvements in *the quality of the leakage* rather than in
true generalisation. They were re-validated under the
cell-holdout regime in `scripts/exp_holdout_ablations.py`; see
`results/exp_holdout_ablations.csv` for the honest deltas.

## Why not "online evaluation" instead

A reasonable alternative (suggested by the same subscriber): for
each test tournament, predict each team using the *previous*
team's-and-earlier state, then SGD-update before the next team.
Question parameters start from priors (`b = 0`, `log_a = 0`) for
unseen questions, giving a hard penalty to the first team. This is
a strict prequential evaluation and is fully leakage-free, but the
"no init from this question" rule makes the absolute logloss
substantially worse and noisy on small tournaments.

The cell-holdout approach is closer to standard cross-validation
in IRT / matrix-factorisation: hold out a random subset of
(team × question) cells, train on the remainder, evaluate on the
held-out cells. It is leakage-free for the held-out cells while
keeping the rest of training nominal, and produces stable, mode-
comparable numbers without first-team penalties.

## What to use going forward

* For headline numbers (front-page model card, deploys, CI): use
  `--holdout 0.10 --holdout-seed 42`. Those are the honest
  numbers.
* For very large ablation grids, fix the seed across runs so the
  metric is computed on the same cells; otherwise sample noise of
  ±0.0010 logloss obscures small effects.
* For visual comparison to historical numbers, the legacy
  time-split can still be run by leaving `--holdout 0.0`. Mark
  such numbers as "leaky" in any document where they appear next
  to honest numbers.
