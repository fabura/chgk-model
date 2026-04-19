# Async / Sync Mode Experiments

This report captures the checked hypotheses, assumptions, measured
results, and follow-up ideas around tournament-mode handling in the
sequential `rating/` model.

## Why this work was needed

The previous sequential model had only one tournament offset `delta_t`
and treated async tournaments mainly by downweighting their updates.
That was too weak for an important domain fact:

- async tournaments are not just "another event type"
- teams can see the question text longer
- some teams use external help
- as a result, async observations are systematically different from
  sync/offline observations

The question was not whether async should be handled differently, but
how to encode that difference without importing pseudo-external
difficulty estimates built from the same answer data.

## Hypotheses

### H1. `true_dl` / `ndcg`-style difficulty estimates should not be used as model input

Status: accepted as a modeling principle.

Reasoning:

- these estimates are derived from the same core answer data that the
  model already sees
- they do not bring genuinely external information
- using them risks injecting an opaque pre-aggregation of the target
  signal back into the model

Decision:

- do not use `true_dl` or `ndcg` as explanatory variables in the
  sequential model
- infer tournament-mode effects directly from the model inputs

### H2. Async should be modeled in the probability model, not only in the optimizer weights

Status: accepted and implemented.

Old approach:

- async mainly had weaker update weight (`w_online`)
- this changed how much the model trusted async observations
- but it did not explicitly represent that async is a different
  observation regime

New approach:

- decompose tournament shift as `delta_t = mu_type[type_t] + eps_t`
- `mu_type` is the systematic effect of the mode
- `eps_t` is the residual shift of a specific tournament

This lets the model explain persistent async/sync differences inside the
probability model rather than only through weaker learning.

### H3. Async should affect player/question updates more weakly than offline/sync

Status: accepted and implemented.

Reasoning:

- async is useful for learning that the mode is different
- but it is a noisier source of evidence about player strength and
  question discrimination

Decision:

- keep `w_online` for player updates
- use additional weaker async weights for question updates
- use separate async weights for mode-effect and residual-offset updates

### H4. The first implementation was too aggressive

Status: confirmed by backtests.

The first defaults improved logloss/Brier strongly but reduced AUC too
much. That suggested the mode effects were absorbing too much structure
that should remain in player/question ranking.

## Implementation

### Parameterization

The sequential model now uses:

- `delta_t = mu_type[type_t] + eps_t`
- `mu_type` for global mode effects (`offline`, `sync`, `async`)
- `eps_t` for per-tournament residual offsets

Design choices:

- `offline` is the baseline (`mu_offline = 0`)
- residuals are centered by type within each week
- async uses weaker update weights than offline/sync

### Files changed

- `rating/tournaments.py`
  - stores `mu_type` and `eps`
  - exposes `total_delta(g)`
  - centers residuals by mode
- `rating/model.py`
  - updates player/question params and mode/residual offsets separately
- `rating/engine.py`
  - introduces per-parameter update weights by tournament type
  - prints learned mode offsets in summaries
- `rating/__main__.py`
  - exposes the new hyperparameters in CLI

## Measured results

All numbers below come from actual backtests run on `data.npz`.

### Baseline used for comparison

`old_like` is an approximation of the previous behavior:

- no learned mode effect
- only residual tournament offsets
- async observations downweighted but not explicitly modeled as a
  different regime

### Full backtest: baseline vs first new defaults

| config | logloss | brier | auc |
|---|---:|---:|---:|
| old_like | 0.784457 | 0.251053 | 0.826148 |
| new_default | 0.611870 | 0.202597 | 0.803281 |

Interpretation:

- calibration/probability quality improved a lot
- ranking quality (`AUC`) degraded
- the new idea was useful, but the first defaults were too strong

Mode offsets learned by `new_default`:

- `sync = +0.639304`
- `async = +0.325822`

These signs did not fully match the original domain expectation that
async should usually look easier. This was treated as a warning that the
mode updates were still too aggressive and were entangled with other
parameters.

### Fast search on last 300 tournaments

We ran a focused search over:

- `eta_mu`
- `eta_eps`
- `reg_mu_type`
- `reg_eps`
- `w_online_questions`
- `w_online_log_a`
- `w_async_mode`
- `w_async_residual`

The search showed a consistent pattern:

- smaller `eta_mu`
- smaller `eta_eps`
- stronger regularization on `mu_type` and `eps`
- much weaker async weight for `log_a`

gave a better logloss/AUC trade-off than the first defaults.

### Best candidates on the full dataset

| config | logloss | brier | auc |
|---|---:|---:|---:|
| new_default | 0.611870 | 0.202597 | 0.803281 |
| t5 | 0.600466 | 0.201015 | 0.812344 |
| t6 | 0.601880 | 0.201788 | 0.814490 |

Interpretation:

- both `t5` and `t6` beat `new_default` on all three metrics
- `t5` is best for `logloss` / `Brier`
- `t6` is best for `AUC`

### Chosen default: `t6`

`t6` was chosen as the new default because it keeps almost all of the
probability-quality improvement while recovering more ranking signal.

Chosen values:

- `w_online_questions = 0.45`
- `w_online_log_a = 0.05`
- `w_async_mode = 0.3`
- `w_async_residual = 0.6`
- `eta_mu = 0.005`
- `eta_eps = 0.03`
- `reg_mu_type = 0.10`
- `reg_eps = 0.20`

Learned mode offsets on the full dataset:

- `sync = +0.330345`
- `async = +0.060908`

This is more conservative than the first defaults and closer to the
intended interpretation: async is still different, but the model is no
longer allowed to explain too much with that difference alone.

## Checked assumptions

### What looks reliable

- async/sync/offline do behave differently enough that a single shared
  tournament offset is too coarse
- learning a mode-level effect improves forecast quality
- async should update `log_a` much more cautiously than offline/sync
- strong regularization on mode and residual offsets is helpful

### What is still uncertain

- the exact semantic meaning of learned `mu_sync` and `mu_async`
- whether part of the async effect should be modeled as an additive
  difficulty shift or as an observation-noise / contamination process
- whether the same parameterization is optimal for all eras of the data

## Useful follow-up ideas

### 1. Report metrics by tournament type in code, not only manually

Backtest currently returns aggregate metrics only. Adding built-in
per-type metrics would make future comparisons much faster and safer.

### 2. Test calibration curves by type

The strongest improvements so far were in logloss/Brier, which suggests
better calibration. This should be checked explicitly for:

- offline
- sync
- async

### 3. Add stronger shrinkage for `log_a`

Async appears especially noisy for question discrimination. A global or
count-aware shrinkage of `log_a` toward zero may improve stability.

### 4. Improve team model

The noisy-OR aggregation is still purely additive over players. This is
likely a larger remaining source of structural error than fine-tuning the
mode offsets.

Potential directions:

- team-size bias
- diminishing returns by roster size
- explicit small-roster correction

### 5. Check paired sync/async packages directly

`canonical_q_idx` creates a strong quasi-experimental setup. A useful
future diagnostic:

- compare predictions on packages seen in both sync and async
- measure how much of the difference is captured by `mu_type`
- inspect whether `eps_t` stays small on such pairs

### 6. Consider a contamination-style async term later

If additive mode offsets remain hard to interpret, a later extension
could model async as partly contaminated observations, e.g. an extra
mode-level lift in observed solve probability rather than only a
difficulty shift.

This should be a second-stage experiment, not the default right now.

### 7. Revisit tuning objective

`logloss` and `AUC` moved in different directions. Future searches should
not optimize only one metric.

Reasonable options:

- optimize logloss with an AUC floor
- optimize a weighted combination of logloss and AUC
- compare configs on calibration + ranking dashboards

## Tried-but-not-helpful: making `ε_t` more sensitive to elite events (2026-04)

Symptom: on elite offline events (e.g. ЧР 11749, 60 teams × 90 questions),
the model over-predicts each team's score by ~7-8 takes.  The realised
average is 35.5, the expected (using all model parameters: `b`, `a`,
`δ_t = μ_type + ε_t`, `δ_size`, `δ_pos` and θ-at-start-of-tournament from
history) is ~43.  The learned `ε_t` for that tournament is only +0.10.

Two interventions were tested and reverted.

### Lower `reg_eps` from 0.20 → 0.05

The per-step shrinkage `eps ← eps · (1 − η·reg_eps)` is applied on every
observation, so for tournaments with thousands of obs it could in principle
crush ε_t toward 0 faster than it grows. Lowering `reg_eps` to 0.05 left
`ε_t` essentially unchanged (std 0.128 → 0.128, range [-0.69, +0.62] → ~same).

Reason: weekly within-type centering `tournaments.center(games_this_week)`
dominates: it forces `mean_within_week_within_type(ε) = 0` after every week,
so any persistent build-up is removed regardless of the per-step shrinkage.

### Weighted within-week centering

Replaced unweighted mean with a weight-by-`n_obs(g)` mean in `center()`,
hoping a small weak tournament could no longer drag down the bar set by a
large hard one. Result: `ε_t` std rose marginally (0.128 → 0.137) and the
overall residual improved (-2.45 → -2.12 takes/team), but **`ε_t` for the
biggest tournaments became smaller**, not larger:

- ЧР 11749: ε_t went from +0.10 → +0.04, expected 43.3 → 44.8, delta worsened.

Reason: when a single tournament dominates the weekly weighted mean, that
mean approaches its own ε. Subtracting it then sets that tournament's ε to
≈ 0. This perverse self-cancellation makes weighted centering actively
worse for the very tournaments we wanted to lift.

### Conclusion

Within-week centering is the dominant identifiability constraint on `ε_t`
and is fundamentally at odds with letting individual elite tournaments
carry a large positive residual. Realistic options for future work:

- replace centering with an L2 prior on `ε_t` and a separate L2 on
  `μ_type` (both shrink toward 0, so identifiability is statistical
  rather than algebraic);
- multi-epoch training so `ε_t` has more passes to grow;
- a per-tournament free intercept on `b` (essentially, let `b` for every
  question of the same tournament inherit a tournament-level offset that
  is L2-regularised but never centered).

These are deeper refactors than a hyperparameter knob and are deferred.

## Recommended current default

Use the code defaults from `Config()` / CLI defaults unless there is a
specific experiment.

Equivalent explicit command:

```bash
python -m rating --mode cached --cache_file data.npz \
  --eta_mu 0.005 \
  --eta_eps 0.03 \
  --reg_mu_type 0.10 \
  --reg_eps 0.20 \
  --w_online_questions 0.45 \
  --w_online_log_a 0.05 \
  --w_async_mode 0.3 \
  --w_async_residual 0.6
```
