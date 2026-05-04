# Calibration on the honest cell-holdout, 2026-05

After the 2026-05 cleanup (`docs/cleanup_2026-05.md`) and with the new
default config (`freeze_log_a=True`, `team_size_max=12`,
`n_extra_epochs=1`, `--holdout 0.10`), we re-ran the calibration
analysis on the leakage-free held-out cells.  Output:
`results/calibration_2026-05_honest.csv`.

## Overall

| `p` bucket | n | mean(p) | mean(y) | bias |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 314 539 | 0.049 | 0.054 | −0.006 |
| 0.1–0.2 | 335 589 | 0.150 | 0.136 | **+0.014** |
| 0.2–0.3 | 323 809 | 0.249 | 0.235 | **+0.015** |
| 0.3–0.4 | 302 008 | 0.349 | 0.340 | +0.009 |
| 0.4–0.5 | 277 444 | 0.449 | 0.449 | +0.001 |
| 0.5–0.6 | 254 166 | 0.549 | 0.558 | −0.009 |
| 0.6–0.7 | 234 741 | 0.649 | 0.662 | **−0.013** |
| 0.7–0.8 | 217 767 | 0.749 | 0.759 | −0.010 |
| 0.8–0.9 | 203 203 | 0.850 | 0.850 | −0.001 |
| **0.9–1.0** | 246 034 | **0.958** | **0.936** | **+0.022** |

Classic S-shape: model is mildly *over-confident in the middle* (says
22 % when actual is 19 %, says 65 % when actual is 67 %) and
**over-confident at the high tail** (says 96 % when actual is 94 %).

## Where the high-tail over-prediction concentrates

Stratifying the `p ∈ [0.9, 1.0]` bucket by team size and tournament
type:

| Slice at p=0.9–1.0 | n | mean(p) | mean(y) | bias |
|---|---:|---:|---:|---:|
| **size = 1 (solo)** | 4 889 | 0.960 | 0.865 | **+0.095** |
| size = 2 | 10 337 | 0.957 | 0.905 | +0.052 |
| size = 3–4 | 33 046 | 0.955 | 0.925 | +0.030 |
| size = 5 | 50 192 | 0.956 | 0.939 | +0.017 |
| size = 6 | 138 433 | 0.959 | 0.943 | +0.016 |
| size = 7+ | 9 137 | 0.960 | 0.933 | +0.027 |
| **async** | 79 050 | 0.959 | 0.921 | **+0.039** |
| offline | 39 421 | 0.963 | 0.931 | +0.032 |
| sync | 127 563 | 0.955 | 0.947 | +0.008 |

The over-prediction is **largest for solo players** (+9.5 p.p.) and
**second-largest for the smallest-team-size async tournaments**.  The
sync mode is essentially calibrated.

## Hypothesised cause: missing lapse rate

The noisy-OR forward `p = 1 − exp(−exp(−b + a·θ))` asymptotes to 1
as `−b + a·θ` grows.  For an "easy" question (small `b`) and a
strong solo player, the model says `p ≈ 0.99`.  Empirically that
player still gets only ~85–90 %, because of:

* **Inattention** (read the question wrong, missed an obvious clue).
* **Time pressure** in a soloed format with no teammates to back-check.
* **Async submission noise** (typos, accidentally hit "skip", got
  distracted between question and answer).
* **Knowledge floor** — even a "trivial" question is sometimes not
  trivial for the specific player.

These produce a **lapse-rate floor**: a non-zero probability of
missing even a question the player "should" get right.  The current
model has no parameter for this; once `−b + a·θ` is large enough
the model says `p ≈ 1` and there is nothing keeping it down.

## Status (2026-05)

Documented as a **known limitation**.  Possible fixes:

1. **Per-mode lapse rate** (`π_offline`, `π_sync`, `π_async`):
   `p_corr = (1 − π) · p`.  Three new learnable parameters; gradient
   updates needed in `process_batch_nb`.
2. **Per-mode logit-affine** (`α_mode`, `β_mode`): six new parameters
   that recalibrate the logit; more general than (1).
3. **Global lapse rate** (one scalar `π`): minimal but ignores the
   sync/async asymmetry.
4. **Post-hoc isotonic calibration** per (mode, size): does not
   change the model.  Adds an inference-time correction.  Could be
   trained on the held-out cells.

None of these is implemented yet — the discussion is open.  For most
downstream uses (player rankings, tournament predictions on real-team
sizes 5–6) the bias is small (±2 p.p.); the issue mostly hurts solo
predictions and the very-easiest async questions.
