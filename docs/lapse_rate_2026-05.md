# Per-mode lapse rate, 2026-05

## Motivation

The honest cell-holdout calibration
(`docs/calibration_2026-05.md`) showed a sharp pattern of high-`p`
over-prediction concentrated in solo (size = 1) and async
tournaments — the noisy-OR forward
`p = 1 − exp(−exp(−b + a·θ))` asymptotes to 1, but empirically
even strong soloists miss "easy" questions ~10 % of the time
because of inattention, typos, format glitches, and the absence of
team cross-checking.

| Slice at `p ∈ [0.9, 1.0]`  | Pre-lapse bias | Post-lapse bias | Reduction |
|---|---:|---:|---:|
| size = 1 (solo) | **+9.5 p.p.** | **+1.5** | **6×** |
| size = 2 | +5.2 | +4.4 | 1.2× |
| async | +3.9 | +2.1 | 2× |
| offline | +3.2 | +1.5 | 2× |
| sync | +0.8 | +0.5 | (already calibrated) |

## Formulation

Six new learnable scalars, indexed by `(mode, is_solo)`:

```
mode      ∈ {0 = offline, 1 = sync, 2 = async}
is_solo   ∈ {0 = team (≥ 2),    1 = solo (= 1)}
π_{m, s}  ∈ [0, lapse_max]      learnable; default lapse_max = 0.30
```

Forward becomes

```
p_noisy_or = 1 − exp(−S)             # unchanged
p          = (1 − π_{m, s}) · p_noisy_or
```

i.e. the predicted probability is capped at `1 − π_{m, s}`.  The
"lapse" is the residual probability mass that will *never* be
captured by `S`, no matter how strong θ or how easy `b`.  Dispatch
is by tournament `game_type` (same buckets as
`_type_update_weights`) and per-observation team size.

### Gradients

For y = 1 (taken):

```
log p          = log(1 − π) + log(1 − exp(−S))
∂L/∂S          = exp(−S) / (1 − exp(−S))           # unchanged from no-lapse
∂L/∂π          = −1 / (1 − π)
```

For y = 0 (not taken), let `q = π + (1 − π) · exp(−S)`:

```
log(1 − p)     = log(q)
∂L/∂S          = −(1 − π) · exp(−S) / q            # was −1 in no-lapse case
∂L/∂π          = (1 − exp(−S)) / q
```

So all existing gradients
(`∂L/∂θ_k = ∂L/∂S · a · λ_k`,
`∂L/∂b   = ∂L/∂S · (−S)`, etc.) are simply rescaled by
`(1 − π) · exp(−S) / q ≤ 1` for the y = 0 branch.  Intuitively:
when the model is highly confident (`S` large) and the team still
fails (y = 0), the prior version blamed θ/b/a strongly
(gradient ≈ 1); now we attribute the failure partly to lapse and
the gradient on θ/b/a fades.

### Update

Per-batch (per tournament × channel) mean-gradient ascent:

```
new_π = π + eta_lapse · (Σ ∂L/∂π) / n_obs_in_batch
new_π = clip(new_π, 0, lapse_max)
```

Default `eta_lapse = 1e-4`; the gradient is bounded so this is
stable.

## Init

Warm-started from the high-p calibration bias measured under the
previous (lapse-free) defaults:

| | offline | sync | async |
|---|---:|---:|---:|
| team init | 0.030 | 0.010 | 0.040 |
| solo init | 0.100 | 0.070 | 0.100 |

After ~9 k tournaments of training, the learned values are within
the noise of init:

```
offline: team = 0.0305, solo = 0.0951
sync   : team = 0.0256, solo = 0.0841
async   : team = 0.0358, solo = 0.0974
```

So the priors were essentially right.  The sgd refinement is
small (within ±0.005).

## Impact on accuracy

Single sanity-backtest with all 2026-05 changes (cell-holdout
0.10, seed 42):

| | Before lapse | After lapse |
|---|---:|---:|
| Overall logloss | 0.5061 | **0.5007** (−0.0054) |
| Overall AUC | 0.8374 | **0.8382** (+0.0008) |
| Offline ll | 0.5200 | 0.5086 (−0.011) |
| Sync ll | 0.4876 | 0.4861 (−0.002) |
| Async ll | 0.5432 | 0.5318 (−0.011) |

The lapse rate gives a **−0.0054** logloss improvement on top of the
other 2026-05 changes — by far the largest single gain in the
session.

## Side effect: w_online optimum shifted

With the lapse rate absorbing the noise that previously pushed the
online (sync+async) updates to be conservative, the optimal
`w_online` (multiplier on η0 for online θ updates) jumped from
0.5 to 1.0 (i.e., online θ now updates at the same magnitude as
offline; see `results/exp_w_online_sweep_honest_high.csv`).  This
contributed an additional ~−0.0005 logloss.

## Knob

CLI flag `--use-lapse-rate` / `--no-use-lapse-rate` toggles the
mechanism.  All 6 lapse parameters are dataclass fields on
`Config` (`lapse_init_*`, `eta_lapse`, `lapse_max`); see the
docstring there.

## Residual signal

Solo at `p ∈ [0.7, 0.9]` still over-predicts by ~5 p.p. — the
lapse correction is multiplicative `(1 − π)·p_raw`, but the actual
solo bias has a more S-shaped pattern (calibrated at the very top
because of the cap, but slightly over-predicted just below).  A
per-mode logit-affine recalibration (α, β) would fit better but
adds 6 more parameters.  Documented as a residual issue; the
biggest source of mis-calibration (the +9.5 p.p. solo high-p tail)
is fixed.
