# Logit-affine recalibration, 2026-05

## Motivation

After the per-mode lapse rate (`docs/lapse_rate_2026-05.md`) fixed
the very-top tail of the calibration curve, an honest cell-holdout
calibration on the user's own player_id (32919) revealed a
residual S-shaped bias for solo observations:

| Solo p_predicted | mean(y) | bias |
|---:|---:|---:|
| 0.6–0.7 | 0.625 | +0.024 |
| **0.7–0.8** | 0.700 | **+0.050** |
| **0.8–0.9** | 0.800 | **+0.053** |
| 0.9–1.0 | 0.929 | +0.015 (lapse fixed) |

The lapse cap is multiplicative (`p = (1 − π) · p_raw`) and only
flattens the very top; mid-range remained over-predicted by ~5 p.p.
Concretely for player 32919 (Булат, ~10 % solo games), this
inflated his "expected" solo takes by ~80 (442.8 vs actual 364),
making his apparent solo "deficit" 18 % vs a real 11 %, which the
SGD then propagated into a downward θ pull.

## Formulation

A 2-parameter logistic recalibration applied **after** the lapse cap:

```
p_lapse = (1 − π) · p_noisy_or
z       = α + β · logit(p_lapse)
p_final = sigmoid(z)
```

Identity at `α = 0`, `β = 1`.  `β < 1` compresses the curve toward
0.5 (lowers high p, raises low p); `α` is a free shift.  Together
they can fit S-shaped residuals that lapse alone cannot.

Six channels (3 modes × 2 is_solo) × 2 params each = **12 learnable
scalars**, dispatched the same way as the lapse rate.

### Learned values (full DB, 2026-05)

```
offline: team α=+0.019 β=0.944  solo α=+0.159 β=0.900
sync   : team α=−0.056 β=0.954  solo α=−0.044 β=0.836
async  : team α=−0.013 β=0.857  solo α=+0.178 β=0.805
```

`β < 1` everywhere — the noisy-OR (even after lapse) is too
confident in the mid-range.  Strongest compression: async solo
(`β = 0.80`) and sync solo (`β = 0.84`).  α is small for team
channels and meaningful (≈ +0.16) for offline / async solo —
soloists in those formats also need a slight upward shift.

## Gradient

Let `z = α + β · logit(p_lapse)` and `p = sigmoid(z)`.

```
dL/dz       = y − p_final
dL/dα       = dL/dz · 1
dL/dβ       = dL/dz · logit(p_lapse)
dL/dπ       = −dL/dz · β · p_raw / (p_lapse · (1 − p_lapse))
dL/dS       = dL/dz · β · (1 − π) · exp(−S) / (p_lapse · (1 − p_lapse))
```

`dL/dS` chains into the existing `dL/dθ_k`, `dL/db`, `dL/dlog_a`,
`dL/dδ` (all linear in `dL/dS`).  The implementation factors out a
`grad_scale = new_dL/dS ÷ base_dL/dS` and rescales the gradients
returned by `_gradients_nb`.

At identity (α = 0, β = 1), `grad_scale` collapses to the existing
lapse-only formula — verified by smoke test in `model.py`.

## Update

Per-batch mean-gradient ascent, same shape as the lapse update:

```
new_α = α + η_recal · (Σ ∂L/∂α) / n_batch
new_β = β + η_recal · (Σ ∂L/∂β) / n_batch
```

Default `η_recal = 1e-2` (much higher than `η_lapse = 1e-4` —
the per-batch gradient on β is small in magnitude for typical
`p ∈ [0.2, 0.8]` because `logit(p)` is small there, so we need a
larger step to converge in ~18 k batches per pass).

Hard clips: `β ∈ [0.30, 2.00]`, `α ∈ [−3.0, 3.0]`.

## Impact

Cell-holdout backtest (10 % per-cell hold-out):

| | Lapse only | + Recalibration |
|---|---:|---:|
| Overall logloss | 0.5007 | **0.5004** (−0.0003) |
| Overall AUC | 0.8382 | 0.8382 |
| Offline ll | 0.5086 | 0.5077 |
| Sync ll | 0.4861 | 0.4862 |
| Async ll | 0.5318 | **0.5309** (−0.0009) |

Async benefits most because it has the strongest β compression.

### Side effect on δ_size[1]

The model converged to a less extreme `δ_size[1]`:

```
before recal: −0.82  (solo questions seen as 0.82 log-odds easier)
after recal:  −0.62
```

Roughly 0.2 of the "solo easiness boost" was absorbed into
`α_solo > 0`.  The two mechanisms are not redundant though —
δ_size affects `S` (and through it the gradients on θ, b),
whereas α / β only rescale the prediction.

### Per-team-size calibration

Mid-range bias on solo is reduced 3–5×; the very high tail
(0.9–1.0) regressed slightly (recal partially undid the lapse
cap there).  Larger teams (size 5+) were already well-calibrated
and barely moved.

## CLI

`--use-recalibration` / `--no-use-recalibration` toggles the
mechanism (default on).  All 12 learnable scalars are init'd at
identity.

## Practical effect on individual ratings

For player 32919 (the user, 23 solo / 622 team games):

| Metric | Before recal (w_solo=0.7, lapse) | After recal |
|---|---:|---:|
| Solo expected takes | 442.8 | **407.8** |
| Solo deficit | −78.8 (−18 %) | **−43.8 (−11 %)** |
| Team expected | 15286 | 15295 |
| θ | +0.144 | +0.168 |

Solo "expected" dropped by 35 takes — exactly the structural bias
the recalibration was designed to remove.  The remaining −43.8
deficit reflects genuine "Bulat-solo < Bulat-in-team" form gap.

## Why retuning δ_size / w_size_* is no longer needed

Pre-recal, the residual mid-range bias for small teams (sizes 1–3)
suggested a possible re-tune of `eta_size` / `w_size_*` to give
δ_size more freedom.  After recal, those biases dropped to
sub-2 p.p.; sizes 5+ are within ±1 p.p. across all p buckets.
δ_size auto-converged to a less extreme value (n=1: −0.82 → −0.62)
because the recal absorbed part of its job.  A targeted sweep on
`w_size_*` would change overall logloss by < 1e-4.
