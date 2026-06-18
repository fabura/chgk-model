# Temperature-scaled credit & η_teammate bump experiments, 2026-06

Three hypotheses tested on the honest cell-holdout (10 %, seed 42),
aimed at reducing the noisy-OR identifiability gap for perennial
floor players in stable rosters.  **All three were rejected or
showed no measurable gain.**  The code changes were reverted;
only the results CSVs and this doc remain.

Source artefacts:

- `results/exp_temperature_sweep.csv` — symmetric τ sweep
- `results/exp_temperature_asym.csv` — asymmetric win/loss test
- `results/exp_eta_teammate_high.csv` — high-η sweep

---

## Hypothesis A — Temperature-scaled credit (REJECTED)

**Idea.** The noisy-OR gradient `∂L/∂θ_k ∝ λ_k = exp(a·θ_k − b − δ)`
concentrates ~70 % of credit on the strongest player (θ ≈ +1) and
~2 % on the weakest (θ ≈ −1.5).  Replace `λ_k` with
`λ'_k = λ_k ** (1/τ)` in the gradient only (forward pass unchanged),
so τ > 1 redistributes credit more evenly.

**Implementation** (reverted).  Added `temperature` parameter to
`Config` and `process_batch_nb`.  Before calling `_gradients_nb`,
computed `lam_tau = lam ** (1/temperature)` when `temperature != 1.0`.

**Results.**

| τ | logloss | AUC | Chernukha θ | Rekshinskaya | Monina |
|---:|---:|---:|---:|---:|---:|
| 1.0 (baseline) | **0.5018** | **0.8373** | −0.392 | −0.154 | −0.192 |
| 1.5 | 0.5431 | 0.8053 | −0.149 | −0.023 | +0.060 |
| 2.0 | 0.6753 | 0.6247 | +0.761 | +0.051 | +2.438 |
| 3.0 | 0.6822 | 0.6055 | +0.749 | +5.256 | +2.956 |
| 4.0 | 0.6818 | 0.6031 | +1.005 | +5.146 | +5.986 |

**Conclusion.** Any τ > 1 monotonically destroys predictive quality.
The noisy-OR gradient is the exact MLE gradient; distorting it harms
optimisation.  This mirrors the `credit_uniform_mix` result from
`docs/experiments/cycles/2026-05/roster_sticking_2026-05.md` (Hypothesis 3) — uniform credit
blend also gave ΔLogloss = 0.00000 with zero benefit.  The model's
credit attribution is statistically correct; the identifiability
problem for floor players cannot be solved by distorting the
gradient.

---

## Hypothesis D — Asymmetric win/loss temperature (REJECTED)

**Idea.** The symmetric temperature flattened credit on both wins
and losses.  An asymmetric variant — τ_win > 1 (spread credit on
taken questions) but τ_loss = 1 (exact blame on misses) — might
redistribute credit without destroying the loss signal.  The
intuition: a taken question is ambiguous ("who contributed?")
while a miss at high confidence is a clear signal ("even the
strongest didn't know").

**Implementation** (reverted).  Added `temperature_win` and
`temperature_loss` to `Config` and `process_batch_nb`.
Per-observation dispatch: `lam_tau = lam ** (1/τ_win)` for y = 1,
`lam ** (1/τ_loss)` for y = 0.

**Results.**

| Config | logloss | AUC | Chernukha θ |
|---|---:|---:|---:|
| baseline τ=1.0 | **0.5018** | **0.8373** | −0.392 |
| τ_win=2, τ_loss=1 | 0.5879 | 0.7683 | +1.685 |
| τ_win=1, τ_loss=2 | 0.7610 | 0.5498 | −10.000 (clipped) |

**Conclusion.** Asymmetry creates a one-way ratchet:
- τ_win > 1, τ_loss = 1: weak players get more credit on wins,
  normal blame on losses → θ drifts up without bound.
- τ_win = 1, τ_loss > 1: strong players get normal credit, extra
  blame → θ collapses to the −10.0 floor.

Neither direction preserves the gradient's statistical consistency.
**Rejected.**

---

## Hypothesis E2 — η_teammate bump to 0.03…0.07 (NO GAIN)

**Idea.** The per-tournament teammate shrinkage
`θ_k -= η · (θ_k − mean_team_θ)` is the proven lever against
identifiability gaps.  The current default `η = 0.02` was picked from
a sweep up to 0.03 (`results/exp_eta_teammate_sweep_honest.csv`).
Test whether pushing higher (0.05, 0.07) further helps floor players
without hurting logloss.

**Results.**

| η_teammate | logloss | AUC | Chernukha θ | Rekshinskaya | Monina |
|---:|---:|---:|---:|---:|---:|
| 0.02 (current) | **0.5018** | 0.83726 | −0.392 | −0.154 | −0.192 |
| 0.03 | **0.5018** | **0.83728** | −0.372 | −0.129 | −0.173 |
| 0.05 | 0.5019 | 0.83716 | −0.355 | −0.103 | −0.149 |
| 0.07 | 0.5022 | 0.83696 | −0.353 | −0.093 | −0.137 |

**Conclusion.** The η = 0.02 plateau extends to 0.03 (ΔLogloss = 0
within 5 significant digits), but:
- The 0.02 → 0.03 improvement is below noise (±0.0005).
- At 0.05 logloss starts regressing (−0.0001).
- At 0.07 the regression is measurable (−0.0004).

The current default `eta_teammate = 0.02` is already at the optimum.
**No change recommended.**

---

## Summary

| Hypothesis | Verdict | Key metric | Notes |
|---|---|---|---|
| A — temperature τ | ❌ Rejected | ll +4.1 % at τ=1.5 | Monotonically degrades; model collapses at τ ≥ 2 |
| D — asymmetric τ | ❌ Rejected | ll +17.6 % / +51.7 % | One-way ratchet biases θ distribution |
| E2 — η_teammate bump | ❌ No gain | ll flat 0.02…0.03 | Current default is optimal; η > 0.03 regresses |

**Lesson.** The noisy-OR gradient is the exact MLE gradient.  Any
heuristic that distorts it — whether symmetric (τ), asymmetric
(τ_win/τ_loss), or additive (`credit_uniform_mix`) — either leaves
logloss unchanged at zero predictive benefit or actively degrades it.
The identifiability problem for floor players is a *data limitation*
(team-level observations), not a *gradient problem*.  The model
correctly reflects that the data cannot distinguish between "this
player is weak" and "this player plays with strong teammates."
Mechanisms that encode a structural prior (η_teammate,
cold_init_theta, lapse, recalibration) are the right tool;
gradient-distorting heuristics are not.
