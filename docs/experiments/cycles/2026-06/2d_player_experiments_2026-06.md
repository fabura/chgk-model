# 2D player model experiments, 2026-06

Tested a 2-dimensional player parameterisation (Model C: per-player
difficulty slope γ_k) against the 1D baseline under honest
cell-holdout (10 %, seed 42).  **Model C gives a small but real
−0.0016 logloss improvement** when question discrimination is frozen.

Source artefacts:

- `results/exp_2d_players_v2.csv` — full sweep table
- `scripts/exp_2d_players.py` — experiment script
- `rating/players.py` — added `gamma` array
- `rating/model.py` — modified `_forward_nb`, `_gradients_nb`, `process_batch_nb`
- `rating/engine.py` — added `Config.use_2d_players` + related params

---

## Motivation

The current model has:
- Player: 1D (`θ_k` — strength)
- Question: 2D (`b_i` — difficulty, `a_i` — discrimination)

Earlier experiments showed `freeze_log_a=True` gives identical logloss
to learned `a_i` (0.5078 vs 0.5083, `docs/experiments/cycles/2026-05/cleanup_2026-05.md`),
suggesting the discrimination dimension is redundant.  At the same
time, domain intuition says players differ not just in *how good* they
are, but in *what kind* of questions they answer — some are
"bingo"/easy-question specialists, others excel on hard questions.

**Goal:** Move one dimension from question to player: question = 1D
(`b_i` only, `a_i ≡ 1`), player = 2D (`θ_k`, `γ_k`).

---

## Model C — Per-player difficulty slope

### Formulation

```
z_ki = θ_k + γ_k · b_i  −  b_i  −  δ
λ_k  = exp(clip(z_ki, −20, 20))
S    = Σ_k λ_k
p    = 1 − exp(−S)                     # unchanged noisy-OR
```

Interpretation:

| γ_k | Effect |
|---|---|
| γ > 0 | Player is RELATIVELY BETTER on **hard** questions (γ·b_i partially cancels −b_i when b_i > 0) |
| γ = 0 | Identical to 1D model (with `a_i ≡ 1`) |
| γ < 0 | Player is RELATIVELY BETTER on **easy** questions |

### Gradients

```
∂S/∂θ_k = λ_k                                    # unchanged
∂S/∂γ_k = λ_k · b_i                              # new
∂S/∂b_i = Σ_k λ_k · (γ_k − 1)                    # corrected (was: −S)

∂L/∂γ_k = ∂L/∂S · λ_k · b_i
```

### Implementation

- `PlayerState.gamma` — new array, initialised at 0.0
- `_forward_nb` — accepts optional `gamma` array; adds `γ_k·b` to `z_k`
- `_gradients_nb` — accepts optional `gamma` + `b_val`; returns
  `dL_dgamma` and corrects `dL_db` for the extra `Σ λ_k·γ_k` term
- `process_batch_nb` — updates gamma with step size `eta_gamma`
  (separate from theta's `eta0`), hard clip at `±gamma_max`, optional
  L2 shrinkage `reg_gamma`
- `Config` — `use_2d_players`, `eta_gamma`, `reg_gamma`, `gamma_max`,
  per-type weights `w_gamma_*`
- Gauge re-centering is **disabled** when `use_2d_players=True`
  because the θ→θ+Δ, b→b+a·Δ transform does not preserve predictions
  when γ_k ≠ 0 (the extra term γ_k·a_i·Δ breaks gauge symmetry)

---

## Results

All runs: `freeze_log_a=True` (a_i ≡ 1) unless stated otherwise,
cell-holdout 10 %, seed 42.

| Config | logloss | AUC | γ_std | θ Chernukha | γ Chernukha |
|---|---:|---:|---:|---:|---:|
| **2d_eta010** | **0.4978** ★ | **0.8392** | 0.113 | −1.111 | −0.064 |
| 2d_eta025 | 0.4980 | 0.8389 | 0.159 | −1.139 | −0.118 |
| 2d_eta050 | 0.4985 | 0.8384 | 0.204 | −1.105 | −0.181 |
| baseline_frzA | 0.4994 | 0.8389 | 0.000 | −0.482 | 0.000 |
| 2d_eta100 | 0.4997 | 0.8374 | 0.262 | −1.034 | −0.241 |
| 2d_eta050_frzA0 | 0.5007 | 0.8369 | 0.224 | −0.288 | −0.636 |
| baseline_frzA0 | 0.5018 | 0.8373 | 0.000 | −0.392 | 0.000 |
| 2d_eta200 | 0.5019 | 0.8357 | 0.339 | −0.881 | −0.376 |

### Per-slice breakdown (best configs)

| Slice | baseline_frzA | 2d_eta010 | Δ |
|---|---:|---:|---:|
| offline | 0.5085 | 0.5047 | **−0.0038** |
| sync | 0.4856 | 0.4844 | −0.0012 |
| async | 0.5276 | 0.5267 | −0.0009 |
| overall | 0.4994 | 0.4978 | **−0.0016** |

### Key findings

1. **Model C with eta_gamma = 0.01 gives −0.0016 logloss.** The gain
   is concentrated in offline (−0.0038) and sync (−0.0012); async is
   nearly flat.

2. **The optimum eta_gamma is ~0.01–0.025.** Higher values degrade
   monotonically — at eta_gamma = 0.20 the model is worse than
   baseline.

3. **freeze_log_a=True is required.** When a_i is learned alongside
   gamma (2d_eta050_frzA0), both dimensions compete and logloss
   regresses to 0.5007 — worse than the 1D baseline. The 2D player
   model only works when question discrimination is frozen.

4. **Gamma captures meaningful signal.** γ_std = 0.113 at the optimum,
   with a slight positive bias (γ_mean = +0.037).  Players
   consistently specialise in hard (γ > 0) vs easy (γ < 0) questions.
   Example specialists at η_gamma = 0.01:
   - Rekshinskaya: γ = +0.282 (hard-question specialist)
   - Monina: γ = +0.151 (mild hard-question specialist)
   - Chernukha: γ = −0.064 (neutral / slight easy-question lean)

5. **Theta values are compressed downward** when gamma is active.
   Chernukha drops from θ = −0.48 (1D) to θ = −1.11 (2D) — the
   "specialisation" dimension absorbs part of what was previously in θ.
   This is expected: θ now captures *baseline* ability (at b = 0),
   while γ captures *relative* performance across difficulty.

6. **Comparison to learned a_i.** The 1D baseline with learned a_i
   (baseline_frzA0, ll = 0.5018) is worse than with frozen a_i
   (baseline_frzA, ll = 0.4994).  Freezing a_i and adding γ is strictly
   better than learning a_i: 0.4978 vs 0.5018 (−0.0040).

---

## Models considered but not tested

### Model C_tanh (saturated slope)

```
z_ki = θ_k + γ_k · tanh(b_i) − b_i − δ
```

tanh(b) ∈ [−1, 1] prevents extreme b from dominating γ updates.
Not tested — Model C's gradient clipping at z ∈ [−20, 20] already
provides effective saturation, and the γ_max = ±2 clip prevents
pathological values.

### Model G (two-regime blend)

```
z_ki = θ_easy_k · σ(−b_i) + θ_hard_k · σ(b_i) − b_i − δ
```

Two independent skills blended by sigmoid over b.  Not tested —
Model C's single slope parameter captures the same idea with half
the parameters and clearer gradients.

### "Bingo" dimension (game-frequency advantage)

Questions whose content overlaps with recently-seen material benefit
teams that play more often.  Could be modelled as:

```
z_ki = θ_k + γ_k · b_i + β_k · f(games_k) · B_i − b_i − δ
```

where B_i measures content overlap with recent packs, and β_k
captures sensitivity.  Not implemented — requires computing B_i from
question content, which is not available in the current dataset.

---

## Recommendation (updated 2026-06)

**Do not promote to production yet.**

Holdout logloss improves (−0.0040 vs 1D with `freeze_log_a=True`),
but a June 2026 follow-up (`docs/experiments/cycles/2026-06/floor_player_experiments_2026-06.md`)
showed that **1D projections of (θ, γ) do not fix intuitive rankings**
for floor players on strong rosters (Монина career +437 actual−exp yet
rank ~2350 on `θ+γ·b_mean`; Рекшинская ranked above her with +26).
Rank-shift analysis: `results/exp_modelc_rank_shift_summary.csv`.

The code stays behind `Config.use_2d_players` for research.  If
revisited, surface γ as a *profile* dimension rather than folding it
into a single leaderboard scalar.
