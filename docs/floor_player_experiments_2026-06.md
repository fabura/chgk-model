# Floor-player / roster-sticking experiments, 2026-06

Follow-up to `docs/roster_sticking_2026-05.md` (Чернуха, Рекшинская,
Монина on stable strong rosters).  Several new levers were tried after
user reports that intuitive strength (especially on hard questions, and
especially for veterans who **massively overperform** career
`actual − expected`) is not reflected in θ or in Model C rankings.

**Headline: nothing in this cycle fixes the motivating case without
either breaking holdout quality or failing on face validity.**  The
production model stays 1D with `eta_teammate=0.02`.  Model C improves
logloss but is **not promoted** — it makes floor-player ranks *less*
plausible when projected to a single scalar.

Related docs (detail per mechanism):

| Doc | Mechanism | Verdict |
|-----|-----------|---------|
| `docs/roster_sticking_2026-05.md` | `eta_teammate`, credit mix, adaptive η | only η_teammate accepted |
| `docs/temperature_credit_experiments_2026-06.md` | τ-scaled λ credit | REJECTED |
| `docs/difficulty_weights_2026-06.md` | loss reweight easy-miss / hard-take | REJECTED |
| `docs/2d_player_experiments_2026-06.md` | Model C γ_k | logloss win, **not promoted** |
| **this doc** | overperf θ floor, rank diagnostics | REJECTED / inconclusive |

Source artefacts kept:

- `results/exp_difficulty_weights_sweep.csv`
- `results/exp_modelc_rank_shift.csv`, `exp_modelc_rank_shift_summary.csv`
- `scripts/exp_difficulty_weights_sweep.py`, `scripts/exp_modelc_rank_shift.py`

Removed after documentation (one-off caches / reports):

- `scripts/diagnostic_overperf_floor.py`
- `scripts/modelc_pack_ranks.py`
- `results/modelc_full.npz`, `results/modelc_pack_ranks.csv`

---

## 1. Problem statement

**Noisy-OR identifiability on team data:** credit on a taken question
goes ∝ `λ_k = exp(a·θ_k − b − δ)`.  On a stable strong roster the
weakest member gets a tiny share, so θ can stay frozen low for years
even when the **team** (and sometimes the player vs implied team
strength) overperforms.

**Motivating players** (2026-05 / 2026-06 diagnostics):

| Player | id | 1D θ (prod) | career actual−exp | note |
|--------|-----|------------:|------------------:|------|
| Чернуха | 34909 | ≈ −0.39 | −70 | lowest-θ on Инк ~96 % of games |
| Рекшинская | 26818 | ≈ −0.15 | +26 | implied−θ gap ≈ 0 |
| Монина | 158668 | ≈ −0.19 | **+437** | largest paradox: huge overperformance, low θ |

User intuition: Monina is stronger than many peers; 1D and Model C
rankings on “medium/hard pack” projections did not match that.

---

## 2. Difficulty-weighted loss — REJECTED

See `docs/difficulty_weights_2026-06.md`.

Reweighting gradients (`diff_w_miss_power`, `diff_w_take_boost`) to
down-weight easy misses and up-weight hard takes **monotonically worsens**
honest holdout logloss (baseline **0.5018** → best solo-only trial
**0.5037**; team channel **0.647** if applied to all obs).  θ inflates
(Семушин 1.29 → 2.09) without predictive gain.

Code: `Config.diff_w_*` left at 0.0 in `rating/engine.py` /
`rating/model.py` (disabled dead-end).

---

## 3. Model C (γ_k) — logloss win, ranking loss

See `docs/2d_player_experiments_2026-06.md`.

| Config | holdout ll | AUC |
|--------|----------:|----:|
| 1D baseline (`freeze_log_a=True`) | 0.5018 | 0.8373 |
| Model C `eta_gamma=0.01` | **0.4978** | **0.8392** |

**−0.0040 logloss** — real, but small.  Helps offline/sync slices;
γ captures “hard-question specialism” (e.g. Семушин γ≈+0.32, Руссо
γ≈+0.45).

### Why not promoted

1. **θ scale changes** — θ becomes “ability at b=0”; veterans on strong
   rosters compress downward (Чернуха −0.39 → −1.11).
2. **1D projection `score(b*) = θ + γ·b*` is a poor global rank** for
   floor players.  At `b_mean≈+1.21` (medium pack):
   - Монина: θ≈−0.94, γ≈+0.17 → rank **~2350** (vs 1D ~1457)
   - Рекшинская: θ≈−0.88, γ≈+0.24 → rank **~1491** (above Monina despite
     +26 vs +437 career overperformance)
3. **Rank shift is large but not aligned with intuition** — Kendall τ≈0.81
   among games≥200 players (`results/exp_modelc_rank_shift_summary.csv`);
   ~9.6 % of pairs discordant; Bulat (32919) 1D ~415 → Model C ~243,
   but that is γ-driven, not floor-player recovery.
4. **“Top teammates” list was misleading** — ranking the 100 strongest
   among *all* ever-teammates surfaces 1–2 game elite guests (1D θ
   0.7–1.1), not loyal core (gt≥20).  Monina beat **4/101** by 1D θ and
   **0/101** by Model C mid projection in that biased list.

**Verdict:** keep `use_2d_players=False` in production; code remains
opt-in for research.

---

## 4. Per-tournament overperformance θ floor — REJECTED

**Idea (user rule):** if a roster beats its pre-play forecast by a
noticeable margin (`actual − expected ≥ margin`), no member should get
a negative net Δθ for that tournament (after gradient + `eta_teammate`).

**Implementation (reverted):** snapshot θ before SGD; after updates,
clamp `Δθ ≥ 0` per player when the roster’s sum of pre-update `p_take`
is beaten by `margin` (default 1.0 take).

### Results (honest holdout 10 %, seed 42)

| config | holdout ll | AUC | train loglik | θ clamps |
|--------|----------:|----:|-------------:|---------:|
| baseline | **0.5018** | 0.8373 | −12 073 276 | — |
| floor margin=1.0 | 0.5017 | 0.8373 | −12 071 249 | 75 052 |
| floor margin=0.5 | 0.5017 | 0.8373 | −12 070 359 | 128 846 |
| floor margin=2.0 | 0.5017 | 0.8373 | −12 072 004 | 23 676 |

Convergence: unchanged — extra epoch completes; no divergence.

### Motivating players (margin=1.0, full train)

| id | θ baseline | θ with floor | Δ |
|----|----------:|-------------:|--:|
| 158668 Монина | −0.192 | −0.194 | −0.003 |
| 26818 Рекшинская | −0.154 | −0.157 | −0.002 |
| 34909 Чернуха | −0.392 | −0.397 | −0.005 |

**Findings:**

1. **Holdout quality flat** (Δll ≈ −0.0001 — noise).  Not a predictive
   win worth the complexity.
2. **Career θ barely moves** — the floor only prevents local minuses on
   overperforming tournaments; it does **not** add positive updates.
   Thousands of other tournaments + butterfly effects from changed
   teammate θ dominate the final gauge.
3. **Same failure class as credit mix** — redistributing who gets hurt
   without changing the likelihood does not fix a **data** identifiability
   limit; it only patches symptoms on ~75k roster-tournament events.

**Verdict: REJECT.**  Code removed from `rating/engine.py` /
`rating/__main__.py` per failed-ablation convention.

---

## 5. Random vs uniform credit (conceptual, not implemented)

User question: is random bonus/penalty assignment different from
uniform?

- **Random one-hot credit** (all weight on one random teammate): same
  **expected** credit as uniform `1/n`, but different variance; with
  nonlinear log-loss and adaptive η, trajectories differ.
- **Independent random bonuses**: breaks conservation — not a valid
  reparameterisation.
- **Uniform credit blend** already ablated in 2026-05: Δlogloss = 0,
  Чернуха unchanged.

Conclusion: randomisation is a noisy variant of uniform redistribution;
prior experiments already show that path does not help floor players.

---

## 6. What still might work (untested)

1. **Per-player lapse** — encode “whiffs easy, delivers on hard” as
   extra noise on easy questions without deflating θ (see
   `docs/difficulty_weights_2026-06.md` §4b).
2. **Display / auxiliary metrics** — career `actual−exp`,
   `team_theta_implied`, loyalty-weighted core rankings (already on site
   for teams); do not force into θ.
3. **Richer data** — who buzzed / individual question attribution (not in
   rating DB).

---

## 7. Production status (unchanged)

- 1D noisy-OR, `freeze_log_a` default False (learned a on site),
  `eta_teammate=0.02`, lapse + recalibration, honest holdout **0.5018**.
- No new defaults from this cycle.
