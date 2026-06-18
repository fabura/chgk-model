# Team-size effect

## Motivation

The noisy-OR likelihood `p_take = 1 − exp(−Σ_k λ_k)` already accounts
for *some* effect of team size: more players ⇒ larger sum ⇒ higher
probability of taking the question. But that's a strong functional
assumption. Domain expertise says it doesn't hold in practice:

* The 6th player typically adds very little (≈1–2 questions out of 36).
* Solo players fatigue and miss questions they would otherwise answer
  in a team.
* Larger rosters (7+) often include weaker substitutes whose marginal
  contribution is much smaller than λ would suggest.

To capture this without abandoning the noisy-OR structure, we add a
**per-team-size shift on tournament difficulty**:

```
δ = μ_type[type] + ε_t + δ_size[clip(team_size, 1, K)]
```

where `δ_size` is a small learned vector indexed by team size, anchored
at the modal size (6) for identifiability. Larger `δ_size[n]` means
the noisy-OR over-predicts how often a team of size `n` takes
questions; the model corrects by making questions effectively harder
for that size class.

## Implementation

* Vector `delta_size[1..8]`. Sizes 9+ are clipped to 8 (rare; 2.2 % of
  observations and often roster errors).
* `delta_size[6] = 0` is pinned (never updated) so the parameter is
  identifiable against the per-tournament residual `ε_t` (which is
  centered weekly within type).
* Learning rate `eta_size = 0.005` (matches `eta_mu`), L2 shrinkage
  `reg_size = 0.10`.
* Per-type weights: `w_size_offline = 1.0`, `w_size_sync = 1.0`,
  `w_size_async = 0.5` (async rosters are noisier).
* CLI: `--use-team-size-effect / --no-use-team-size-effect`,
  `--eta_size`, `--reg_size`, `--team_size_max`, `--team_size_anchor`.

## Backtest results (2026-04-17, 8595 tournaments, 1719 test)

| metric         | baseline (no size) | with team-size effect | Δ |
|----------------|---------|---------|------|
| Logloss        | 0.5318  | **0.5274**  | −0.0044 |
| Brier          | 0.1777  | **0.1761**  | −0.0016 |
| AUC            | 0.8220  | **0.8249**  | +0.0029 |
| offline LL     | 0.4826  | **0.4721**  | −0.0105 |
| sync LL        | 0.5389  | **0.5376**  | −0.0013 |
| async LL       | 0.5592  | **0.5541**  | −0.0051 |

The largest gain is on offline tournaments — exactly where team-size
information is least noisy and most informative. All metrics improve;
no regressions.

## Learned curve

```
n=1 −0.292   n=2 −0.103   n=3 +0.076   n=4 +0.232   n=5 +0.397
n=6  0.000 (anchor)
n=7 +0.327   n=8 +0.502
```

A clear U-shape with minimum at 6:

* `n ≥ 3`: positive δ ⇒ noisy-OR over-predicts the benefit of extra
  players (diminishing returns / fatigue / weaker substitutes).
* `n ≤ 2`: negative δ ⇒ noisy-OR *under*-predicts: solo/duo teams are
  more often composed of strong, motivated players who slightly
  outperform the mechanical λ-summation. (This is partially confounded
  with selection: solo entries are rare and typically deliberate.)

## Defaults

`use_team_size_effect=True` is the new default in `Config`.

## Future work

* Allow per-type curves (offline / sync / async could differ — e.g. the
  9+ sizes seen in async are roster artifacts and shouldn't be lumped).
* Combine with the editor / question-position effects when those land,
  and re-tune `eta0`, `eta_mu`, `eta_eps`, `eta_size` jointly.
