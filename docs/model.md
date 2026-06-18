# Модель и обучение

Детальное описание probabilistic model, defaults и истории изменений.
Операционный контекст (сайт, refresh, API) — в [`../AGENTS.md`](../AGENTS.md).

Источник правды для текущих defaults: `rating/engine.py::Config`.

## Параметры

**ChGK** (Что? Где? Когда?) оценивает:

- **θ** (theta) — сила игрока
- **b** — сложность вопроса
- **a** — дискриминация вопроса (селективность)

из бинарных ответов команд (взято / не взято) и составов. Данные: `tournaments`, `tournament_results.points_mask`, `tournament_rosters`.

## Формула (noisy-OR)

```
z_k = -(b_i + δ) + a_i * θ_k
λ_k = exp(z_k)
S = Σ_k λ_k
p_take = 1 - exp(-S)
```

После lapse и recalibration (см. ниже): `p = (1−π)·p_noisy_or`, затем logit-affine per (mode × is_solo).

`δ = δ_size[clip(team_size, 1, K)] + δ_pos[q_index_in_tournament % tour_len]`:

- `δ_size` — per-team-size shift (anchor at 6) → [`team_size_experiments.md`](team_size_experiments.md)
- `δ_pos` — per-position-in-tour shift (anchor at 0) → [`position_in_tour_experiments.md`](position_in_tour_experiments.md)

Per-mode (`μ_type`) и per-tournament (`ε_t`) offsets **удалены** в 2026-04; различия форматов — через веса обновления (`w_online`, `w_sync`, …).

## Sequential online rating (`rating/`)

Хронологический SGD турнир за турниром.

```bash
python -m rating --mode cached --cache_file data.npz
python -m rating --mode cached --cache_file data.npz --backtest
python -m rating --mode cached --cache_file data.npz --results_npz results/seq.npz
```

### Ключевые defaults (production)

| Механизм | Default | Документ |
|----------|---------|----------|
| Mode handling `t6` | async θ слабее | [`async_mode_experiments.md`](async_mode_experiments.md) |
| Calendar decay | `rho_calendar=1.0` (off) | [`calendar_decay_experiments.md`](calendar_decay_experiments.md) |
| Team size δ | `use_team_size_effect=True`, anchor 6, max 12 | [`team_size_experiments.md`](team_size_experiments.md) |
| Position δ | `use_pos_effect=True`, `tour_len=12` | [`position_in_tour_experiments.md`](position_in_tour_experiments.md) |
| Solo channel | `use_solo_channel=True`, `w_solo=0.7` | [`solo_channel_experiments.md`](solo_channel_experiments.md) |
| Cold start | `cold_init_theta=-1.0`, `games_offset=0.25` | `scripts/exp_cold_start_grid.py` |
| Frozen a | `freeze_log_a=True` (a≡1) | `results/exp_holdout_ablations.csv` |
| Extra epoch | `n_extra_epochs=1` | `results/exp_multi_epoch_honest.csv` |
| Lapse rate | `use_lapse_rate=True` | [`lapse_rate_2026-05.md`](lapse_rate_2026-05.md) |
| Recalibration | `use_recalibration=True` | [`recalibration_2026-05.md`](recalibration_2026-05.md) |
| η₀, w_online | `0.22`, `1.0` | `results/exp_eta0_sweep_honest*.csv` |
| Yearly recenter | target −0.70, period 365d | см. ниже |
| Honest holdout | 10%, seed 42 | [`leakage_2026-05.md`](leakage_2026-05.md) |

**Backtest logloss (full DB, honest cell-holdout): 0.5004** (`--holdout 0.10 --holdout-seed 42`).
Legacy time-split 0.485 был leaky ~+5% overall; не сравнивать напрямую.

### Hyperparameters (`Config`)

Полный список полей в `rating/engine.py::Config`:

`eta0`, `rho_calendar`, `decay_period_days`, `cold_init_theta`, `games_offset`,
`w_online`, `w_online_questions`, `w_online_log_a`, `w_offline`, `w_sync`,
`eta_size`, `eta_pos`, `eta_teammate`, `reg_size`, `reg_pos`, `reg_theta`, `reg_b`,
`reg_log_a`, `team_size_max`, `team_size_anchor`, `w_size_offline/sync/async`,
`tour_len`, `pos_anchor`, `use_solo_channel`, `w_solo`, `w_solo_questions`,
`w_solo_log_a`, `w_size_solo`, `w_pos_solo`, `recenter_period_days`,
`recenter_target`, `recenter_min_games`, `recenter_active_days`,
`noisy_or_init`, `theta_bar_init`, `theta_bar_min_games`,
`n_extra_epochs`, `extra_test_fraction`, `freeze_log_a`,
`use_lapse_rate`, `lapse_init_*`, `eta_lapse`, `lapse_max`,
`use_recalibration`, `recal_*`, `holdout_obs_fraction`, `holdout_seed`.

Удалено в 2026-05: `rho`, `use_calendar_decay`, `cold_init_factor`,
`cold_init_use_team_mean` → [`cleanup_2026-05.md`](cleanup_2026-05.md).

Тюнинг: `python -m rating --mode cached --cache_file data.npz --tune`

### Yearly gauge re-centering

Каждые `recenter_period_days` (365): медиана θ «активных ветеранов»
(`games >= 200`, active within 365d) → `recenter_target` (−0.70).
Gauge transform: `θ ↑ Δ`, `b ↑ a·Δ` — predictions invariant.
Фильтр season aggregates: `exclude_seasonal_aggregates=True` в `data.py`.

### Paired tournaments

`canonical_q_idx` — sync+async пары делят b, a. Порядок: `start_datetime`.

## Файлы пакета `rating/`

| File | Role |
|------|------|
| `model.py` | Noisy-OR forward, gradients |
| `players.py` | `PlayerState` — θ, adaptive η |
| `questions.py` | `QuestionState` — b, log_a, init |
| `decay.py` | Calendar decay |
| `tournaments.py` | TYPE_OFFLINE/SYNC/ASYNC |
| `engine.py` | `Config`, `run_sequential()`, δ_size, δ_pos |
| `backtest.py` | Cell-holdout metrics |
| `io.py` | `load_results_npz()` |
| `simulate.py` | Runtime forecast (`/forecast/*`) |

## История изменений (кратко)

Хронология retune и ablation — в experiment docs. Основные вехи:

- **2026-04 lean**: убраны μ_type, ε_t; `eta_teammate` 0.005→0.02
- **2026-04 noisy-OR init**: `b_init = log(n) - log(-log(1-p))` → [`noisy_or_init_experiments.md`](noisy_or_init_experiments.md)
- **2026-04 θ̄-aware init**: + θ̄ mature players → [`theta_bar_init_experiments.md`](theta_bar_init_experiments.md)
- **2026-05**: lapse, recal, honest holdout, freeze log_a, cleanup legacy decay
- **2026-06**: floor-player experiments — rejected → [`floor_player_experiments_2026-06.md`](floor_player_experiments_2026-06.md)

Полный индекс: [`experiments_summary_ru.md`](experiments_summary_ru.md).

## Интерпретация θ

[`interpretation.md`](interpretation.md) — таблицы θ → probability.

```bash
python scripts/theta_to_prob.py 0.5 1.0
python scripts/theta_to_prob.py --table
```
