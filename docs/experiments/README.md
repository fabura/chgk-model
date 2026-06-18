# Эксперименты и исследования

История абляций, принятых механизмов и отклонённых гипотез. **Не начинай поиск с корня `docs/`** — смотри сюда.

## С чего начать

| Задача | Файл |
|--------|------|
| Что в проде, что отвергли, хронология | [`experiments_summary_ru.md`](experiments_summary_ru.md) |
| Механизм в production (`δ_size`, lapse, init, …) | [`mechanisms/`](mechanisms/) |
| Цикл изменений за месяц | [`cycles/`](cycles/) |
| Пост-анализ турниров, venue, tilt (не абляции) | [`analysis/`](analysis/) |

## Структура

```
experiments/
  experiments_summary_ru.md   ← главный индекс (✅/❌/⚠️)
  mechanisms/                 ← долгоживущие механизмы модели
  cycles/
    2026-04/                  ← lean refactor, error structure, multi-epoch
    2026-05/                  ← leakage, lapse, recal, roster sticking
    2026-06/                  ← floor players, 2D γ, rejected fixes
  analysis/                   ← черновики постов, эмпирика без смены Config
```

## mechanisms/

| Doc | Тема |
|-----|------|
| [async_mode_experiments.md](mechanisms/async_mode_experiments.md) | offline / sync / async, веса `w_*` |
| [calendar_decay_experiments.md](mechanisms/calendar_decay_experiments.md) | calendar decay vs per-tournament |
| [team_size_experiments.md](mechanisms/team_size_experiments.md) | `δ_size` |
| [position_in_tour_experiments.md](mechanisms/position_in_tour_experiments.md) | `δ_pos` |
| [solo_channel_experiments.md](mechanisms/solo_channel_experiments.md) | solo update channel |
| [noisy_or_init_experiments.md](mechanisms/noisy_or_init_experiments.md) | noisy-OR-aware `b` init (round 1) |
| [theta_bar_init_experiments.md](mechanisms/theta_bar_init_experiments.md) | θ̄-aware init (round 2) |

## cycles/

### 2026-04

| Doc | Тема |
|-----|------|
| [error_structure_2026-04.md](cycles/2026-04/error_structure_2026-04.md) | Срезы ошибок, `n_extra_epochs`, `reg_size` |

### 2026-05

| Doc | Тема |
|-----|------|
| [leakage_2026-05.md](cycles/2026-05/leakage_2026-05.md) | Утечка time-split, cell-holdout |
| [calibration_2026-05.md](cycles/2026-05/calibration_2026-05.md) | Диагностика калибровки |
| [cleanup_2026-05.md](cycles/2026-05/cleanup_2026-05.md) | Удаление legacy Config |
| [lapse_rate_2026-05.md](cycles/2026-05/lapse_rate_2026-05.md) | Per-mode lapse π |
| [recalibration_2026-05.md](cycles/2026-05/recalibration_2026-05.md) | Logit-affine recal |
| [roster_sticking_2026-05.md](cycles/2026-05/roster_sticking_2026-05.md) | Floor player на сильном составе |

### 2026-06

| Doc | Тема |
|-----|------|
| [floor_player_experiments_2026-06.md](cycles/2026-06/floor_player_experiments_2026-06.md) | Сводный цикл floor players |
| [difficulty_weights_2026-06.md](cycles/2026-06/difficulty_weights_2026-06.md) | Loss reweight by difficulty |
| [temperature_credit_experiments_2026-06.md](cycles/2026-06/temperature_credit_experiments_2026-06.md) | Temperature-scaled credit |
| [2d_player_experiments_2026-06.md](cycles/2026-06/2d_player_experiments_2026-06.md) | Model C: θ + γ |

## analysis/

Наблюдательные отчёты и черновики — **не меняли `Config`**.

| Doc | Тема |
|-----|------|
| [tilt_resilience_2026-06.md](analysis/tilt_resilience_2026-06.md) | Tilt после промахов |
| [mono_venues_post.md](analysis/mono_venues_post.md) | Моноплощадки в синхронах |
| [chr2026_retrospective_draft.md](analysis/chr2026_retrospective_draft.md) | Черновик поста II ЧР-2026 |

## Как поддерживать

При новом эксперименте:

1. Добавь детальный doc в `mechanisms/`, `cycles/YYYY-MM/` или `analysis/`.
2. Обнови [`experiments_summary_ru.md`](experiments_summary_ru.md) в том же PR.
3. При смене `Config` defaults — также [`../model.md`](../model.md).
