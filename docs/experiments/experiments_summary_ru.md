# Сводка экспериментов ChGK Model

Справочный индекс: что пробовали, когда (по датам в именах файлов и содержимому доков), что вошло в продакшен, что отвергли. Подробности — в связанных документах.

**Метрики.** До мая 2026 большинство абляций считалось на **20 % time-split** (последние турниры по дате) — метрика **оптимистична** из‑за утечки в инициализацию `b` ([leakage_2026-05.md](cycles/2026-05/leakage_2026-05.md)). С мая 2026 честный дефолт бэктеста: **cell-holdout 10 %**, seed 42 (`--holdout 0.10`). Текущий продакшен (после цикла 2026-05): honest logloss **≈ 0.5004**, AUC **≈ 0.838**.

**Обозначения статуса:** ✅ в продакшене · ⚠️ частично / диагностика · ❌ отвергнуто · 🔬 код есть, не в проде · ❓ неоднозначно / не перепроверено на honest holdout

## Как поддерживать этот документ

**Когда обновлять** — после любого эксперимента, который меняет дефолты `Config`, принимается или отвергается, или получает новый отчёт в `docs/experiments/`.

**Что добавить** — дату (месяц), гипотезу в одну строку, ключевые метрики (logloss, AUC, срезы), статус (✅ / ❌ / ⚠️ / 🔬 / ❓), ссылку на подробный doc.

**Куда** — в «Хронологию» (подраздел нужного месяца); при необходимости — пункт в «Тематический указатель»; при смене продакшен-дефолтов — блок «Текущий продакшен».

**Метрики** — всегда помечай: leaky (20 % time-split) или honest cell-holdout 10 % (см. блок **Метрики** выше). Не сравнивай числа между режимами.

Подробные выводы и таблицы — в `docs/experiments/mechanisms/`, `docs/experiments/cycles/`, `docs/experiments/analysis/`; **этот файл — только индекс**.

**Для агентов:** обновляй сводку в том же PR/commit, что experiment-doc, изменение `Config` или смену статуса (promote/reject).

---

## Хронология

### Ранний цикл (до апреля 2026)

#### Режимы турниров: offline / sync / async — [async_mode_experiments.md](mechanisms/async_mode_experiments.md)

| Идея | Гипотеза | Результат | Статус |
|------|----------|-----------|--------|
| Не использовать `true_dl` / NDCG как вход | Псевдо-внешняя сложность из тех же ответов | Принято как принцип | ✅ |
| `δ_t = μ_type + ε_t` вместо одного сдвига | Асинхрон — другой режим наблюдения, не только меньший вес | Logloss 0.78→0.61, но AUC упал | ⚠️ позже убрано (см. апрель) |
| Первые агрессивные веса async | — | Слишком сильное поглощение структуры | ❌ |
| Профиль **t6** (`eta_mu`, `reg_*`, `w_async_*`, …) | Компромисс logloss / AUC | t6: logloss 0.602, AUC 0.814 vs new_default 0.612 / 0.803 | ✅ (на момент внедрения) |
| Снижение `reg_eps`, взвешенное центрирование `ε_t` для элитных турниров (2026-04) | Крупные офлайн-пакеты недооцениваются | Ухудшение на ЧР; центрирование доминирует | ❌ |

#### Календарный decay — [calendar_decay_experiments.md](mechanisms/calendar_decay_experiments.md)

| Идея | Результат (vs t6 baseline) | Статус |
|------|---------------------------|--------|
| Глобальный `ρ` за турнир | θ к концу обучения ×0.014 — история стирается | ❌ (legacy) |
| Per-player calendar decay | `rho_calendar=1.0`: logloss **0.532** (−0.070), AUC 0.822 | ✅ механизм; **ρ=1.0 = без decay** в проде |
| `reg_theta` 0.001–0.01 | Хуже baseline | ❌ |
| `cold_init_factor=0.5` | Слегка хуже | ❌ |

#### Холодный старт (без отдельного doc; см. [cleanup_2026-05.md](cycles/2026-05/cleanup_2026-05.md), `scripts/exp_cold_start_grid.py`)

| Идея | Результат | Статус |
|------|-----------|--------|
| Наследование среднего θ команды | Многолетний дрейф медианы θ | ❌ убрано 2026-05 |
| Фиксированный prior `cold_init_theta=-1.0` + rookie boost `games_offset=0.25` | Сетка 12 ячеек в `exp_cold_start_grid*` | ✅ |

---

### Апрель 2026

#### «Lean refactor»: убрать `μ_type` и `ε_t` — AGENTS.md, [error_structure_2026-04.md](cycles/2026-04/error_structure_2026-04.md)

| Изменение | Метрики (20 % time-split, leaky) | Статус |
|-----------|----------------------------------|--------|
| Удалены per-mode `μ_type` и per-tournament `ε_t` (~8746 параметров) | Logloss 0.5365→**0.5309** (−0.0056), AUC 0.8065→0.8115 | ✅ |
| Добавлен `eta_teammate=0.005` (усадка к среднему состава) | См. [roster_sticking_2026-05.md](cycles/2026-05/roster_sticking_2026-05.md) | ✅ (позже 0.02) |
| Ретюн после cleanup (`retune_2026-04*.csv`, 38 trials) | `eta0=0.05`, `w_sync=0.7`, `w_online_questions=0.30` | ✅ на тот момент |

Другие фильтры/механизмы из AGENTS.md (без отдельного experiment-doc): исключение сезонных агрегатов (`exclude_seasonal_aggregates`); годовое **re-centering** медианы θ ветеранов (`recenter_target≈−0.70`) — инвариантно к предсказаниям, фиксирует дрейф шкалы.

#### Эффект размера команды `δ_size` — [team_size_experiments.md](mechanisms/team_size_experiments.md) (2026-04-17)

| Метрика | Без эффекта | С эффектом | Δ |
|---------|------------|------------|---|
| Logloss | 0.5318 | **0.5274** | −0.0044 |
| AUC | 0.8220 | **0.8249** | +0.0029 |

Якорь на размере 6. ✅ `use_team_size_effect=True`.

#### Позиция в туре `δ_pos` — [position_in_tour_experiments.md](mechanisms/position_in_tour_experiments.md)

| Метрика | + team_size | + pos (anchor 0) | Δ vs исходный baseline |
|---------|-------------|------------------|------------------------|
| Logloss | 0.5274 | **0.5225** | −0.0093 |
| AUC | 0.8249 | **0.8255** | +0.0035 |

✅ `use_pos_effect=True`, `tour_len=12`, `pos_anchor=0`.

#### Solo-канал — [solo_channel_experiments.md](mechanisms/solo_channel_experiments.md)

| `w_solo` | Logloss | AUC | Заметки |
|----------|---------|-----|---------|
| legacy (1.0) | 0.5276 | 0.8151 | Солоисты раздувают θ |
| **0.3** | **0.5268** | 0.8159 | Выбран: баланс метрик и дефляции солоистов |
| 0.0 (выкинуть соло) | 0.5272 | — | Слишком грубо |

✅ `use_solo_channel=True`, `w_solo=0.3` (позже 0.7 после lapse, см. май).

#### Noisy-OR-aware init вопросов (Round 1) — [noisy_or_init_experiments.md](mechanisms/noisy_or_init_experiments.md)

| Конфиг | Logloss | AUC | offline |
|--------|---------|-----|---------|
| legacy init | 0.5270 | 0.8158 | 0.5075 |
| **combo_full** (init + ретюн) | **0.5182** | **0.8333** | **0.4791** |

Формула: `b = log(n) − log(−log(1−p))`. Ретюн: `eta0=0.04`, `w_sync=0.5`, `w_online_questions=0.15`, `eta_size=eta_pos=0.001`. ✅ (часть прироста на leaky split — см. май).

#### θ̄-aware init (Round 2) — [theta_bar_init_experiments.md](mechanisms/theta_bar_init_experiments.md)

| Конфиг | Logloss | AUC |
|--------|---------|-----|
| только noisy-OR init | 0.5182 | 0.8333 |
| + θ̄ без ретюна | 0.4950 | 0.8395 |
| **+ ретюн `eta0=0.15`** | **0.4877** | **0.8455** |

Диагностика Высшей лиги Москвы: mean(actual−expected) с −5.5 до ≈0. ✅ `theta_bar_init=True`, `theta_bar_min_games=3`.

Отвергнуто в том же цикле: `b_pack_shrinkage`, `pack_prior_w` (упоминание в [theta_bar_init_experiments.md](mechanisms/theta_bar_init_experiments.md) / [roster_sticking_2026-05.md](cycles/2026-05/roster_sticking_2026-05.md)).

#### Структура ошибок и multi-epoch — [error_structure_2026-04.md](cycles/2026-04/error_structure_2026-04.md)

Ключевые находки (20 % tail, leaky logloss ≈0.488):

- Async: logloss 0.517 vs offline 0.448; +1 п.п. недопредсказание.
- Маленькие команды (1–2): остаток +0.04 / +0.02.
- `n_extra_epochs=1`: logloss **0.4877→0.4812** (−0.0065); K≥2 переобучение.

| Идея | Результат | Статус |
|------|-----------|--------|
| **`n_extra_epochs=1`** | Sweet spot | ✅ |
| Curriculum filter (пропуск редких canonical) | Монотонно хуже | ❌ |
| Per-mode `δ_size[mode][size]` | <0.0001 logloss | ❌ |
| **`reg_size: 0.10→0.0`** | Logloss 0.4877→**0.4872** (−0.0005), ~96 % oracle | ✅ |

---

### Май 2026

#### Утечка в бэктесте и cell-holdout — [leakage_2026-05.md](cycles/2026-05/leakage_2026-05.md), [calibration_2026-05.md](cycles/2026-05/calibration_2026-05.md)

| Режим оценки | Logloss | Δ |
|--------------|---------|---|
| Leaky time-split | 0.4850 | — |
| **Honest cell-holdout 10 %** | **0.5083** | **+0.023 (+4.8 %)** |
| offline slice | 0.4501 → 0.5213 | **+15.8 %** |

✅ `holdout_obs_fraction=0.10`; `--backtest` по умолчанию с holdout (legacy: `--holdout 0.0`).

Перепроверка абляций: `results/exp_holdout_ablations.csv` — приросты noisy-OR / θ̄-init частично «улучшение утечки».

#### Cleanup конфига — [cleanup_2026-05.md](cycles/2026-05/cleanup_2026-05.md)

| Удалено / изменено | Причина | Статус |
|--------------------|---------|--------|
| `rho`, `use_calendar_decay`, per-tournament decay | Заменён calendar path; мёртвый код | ✅ удалено |
| `cold_init_use_team_mean`, `cold_init_factor` | Фиксированный cold-start лучше | ✅ удалено |
| **`freeze_log_a=True`** | Honest logloss идентичен; async −0.0016 | ✅ |
| **`team_size_max: 8→12`** | Калибровка размеров 10+ | ✅ |

#### Калибровка → lapse rate — [calibration_2026-05.md](cycles/2026-05/calibration_2026-05.md), [lapse_rate_2026-05.md](cycles/2026-05/lapse_rate_2026-05.md)

Высокие `p`: solo +9.5 п.п., async +3.9 п.п. bias.

| | До lapse | После lapse |
|---|---------|-------------|
| Overall logloss | 0.5061 | **0.5007** (−**0.0054**) |
| Solo high-p bias | +9.5 п.п. | +1.5 п.п. |

`p = (1−π_{mode,solo})·p_noisy_or`, 6 параметров π. ✅ `use_lapse_rate=True`.

Побочный ретюн: **`w_online` 0.5→1.0**, **`eta0`→0.22**, **`w_solo` 0.3→0.7** (AGENTS.md, `exp_w_online_sweep_honest_high.csv`).

#### Logit-affine рекалибровка — [recalibration_2026-05.md](cycles/2026-05/recalibration_2026-05.md)

| | Lapse only | + Recal |
|---|-----------|---------|
| Logloss | 0.5007 | **0.5004** (−0.0003) |

12 параметров (α, β) по (mode × solo). ✅ `use_recalibration=True`.

#### «Залипание» слабого игрока в сильном составе — [roster_sticking_2026-05.md](cycles/2026-05/roster_sticking_2026-05.md)

Кейсы: Чернуха, Рекшинская, Монина.

| Гипотеза | Honest logloss | Вердикт |
|----------|----------------|---------|
| **`eta_teammate` 0.005→0.02** | 0.50226→**0.50178** | ✅ |
| Residual-aware adaptive η | Сигнал ~80 % team-context | ❌ |
| `credit_uniform_mix` | Δlogloss = 0 | ❌ |
| Сдвиг `w_online` ради Рекшинской | Хуже async | ❌ без изменений |

---

### Июнь 2026

Сводный цикл: [floor_player_experiments_2026-06.md](cycles/2026-06/floor_player_experiments_2026-06.md). **Продакшен не менялся** (кроме уже принятого `eta_teammate=0.02`).

#### Взвешивание по сложности — [difficulty_weights_2026-06.md](cycles/2026-06/difficulty_weights_2026-06.md)

| Конфиг | Honest logloss |
|--------|----------------|
| baseline | **0.5018** |
| best solo-only reweight | 0.5037 |
| на всех obs | **0.6472** (коллапс) |

❌ `diff_w_*` остаются в Config с дефолтом 0.

#### Temperature-scaled credit — [temperature_credit_experiments_2026-06.md](cycles/2026-06/temperature_credit_experiments_2026-06.md)

| τ | Logloss |
|---|---------|
| 1.0 | **0.5018** |
| 1.5 | 0.5431 |
| 2.0 | 0.6753 |

❌ симметричный и асимметричный τ; ❌ `eta_teammate` >0.03 без выигрыша.

#### Model C: 2D игрок (θ + γ) — [2d_player_experiments_2026-06.md](cycles/2026-06/2d_player_experiments_2026-06.md)

| Конфиг | Honest logloss | AUC |
|--------|----------------|-----|
| 1D baseline | 0.5018 | 0.8373 |
| **2d `eta_gamma=0.01`** | **0.4978** | **0.8392** |

🔬 **−0.0040 logloss**, но ранги «floor players» на проекции `θ+γ·b` хуже интуиции (Монина ~2350). `use_2d_players=False` в проде.

#### Пол θ при перевыполнении команды — [floor_player_experiments_2026-06.md](cycles/2026-06/floor_player_experiments_2026-06.md) §4

| margin | logloss | Δθ мотивирующих игроков |
|--------|---------|-------------------------|
| baseline | 0.5018 | — |
| floor | 0.5017 | ≈0 |

❌ код убран.

#### Диагностики (не меняют модель)

| Работа | Док | Статус |
|--------|-----|--------|
| Tilt после «избегаемых» промахов | [tilt_resilience_2026-06.md](analysis/tilt_resilience_2026-06.md) | ⚠️ observational; z≈−19.6 |
| Моноплощадки в синхронах | [mono_venues_post.md](analysis/mono_venues_post.md) | ⚠️ эмпирика API, lift ≈0 у опытных |
| Ретроспектива II ЧР-2026 | [chr2026_retrospective_draft.md](analysis/chr2026_retrospective_draft.md) | ⚠️ пост-анализ |

---

## Тематический указатель

### Инициализация и утечки

- Legacy `b = −log(p)` → noisy-OR → θ̄-aware: [noisy_or_init_experiments.md](mechanisms/noisy_or_init_experiments.md), [theta_bar_init_experiments.md](mechanisms/theta_bar_init_experiments.md)
- Утечка time-split и cell-holdout: [leakage_2026-05.md](cycles/2026-05/leakage_2026-05.md)
- Экстремальные `r` при init, clamp `b≈9.6`: [error_structure_2026-04.md](cycles/2026-04/error_structure_2026-04.md) §1.7 — **take-rate-free init не внедрён**

### Структурные сдвиги сложности

- Размер команды, позиция в туре: [team_size_experiments.md](mechanisms/team_size_experiments.md), [position_in_tour_experiments.md](mechanisms/position_in_tour_experiments.md)
- Режимы (исторически `μ_type+ε_t`, снято в lean refactor): [async_mode_experiments.md](mechanisms/async_mode_experiments.md)

### Игрок и состав

- Cold-start, solo-канал, teammate shrinkage: [solo_channel_experiments.md](mechanisms/solo_channel_experiments.md), [roster_sticking_2026-05.md](cycles/2026-05/roster_sticking_2026-05.md), [cleanup_2026-05.md](cycles/2026-05/cleanup_2026-05.md)
- Атрибуция кредита (uniform mix, temperature, difficulty weights): отвергнуто — см. май–июнь docs выше
- 2D игрок γ: [2d_player_experiments_2026-06.md](cycles/2026-06/2d_player_experiments_2026-06.md)

### Калибровка вероятностей

- Диагностика S-кривой: [calibration_2026-05.md](cycles/2026-05/calibration_2026-05.md), [error_structure_2026-04.md](cycles/2026-04/error_structure_2026-04.md) §1.1
- Lapse + recalibration: [lapse_rate_2026-05.md](cycles/2026-05/lapse_rate_2026-05.md), [recalibration_2026-05.md](cycles/2026-05/recalibration_2026-05.md)
- Per-player lapse — **не тестировался** ([difficulty_weights_2026-06.md](cycles/2026-06/difficulty_weights_2026-06.md) §4b)

### Обучение и гиперпараметры

- Multi-epoch, `reg_size`, freeze `a`: [error_structure_2026-04.md](cycles/2026-04/error_structure_2026-04.md), [cleanup_2026-05.md](cycles/2026-05/cleanup_2026-05.md)
- Крупные ретюны: noisy-OR (`exp_noisy_or_init_retune.csv`), θ̄ (`exp_theta_bar_retune.csv`), post-cleanup (`retune_2026-04*.csv`) — CSV в `results/`, в репозитории могут отсутствовать

### Интерпретация θ

- [interpretation.md](../interpretation.md) — не эксперимент, справка по шкале

---

## Текущий продакшен (сводка дефолтов `Config`)

По AGENTS.md и докам 2026-05/06 (honest logloss **0.5004**):

| Компонент | Значение / флаг |
|-----------|-----------------|
| Ядро | noisy-OR, `freeze_log_a=True`, `n_extra_epochs=1` |
| Init | `noisy_or_init`, `theta_bar_init` |
| Сдвиги | `δ_size` (max 12), `δ_pos` (tour 12) |
| Cold-start | `cold_init_theta=-1.0`, `games_offset=0.25` |
| Каналы | solo channel `w_solo=0.7`; `w_online=1.0`; `eta0=0.22` |
| Калибровка | lapse + recalibration |
| Состав | `eta_teammate=0.02` |
| Decay | `rho_calendar=1.0` (без календарного decay в θ) |
| Оценка | `--holdout 0.10 --holdout-seed 42` |
| **Не в проде** | `use_2d_players`, `diff_w_*`, temperature credit, overperf floor |

Display-only: `theta_display` с inactivity shrink — [AGENTS.md](../AGENTS.md), не влияет на обучение.

---

## Пробелы в документации

- **Cold-start grid** (`exp_cold_start_grid.py`): нет отдельного doc; только [cleanup_2026-05.md](cycles/2026-05/cleanup_2026-05.md) и AGENTS.md.
- **Ретюн 2026-04** (`retune_2026-04*.csv`): упоминание в AGENTS, отдельного отчёта нет.
- **Re-centering / seasonal filter**: описаны в AGENTS.md, без experiment-doc.
- **`results/*.csv`**: многие артефакты перечислены в доках, но **не закоммичены** в репозиторий (поиск не нашёл CSV).
- **Ранние async-эксперименты**: даты запусков не зафиксированы, только относительные сравнения.
- **Per-player lapse**, **take-rate-free b init**, **Model C в проде** — обсуждены как next steps, не завершены.
- **mono_venues**, **tilt**, **ЧР-2026** — прикладной анализ, не абляции модели.

---

*Последнее обновление сводки: по состоянию docs/ на июнь 2026.*
