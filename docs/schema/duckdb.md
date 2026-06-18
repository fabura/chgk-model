# DuckDB — сайт (`website/data/chgk.duckdb`)

Read-only БД для FastAPI. Собирается `website/build/build_db.py`.
DDL — константа `DDL` в том же файле (~строка 839). **При изменении DDL обновляй этот документ.**

## Основные таблицы

### `players`

| Колонка | Описание |
|---------|----------|
| `player_id` PK | ID из rating DB |
| `last_name`, `first_name` | Имя |
| `theta` | Сырая сила модели |
| `theta_display` | θ с inactivity-shrink (для рейтинга) |
| `games` | Число наблюдений |
| `last_game_date` | Дата последней игры |

### `tournaments`

| Колонка | Описание |
|---------|----------|
| `tournament_id` PK | |
| `game_idx` | Индекс в хронологии (`idx_to_game_id`) |
| `title`, `type` | Метаданные |
| `start_date`, `end_date` | |
| `n_questions`, `n_teams` | |
| `pack_id`, `pack_title` | Из questions.db |

### `questions`

Канонические вопросы (после merge пар sync+async).

| Колонка | Описание |
|---------|----------|
| `canonical_idx` PK | |
| `primary_tournament_id`, `primary_q_in_tournament` | |
| `b`, `a` | Параметры модели |
| `n_obs`, `n_taken` | Агрегат по paired tournaments |
| `text`, `answer`, `zachet`, `comment`, `source` | Из questions.db |
| `authors_json`, `editors_json` | JSON |

### `question_aliases`

Per-tournament слот → canonical. **`n_obs`/`n_taken` здесь** — для take rate на странице турнира (не дублировать paired).

### `team_games`

Одна строка = команда на турнире.

| Колонка | Описание |
|---------|----------|
| `tournament_id`, `team_id`, `team_name` | |
| `n_players_active` | Размер ростера |
| `score_actual` | Взято вопросов |
| `expected_takes` | Ожидание модели (pre-tournament θ) |
| `team_theta_implied` | θ, implied из факта (график команды) |
| `place` | Место |
| `has_breakdown` | Есть ли points_mask |
| `points_mask` | Маска для H2H compare |

### `player_games`

Игрок × турнир × команда.

| Колонка | Описание |
|---------|----------|
| `theta_after` | θ после турнира |
| `n_takes_team`, `expected_takes_team` | На уровне команды |

### `player_history`

| Колонка | Описание |
|---------|----------|
| `player_id`, `tournament_id` | |
| `theta` | В **финальной** шкале (после recenter) |
| `rank_global`, `n_active` | Ранг среди qualified игроков на дату |

### `pack_editors`

`tournament_id`, `editor_name`

### `model_params`

Одна строка, JSON с `delta_size`, `delta_pos`, `lapse`, `recal` — для `/forecast/*`.

### `site_meta`

Одна строка: `data_as_of`, `model_built_at`, `map_scratch_meta`.

## Map-таблицы (`map_tables.py`)

| Таблица | Назначение |
|---------|------------|
| `map_venues` | Площадки с координатами |
| `map_venue_stats` | Агрегаты по площадкам |
| `map_player_regions` | Игрок × регион |
| `map_player_towns` | Игрок × город |

## Связи

```
players ──< player_games >── tournaments
players ──< player_history >── tournaments
tournaments ──< team_games
tournaments ──< question_aliases >── questions
tournaments ──< pack_editors
```

## Индексы

См. конец `DDL` в `build_db.py`: `idx_player_games_player`, `idx_team_games_tournament`, и др.

## Потребители

| Модуль | Таблицы |
|--------|---------|
| `website/app/main.py` | Все основные |
| `website/app/forecast.py` | `model_params`, `tournaments`, `questions` |
| `website/app/compare_h2h.py` | `team_games.points_mask`, `players` |
