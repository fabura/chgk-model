# PostgreSQL — рейтинговая БД (`public.*`)

Источник: nightly dump из R2 → локальный Postgres (`rating-db` docker-compose).
Дельты: `python -m rating_api` пишет в те же таблицы `public.*`.

Подключение: `DATABASE_URL` (default `postgresql://postgres:password@127.0.0.1:5432/postgres`).

## Полная DDL (live snapshot)

Репозиторий `rating-db` — источник схемы. Для актуального списка колонок
из локального Postgres:

```bash
python scripts/introspect_postgres_schema.py --out docs/schema/postgres_live.md
```

Файл `postgres_live.md` генерируется вручную и коммитится при смене схемы
rating-db. CI проверяет только `docs/schema/postgres.md` (контракт проекта).

## Таблицы, которые читает модель

### `public.tournaments`

Метаданные турнира. Один турнир = одна «игра» с `questions_count` вопросов.

| Колонка | Использование в проекте |
|---------|-------------------------|
| `id` | PK; `tournament_id` / `game_id` |
| `title` | Сайт, детект пар sync+async |
| `type` | `очник` / `синхрон` / `асинхрон` → mode index 0/1/2 |
| `questions_count` | Число слотов вопросов |
| `start_datetime`, `end_datetime` | Хронология, фильтр дат |
| `last_edited_at` | Курсор `rating_api` |
| `typeoft_id`, `maii_rating` | Пишет API mirror |

**Фильтры при загрузке** (`data.py::load_from_db`):
- `questions_count >= min_questions` (default 10)
- наличие `points_mask` в results (если `only_with_question_data`)
- `start_datetime >= min_tournament_date` (default 2015-01-01)
- исключение season aggregates (`exclude_seasonal_aggregates`)

### `public.tournament_results`

Результаты команд. Ключевая колонка — **`points_mask`**: строка `'0'`/`'1'`, длина = число вопросов, `1` = взят.

| Колонка | Использование |
|---------|---------------|
| `tournament_id`, `team_id` | FK |
| `points_mask` | Бинарные исходы → `Sample.taken` |
| `position` | Место; `NULL` → DSQ / phantom filter |
| `total`, `team_title` | Сайт, API mirror |

**Фильтры**: async с `position` и нулевой маской отбрасываются как «зарегистрировались, не играли».

### `public.tournament_rosters`

Составы: одна строка = один игрок в команде на турнире.

| Колонка | Использование |
|---------|---------------|
| `tournament_id`, `team_id`, `player_id` | Состав → `player_indices` в Sample |
| `flag`, `is_captain` | API mirror only |

Игроки с `< min_games` (default 10) вырезаются из ростеров при загрузке.

### `public.tournament_editors`

Редакторы пака (player_id на турнир). Сайт: `pack_editors` в DuckDB.

### `public.players`

| Колонка | Использование |
|---------|---------------|
| `id` | `player_id` |
| `first_name`, `last_name` | Сайт, поиск |

### `public.teams`

| Колонка | Использование |
|---------|---------------|
| `id`, `title` | Названия команд на сайте |

## Вспомогательные таблицы

### `public.true_dls`

Сложность турнира (per-team в дампе; мы агрегируем `AVG(true_dl)` per tournament).
Опционально для `IndexMaps.tournament_dl`. API mirror **не пишет** — только dump.

### `public.ndcg`

Fallback сложности, если нет `true_dls`.

## Связи

```
tournaments (1) ──< tournament_results (team_id → teams)
              ──< tournament_rosters (team_id → teams, player_id → players)
              ──< tournament_editors (player_id → players)
              ──< true_dls / ndcg (optional)
```

## Что пишет `rating_api`

Per tournament, в транзакции: DELETE + INSERT в
`tournaments`, `tournament_results`, `tournament_rosters`, `tournament_editors`.

**Пустой results=[]**: metadata и editors обновляются; results/rosters **не трогаются**
(сохраняем данные из последнего dump).

См. `rating_api/upsert.py`, `docs/schema/api-overlay.md`.

## Где в коде

| Операция | Файл |
|----------|------|
| Основная загрузка | `data.py::load_from_db` |
| Пары sync+async | `data.py::detect_paired_tournaments` |
| Bake сайта | `website/build/build_db.py` |
| API upsert | `rating_api/upsert.py` |
