# Карта репозитория

Где что лежит. **Источник правды для навигации агентов** — при добавлении пакета или значимого скрипта обновляй этот файл.

## Корень

| Путь | Назначение |
|------|------------|
| `data.py` | Загрузка из Postgres, `IndexMaps`, кеш `.npz`/`.pkl`, синтетика, детект пар sync+async |
| `requirements.txt` | Зависимости Python (модель + скрипты) |
| `data.npz` | Кеш наблюдений (не в git) |
| `results/seq.npz` | Результаты обучения (не в git) |
| `AGENTS.md` | Контекст для AI: модель, defaults, пайплайн |
| `CLAUDE.md` | 12 правил кодирования (шаблон) |
| `README.md` | Quick start, refresh, layout |

## `rating/` — sequential online rating

| Файл | Роль |
|------|------|
| `__main__.py` | CLI: `--mode db/cached/synthetic`, backtest, tune, export |
| `engine.py` | `Config`, `run_sequential()` — главный цикл SGD |
| `model.py` | Noisy-OR forward + градиенты |
| `players.py` | `PlayerState` — θ, adaptive η |
| `questions.py` | `QuestionState` — b, log_a, init из take rate |
| `decay.py` | Calendar decay между турнирами |
| `tournaments.py` | Кодирование типов (offline/sync/async) |
| `backtest.py` | Cell-holdout evaluation |
| `io.py` | `load_results_npz()` / `RatingResults` |
| `simulate.py` | Прогноз состава на паке (`/forecast/*`) |
| `batch_theta.py` | Батчевые θ-операции |
| `pack_calib.py` | Калибровка на уровне пака |
| `tune.py` | Grid / random search гиперпараметров |
| `h2h.py` | Head-to-head сравнение игроков |

Запуск: `python -m rating --mode cached --cache_file data.npz`

## `rating_api/` — зеркало api.rating.chgk.info

| Файл | Роль |
|------|------|
| `client.py` | HTTP-клиент, пагинация по `lastEditDate` |
| `parse.py` | JSON → dataclasses (как строки `public.*`) |
| `upsert.py` | DELETE+INSERT per tournament в Postgres |
| `pg_state.py` | Схема `api_overlay`, курсор синхронизации |
| `sync.py` | Оркестратор discover → fetch → parse → upsert |

Запуск: `python -m rating_api`

## `website/` — FastAPI + DuckDB

| Путь | Роль |
|------|------|
| `build/build_db.py` | Сборка `chgk.duckdb` из npz + PG + questions.db |
| `build/map_tables.py` | Таблицы `/map` (venues, player regions) |
| `app/main.py` | Маршруты HTML + API |
| `app/db.py` | Read-only DuckDB connection, hot reload |
| `app/forecast.py` | Логика страниц прогноза |
| `app/compare_h2h.py` | Head-to-head compare page |
| `app/templates/` | Jinja2 шаблоны |
| `app/static/` | JS (forecast, map), geojson |
| `data/chgk.duckdb` | БД сайта (~390 MB, gitignore) |
| `deploy/` | Docker, nginx, deploy.sh, refresh-db.sh |

Маршруты (`main.py`): `/`, `/tournaments`, `/teams`, `/player/{id}`,
`/team/{id}`, `/tournament/{id}`, `/map`, `/compare`, `/forecast/*`,
`/search`, `/methodology`, `/admin/reload-db`.

## `venue_overlay/` — площадки синхронов

| Файл | Роль |
|------|------|
| `store.py` | DDL + DuckDB `data/venue_overlay.duckdb` |
| `fetch.py` | Загрузка venue из API |
| `api.py` | HTTP helpers |

Скрипт: `scripts/fetch_venue_overlay.py`

## `scripts/` — аналитика и эксперименты

Не перечисляем все ~80 файлов. Группы:

| Префикс / паттерн | Назначение |
|-------------------|------------|
| `refresh_*.sh` | Nightly pipeline (postgres → train → duckdb) |
| `exp_*.py` | Sweep одного гиперпараметра / абляция |
| `diagnostic_*.py` | Разовые диагностики (roster sticking, tilt, …) |
| `analyse_*.py` | Анализ ошибок, venue, CHR, weaknesses |
| `compare_*.py` | Сравнение с baselines / chgk.fun |
| `count_*.py`, `lookup_*.py` | Утилиты |
| `player_report.py`, `eval_h2h_ranking.py` | Отчёты по игрокам / H2H |

Полный список: `ls scripts/`.

## `tests/`

| Файл | Что тестирует |
|------|---------------|
| `test_simulate.py` | `rating/simulate.py` |
| `test_rating_api_*.py` | API mirror (parse, upsert, sync, client) |
| `test_async_mode_effects.py` | Mode weights |
| `test_compare_h2h.py`, `test_h2h_eval.py` | H2H compare |

Запуск: `python -m pytest tests/`

## Внешние зависимости (не в этом репо)

| Ресурс | Где |
|--------|-----|
| Rating Postgres dump | `rating-db/` (sibling repo), R2 backup |
| Тексты вопросов | `chgk-embedings/data/questions.db` |
| Production VPS | `website/deploy/`, IP в `AGENTS.md` |

## Типичные команды

```bash
# Обучение
python -m rating --mode cached --cache_file data.npz --results_npz results/seq.npz

# Сборка сайта
python website/build/build_db.py --cache data.npz --results results/seq.npz \
  --questions-db /path/to/chgk-embedings/data/questions.db

# Полный refresh
./scripts/refresh_data.sh

# Тесты
python -m pytest tests/ -q
```
