# Документация ChGK Model — навигация

**Начни здесь**, если ты агент или новый разработчик. Не ищи структуру проекта с нуля — она описана ниже.

## Быстрый старт

| Задача | Куда смотреть |
|--------|---------------|
| Где какой модуль / файл | [`repo-map.md`](repo-map.md) |
| Таблицы Postgres (рейтинговая БД) | [`schema/postgres.md`](schema/postgres.md) |
| Таблицы DuckDB (сайт) | [`schema/duckdb.md`](schema/duckdb.md) |
| Кеш `data.npz` и `results/seq.npz` | [`schema/cache.md`](schema/cache.md) |
| Тексты вопросов (`questions.db`) | [`schema/questions-db.md`](schema/questions-db.md) |
| Venue overlay, API overlay | [`schema/venue-overlay.md`](schema/venue-overlay.md), [`schema/api-overlay.md`](schema/api-overlay.md) |
| Схема данных — обзор и связи | [`schema/README.md`](schema/README.md) |
| Модель, гиперпараметры, пайплайн | [`../AGENTS.md`](../AGENTS.md) |
| Запуск, refresh, деплой | [`../README.md`](../README.md), [`../website/README.md`](../website/README.md) |
| Интерпретация θ | [`interpretation.md`](interpretation.md) |
| Эксперименты (индекс) | [`experiments_summary_ru.md`](experiments_summary_ru.md) |

## Поток данных (end-to-end)

```
rating-db Postgres (public.*)
        │
        ├─► python -m rating_api  ──► api_overlay.* (дельты с API)
        │
        ▼
  data.py::load_from_db()
        │
        ▼
    data.npz  (наблюдения + IndexMaps)
        │
        ▼
  python -m rating  (rating/engine.py)
        │
        ▼
  results/seq.npz  (θ, b, a, история, δ_size, δ_pos, lapse, recal)
        │
        ├─► questions.db (chgk-embedings) — тексты вопросов
        ├─► Postgres — имена, ростеры, места
        │
        ▼
  website/build/build_db.py
        │
        ▼
  website/data/chgk.duckdb  ──► FastAPI (website/app/)
```

## Правила поддержки документации

При изменении кода **обновляй соответствующий md в том же PR**:

| Что изменилось | Обновить |
|----------------|----------|
| DDL в `build_db.py` | `schema/duckdb.md` |
| SQL к `public.*` в `data.py` / `build_db.py` | `schema/postgres.md` |
| Формат `data.npz` / `seq.npz` | `schema/cache.md` |
| Новый пакет / значимый скрипт | `repo-map.md` |
| `Config` defaults / эксперимент | `experiments_summary_ru.md` + детальный doc |
| Новая таблица в overlay / API mirror | соответствующий `schema/*.md` |

Подробнее: `.cursor/rules/docs-maintenance.mdc`.

## Структура `docs/`

```
docs/
  INDEX.md              ← этот файл
  repo-map.md           ← карта модулей и файлов
  schema/               ← таблицы и связи
    README.md
    postgres.md
    duckdb.md
    cache.md
    questions-db.md
    venue-overlay.md
    api-overlay.md
  interpretation.md     ← θ и вероятности
  experiments_summary_ru.md
  *_experiments.md      ← детали экспериментов
  *_YYYY-MM.md          ← разовые исследования
```
