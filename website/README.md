# ChGK Model — сайт результатов

Веб-интерфейс для просмотра результатов модели ChGK: рейтинг игроков, история их выступлений, разбор турниров и вопросов.

## Стек

- **Backend**: FastAPI (Python)
- **БД**: DuckDB (read-only, собирается отдельным скриптом)
- **Шаблоны**: Jinja2 (server-side rendering — для SEO и простоты)
- **Графики**: Plotly (через CDN)
- **CSS**: Tailwind (через CDN)
- **Деплой**: один Docker-образ

## Структура

```
website/
  build/
    build_db.py         — собирает chgk.duckdb из cache + результатов модели + рейтинговой БД + базы вопросов
    build_full.log      — лог последней полной сборки (gitignore)
  app/
    main.py             — FastAPI-приложение
    db.py               — соединение с DuckDB (read-only)
    templates/          — Jinja2-шаблоны
      base.html
      top_players.html
      player.html
      tournament.html
      methodology.html
      search.html
    static/             — статика (если понадобится)
  data/
    chgk.duckdb         — собранная БД (~300 MB, не коммитится)
  Dockerfile
  requirements.txt
  README.md
```

## Источники данных

| Что | Откуда |
|-----|--------|
| Наблюдения (game, question, team, taken, roster) | `../data.npz` |
| θ игроков, b/a вопросов, история θ | `../results/seq.npz` |
| Имена игроков, метаданные турниров, ростеры команд, названия | рейтинговая БД (`public.players`, `public.tournaments`, `public.tournament_rosters`, `public.tournament_results`, `public.teams`) |
| Тексты вопросов, авторы, редакторы пака | `chgk-embedings/data/questions.db` |

## MVP-страницы

| Путь | Что показывает |
|------|----------------|
| `/` | Топ игроков по θ (фильтр по числу игр) |
| `/player/{id}` | Профиль игрока: график θ во времени, таблица турниров (взято / ожидание / Δ / место), топ сокомандников |
| `/tournament/{id}` | Страница турнира: scatter `b vs a` по вопросам, таблица команд (с топ-составом), таблица вопросов (с текстами при наличии) |
| `/methodology` | Объяснение модели и качества |
| `/search?q=…` | Простой поиск игроков и турниров |

## Сборка БД

```bash
# Полная сборка (~5–10 минут на 8.6K турниров)
cd /Users/fbr/Projects/personal/сhgk-model
source .venv/bin/activate
python website/build/build_db.py \
  --cache data.npz \
  --results results/seq.npz \
  --questions-db /Users/fbr/Projects/personal/chgk-embedings/data/questions.db \
  --out website/data/chgk.duckdb

# Быстрая sample-сборка (для разработки, ~15 секунд)
python website/build/build_db.py --limit-tournaments 50 --out website/data/chgk_sample.duckdb
```

Что на входе:
- `data.npz` — кеш наблюдений и индексных карт
- `results/seq.npz` — обученные параметры модели (θ, b, a, история θ)
- Доступ к рейтинговой БД через `DATABASE_URL` (по умолчанию `postgresql://postgres:password@127.0.0.1:5432/postgres`)
- `questions.db` из репозитория `chgk-embedings`

## Запуск (dev)

```bash
cd /Users/fbr/Projects/personal/сhgk-model
source .venv/bin/activate
pip install -r website/requirements.txt

# С полной БД
CHGK_DB_PATH=$(pwd)/website/data/chgk.duckdb \
  PYTHONPATH=website \
  uvicorn app.main:app --reload --port 8000

# С sample-БД
CHGK_DB_PATH=$(pwd)/website/data/chgk_sample.duckdb \
  PYTHONPATH=website \
  uvicorn app.main:app --reload --port 8000
```

Открыть `http://127.0.0.1:8000/`.

## Деплой через Docker

```bash
# Сначала соберите DB локально
python website/build/build_db.py --out website/data/chgk.duckdb

# Затем образ
docker build -t chgk-model -f website/Dockerfile .
docker run --rm -p 8000:8000 chgk-model
```

## Известные ограничения MVP

- **Колонка «Ожидание»** считается на финальных значениях θ (а не на тех, что были у игрока на момент турнира). Поэтому для очень старых матчей метрика смещена. В следующей версии: использовать θ_до_турнира из `player_history`.
- **Покрытие текстами вопросов**: ~35 % слотов имеют текст из `questions.db` (привязка по `tournaments_json`). Для остальных показываются только `b`, `a` и take rate.
- **Авторство** на уровне отдельного вопроса не показывается — слишком много неопределённости. Показываем только редакторов всего пака.
- **Эффекты модели** `δ_pos`, `δ_size`, `μ_type`, `ε_t` пока не учитываются в «ожидании» (только `b`, `a`, θ). Точность улучшится, если расширить `--results_npz` экспортом этих параметров.
