# SQLite — `questions.db` (chgk-embedings)

Внешний репозиторий. Путь по умолчанию в `build_db.py`:
`chgk-embedings/data/questions.db`.

## Таблица `questions`

Читает: `website/build/build_db.py::fetch_question_texts`.

| Колонка | Использование |
|---------|---------------|
| `id` | question_id |
| `number` | 1-based номер в паке → slot = number - 1 |
| `text`, `answer`, `zachet`, `nezachet`, `comment`, `source` | Контент |
| `pack_id`, `pack_title` | Метаданные пака |
| `tour_id`, `tour_number`, `tour_title` | Тур |
| `authors_json`, `editors_json` | JSON массивы `{name: ...}` |
| `tournaments_json` | JSON `[{id: tournament_id, ...}]` — привязка к турнирам rating DB |

## Логика привязки

1. Скан всех строк с непустым `tournaments_json`.
2. Для каждого `tournament_id` из JSON, если он в нашем наборе турниров:
   ключ `(tournament_id, slot)` → текст вопроса.
3. Первое совпадение побеждает (дедуп).
4. Pack editors: модальный `pack_id` per tournament → редакторы пака.

Покрытие ~35% слотов (зависит от качества `tournaments_json`).

## Куда попадает

| DuckDB | Источник |
|--------|----------|
| `questions.text`, `answer`, … | `questions` |
| `tournaments.pack_id`, `pack_title` | агрегат по pack |
| `pack_editors` | editors_json пака |

## Другие скрипты

`scripts/analyse_chr.py`, `scripts/sync_async_question_residuals.py`,
`scripts/analyse_player_weaknesses.py` — ad-hoc чтение той же БД.
