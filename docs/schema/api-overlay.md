# PostgreSQL — `api_overlay` schema

Служебная схема для зеркала API. **Не в `public`** — `restore.sh` в rating-db
дропает `public`, а `api_overlay` переживает refresh.

DDL: `rating_api/pg_state.py`.

## `api_overlay.fetch_state`

| Колонка | Описание |
|---------|----------|
| `tournament_id` PK | |
| `last_fetched_at` | Когда последний fetch |
| `api_last_edit_date` | `lastEditDate` с API |
| `http_status` | |
| `n_results`, `n_rosters` | Счётчики |
| `error_message` | При ошибке |

## Курсор синхронизации

Default since = `MAX(public.tournaments.last_edited_at)`.

Пагинация: `api.rating.chgk.info/tournaments?lastEditDate[strictly_after]=…`

## Что пишется куда

| Данные | Целевая таблица |
|--------|-----------------|
| Tournament metadata | `public.tournaments` |
| Results | `public.tournament_results` |
| Rosters | `public.tournament_rosters` |
| Editors | `public.tournament_editors` |
| Fetch audit | `api_overlay.fetch_state` |

**Не зеркалится**: `true_dls` (per-team, FK на models).

См. `rating_api/upsert.py`, `AGENTS.md` § API mirror.
