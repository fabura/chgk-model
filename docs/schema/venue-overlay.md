# DuckDB — venue overlay (`data/venue_overlay.duckdb`)

Площадки синхронных турниров из API. **Не** в rating Postgres dump.

DDL: `venue_overlay/store.py`. Пишет: `scripts/fetch_venue_overlay.py`.

## Таблицы

### `venues`

Справочник площадок (`venue_id` PK): name, town, type, `is_online`.

### `team_tournament_venue`

PK `(tournament_id, team_id)` → `venue_id`, `synch_request_id`.

### `tournament_venues`

PK `(tournament_id, venue_id)`: `teams_played`, `is_mono`, `date_start`.

### `synch_requests`

`synch_request_id` PK: tournament, venue, `date_start`, status, `approximate_teams_count`.

### `venue_fetch_state`

Состояние fetch per tournament (HTTP status, ошибки).

## Связи

```
venues ──< team_tournament_venue >── tournaments (logical)
       ──< tournament_venues
synch_requests ──► venues, tournaments
```

## Использование

- `scripts/analyse_venue_effects.py` — join с `chgk.duckdb` expected vs actual
- `website/build/map_tables.py` — карта площадок (частично из overlay + geo)
- `scripts/zurich_wednesday_stats.py` — статистика по Zurich

См. также `venue_overlay/README.md`.
