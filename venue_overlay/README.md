# Venue overlay

Sync tournament **venue** assignments from the public rating API, stored locally for analysis (not in the rating Postgres dump).

## Fetch

```bash
# Incremental (skips tournaments already in venue_fetch_state with HTTP 200)
python scripts/fetch_venue_overlay.py --resume

# One tournament
python scripts/fetch_venue_overlay.py --tournament-id 13606 --no-resume

# From data.npz game list instead of Postgres
python scripts/fetch_venue_overlay.py --source cache --cache data.npz --resume
```

Output: `data/venue_overlay.duckdb` with tables `venues`, `team_tournament_venue`,
`tournament_venues` (incl. `date_start`, `synch_request_id`), `synch_requests`
(`dateStart` from API), `venue_fetch_state`.

API: `GET https://api.rating.chgk.info/tournaments/{id}/results` (optional `?venue=` filter for debugging).

## Analyse

After `website/data/chgk.duckdb` is built:

```bash
python scripts/analyse_venue_effects.py
```

Writes `results/venue_effects_slices.csv` (mean actual − expected by venue bucket).

Backfill play dates for existing overlay:

```bash
python scripts/backfill_synch_request_dates.py --zurich-only --since 2025-01-01
python scripts/zurich_wednesday_stats.py   # → results/zurich_wednesday_telegram.md
```

## Refresh pipeline

`scripts/refresh_data.sh` runs the fetch after Postgres restore (`--skip-venue` to disable).
