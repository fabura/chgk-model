"""Incremental loader for api.rating.chgk.info.

Complements (does not replace) the rating-backup → Postgres dump path.
On every run:
  1. Pick a cursor (max(last_edited_at) from public.tournaments).
  2. Walk /tournaments?lastEditDate[after]=cursor&order[lastEditDate]=asc
     until the page is empty.
  3. For each changed/new id: fetch metadata + results+rosters+editors,
     parse into PG-shaped rows, upsert into public.* (delete-then-insert
     per tournament, in a single transaction).
  4. Record outcome in api_overlay.fetch_state.

F.2 (this version): dry-run only. F.3 will wire actual writes.
"""
from rating_api.client import RatingApiClient, RatingApiError  # noqa: F401
from rating_api.parse import (  # noqa: F401
    ParsedEditor,
    ParsedResult,
    ParsedRoster,
    ParsedTournament,
    parse_results_blob,
    parse_tournament_blob,
)
