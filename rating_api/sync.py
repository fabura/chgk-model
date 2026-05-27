"""Orchestrator: discover changed tournaments → fetch → parse → (later) upsert.

F.2 scope: dry-run only.  Walks discovery, fetches each tournament's
metadata + results, parses into PG-shaped structures, prints a summary,
records the outcome in api_overlay.fetch_state.  Does NOT touch
public.* — that's F.3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rating_api.client import RatingApiClient, RatingApiError
from rating_api.parse import (
    ParsedTournamentBundle,
    parse_editors,
    parse_results_blob,
    parse_tournament_blob,
)
from rating_api.pg_state import discovery_cursor, ensure_schema, open_conn, record_fetch


@dataclass
class SyncStats:
    discovered: int = 0
    fetched_ok: int = 0
    fetched_err: int = 0
    skipped: int = 0
    total_results: int = 0
    total_rosters: int = 0


def fetch_bundle(client: RatingApiClient, tournament_id: int) -> ParsedTournamentBundle:
    """Fetch metadata + results for one tournament, return parsed bundle."""
    meta_blob = client.get_tournament(tournament_id)
    tournament = parse_tournament_blob(meta_blob)

    rows = client.get_results(
        tournament_id, include_team_members=True, include_masks=True
    )
    results, rosters = parse_results_blob(tournament_id, rows)
    editors = parse_editors(tournament, meta_blob)

    return ParsedTournamentBundle(
        tournament=tournament,
        results=results,
        rosters=rosters,
        editors=editors,
    )


def discover_changed(
    client: RatingApiClient,
    *,
    since: str,
    limit: int | None = None,
) -> Iterable[dict]:
    """Yield up to `limit` tournament-summary blobs changed since `since`
    (YYYY-MM-DD)."""
    n = 0
    for blob in client.iter_tournaments_changed_since(since):
        yield blob
        n += 1
        if limit is not None and n >= limit:
            return


def run_sync(
    *,
    since: str | None = None,
    limit: int | None = None,
    dry_run: bool = True,
    database_url: str | None = None,
    client: RatingApiClient | None = None,
    verbose: bool = True,
) -> SyncStats:
    """End-to-end sync (dry-run for now).

    Parameters
    ----------
    since
        ISO date "YYYY-MM-DD" to use as lastEditDate cursor.  If None,
        derived from MAX(public.tournaments.last_edited_at).
    limit
        Stop after this many discovered tournaments (debug knob).
    dry_run
        If True, never writes to public.* (still writes to
        api_overlay.fetch_state for observability).  F.2 hard-codes
        True; F.3 will lift this.
    """
    if not dry_run:
        raise NotImplementedError(
            "Actual upsert into public.* is F.3.  Re-run with dry_run=True."
        )

    client = client or RatingApiClient()
    conn = open_conn(database_url)
    try:
        ensure_schema(conn)
        if since is None:
            since = discovery_cursor(conn)
            if since is None:
                raise RuntimeError(
                    "No lastEditDate cursor available — public.tournaments is "
                    "empty.  Run a dump restore first, or pass --since."
                )
            if verbose:
                print(f"[sync] cursor: lastEditDate >= {since} "
                      f"(from MAX(public.tournaments.last_edited_at))")

        stats = SyncStats()

        for summary in discover_changed(client, since=since, limit=limit):
            stats.discovered += 1
            tid = summary.get("id")
            if not isinstance(tid, int):
                stats.skipped += 1
                continue

            le = summary.get("lastEditDate")
            try:
                bundle = fetch_bundle(client, int(tid))
            except RatingApiError as e:
                stats.fetched_err += 1
                record_fetch(
                    conn,
                    tournament_id=int(tid),
                    api_last_edit_date=le,
                    http_status=e.status or 0,
                    n_results=0,
                    n_rosters=0,
                    error_message=str(e),
                )
                if verbose:
                    print(f"  [err] tid={tid}: {e}")
                continue

            stats.fetched_ok += 1
            stats.total_results += len(bundle.results)
            stats.total_rosters += len(bundle.rosters)
            record_fetch(
                conn,
                tournament_id=int(tid),
                api_last_edit_date=bundle.tournament.last_edited_at or le,
                http_status=200,
                n_results=len(bundle.results),
                n_rosters=len(bundle.rosters),
                error_message=None,
            )

            if verbose:
                t = bundle.tournament
                name = (t.title or "")[:48]
                print(
                    f"  [ok ] tid={tid:<6} type={t.type!r:<22} "
                    f"qcount={t.questions_count}  "
                    f"results={len(bundle.results):>4}  "
                    f"rosters={len(bundle.rosters):>5}  "
                    f"editors={len(bundle.editors):>2}  "
                    f"name={name!r}"
                )

        if verbose:
            print(
                f"[sync] done: discovered={stats.discovered} "
                f"ok={stats.fetched_ok} err={stats.fetched_err} "
                f"skipped={stats.skipped}  "
                f"results={stats.total_results} rosters={stats.total_rosters}"
            )
            print(
                "[sync] DRY-RUN: no rows written to public.*. "
                "api_overlay.fetch_state was updated."
            )

        return stats
    finally:
        conn.close()
