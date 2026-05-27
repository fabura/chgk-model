"""Idempotent per-tournament upsert into the rating-dump PG tables.

Strategy
--------
Wrap everything for one tournament in a single transaction.  Per
table: ``DELETE WHERE tournament_id = %s`` then bulk INSERT the parsed
rows.  This works even though ``public.tournaments.id`` has no UNIQUE
constraint (so ``ON CONFLICT`` is unavailable on that table) and is
fully idempotent — a second run of the same data leaves PG identical.

Empty-payload rule
------------------
If the API returned ``results=[]`` (a tournament where teams haven't
posted scores yet), we must NOT delete from ``tournament_results`` /
``tournament_rosters``: that would erase whatever the last dump
contained.  The metadata row in ``tournaments`` and the editor list
are still safe to refresh — those only grow more accurate over time.

What we do NOT write
--------------------
- ``true_dls``: per-team granularity + FK on ``models``; the API only
  surfaces a single aggregate ``trueDL`` per tournament.  The dump
  remains the source of truth here; ``load_from_db`` already handles
  missing rows gracefully.
"""
from __future__ import annotations

from typing import Iterable

from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import execute_values

from rating_api.parse import (
    ParsedEditor,
    ParsedResult,
    ParsedRoster,
    ParsedTournament,
    ParsedTournamentBundle,
)


def _upsert_tournament_row(cur, t: ParsedTournament) -> None:
    """DELETE+INSERT one row in public.tournaments.

    public.tournaments has no UNIQUE on id (only a non-unique btree
    index), so ON CONFLICT isn't available; the delete-then-insert
    pattern is intentional.
    """
    cur.execute("DELETE FROM public.tournaments WHERE id = %s", (t.id,))
    cur.execute(
        """
        INSERT INTO public.tournaments (
            id, title, type, typeoft_id, questions_count,
            start_datetime, end_datetime, last_edited_at,
            maii_rating, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """,
        (
            t.id,
            t.title,
            t.type,
            t.typeoft_id,
            t.questions_count,
            t.start_datetime,
            t.end_datetime,
            t.last_edited_at,
            t.maii_rating,
        ),
    )


def _replace_results(cur, tournament_id: int, results: list[ParsedResult]) -> None:
    cur.execute(
        "DELETE FROM public.tournament_results WHERE tournament_id = %s",
        (tournament_id,),
    )
    if not results:
        return
    execute_values(
        cur,
        """
        INSERT INTO public.tournament_results (
            tournament_id, team_id, team_title, total, position,
            points_mask, updated_at
        ) VALUES %s
        """,
        [
            (
                r.tournament_id,
                r.team_id,
                r.team_title,
                r.total,
                r.position,
                r.points_mask,
            )
            for r in results
        ],
        template="(%s, %s, %s, %s, %s, %s, NOW())",
    )


def _replace_rosters(cur, tournament_id: int, rosters: list[ParsedRoster]) -> None:
    cur.execute(
        "DELETE FROM public.tournament_rosters WHERE tournament_id = %s",
        (tournament_id,),
    )
    if not rosters:
        return
    execute_values(
        cur,
        """
        INSERT INTO public.tournament_rosters (
            tournament_id, team_id, player_id, flag, is_captain, updated_at
        ) VALUES %s
        """,
        [
            (r.tournament_id, r.team_id, r.player_id, r.flag, r.is_captain)
            for r in rosters
        ],
        template="(%s, %s, %s, %s, %s, NOW())",
    )


def _replace_editors(cur, tournament_id: int, editors: list[ParsedEditor]) -> None:
    cur.execute(
        "DELETE FROM public.tournament_editors WHERE tournament_id = %s",
        (tournament_id,),
    )
    if not editors:
        return
    execute_values(
        cur,
        """
        INSERT INTO public.tournament_editors (
            tournament_id, player_id, created_at, updated_at
        ) VALUES %s
        """,
        [(e.tournament_id, e.player_id) for e in editors],
        template="(%s, %s, NOW(), NOW())",
    )


def upsert_bundle(conn: PGConnection, bundle: ParsedTournamentBundle) -> dict[str, int]:
    """Write one tournament + its dependent rows in a single transaction.

    Returns a small stats dict for logging.  Raises on DB error after
    rolling back; the caller is responsible for ``record_fetch`` with
    the error message.
    """
    tid = bundle.tournament.id
    n_results = len(bundle.results)
    n_rosters = len(bundle.rosters)
    n_editors = len(bundle.editors)
    skipped_results = n_results == 0  # see "Empty-payload rule" in module docstring

    try:
        with conn.cursor() as cur:
            _upsert_tournament_row(cur, bundle.tournament)
            if not skipped_results:
                _replace_results(cur, tid, bundle.results)
                _replace_rosters(cur, tid, bundle.rosters)
            _replace_editors(cur, tid, bundle.editors)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "n_results_written": 0 if skipped_results else n_results,
        "n_rosters_written": 0 if skipped_results else n_rosters,
        "n_editors_written": n_editors,
        "results_skipped_empty": int(skipped_results),
    }
