"""Postgres-side state for the API mirror.

Schema choice — `api_overlay` (NOT `public`):
    rating-db's restore.sh drops the `public` schema before pg_restore,
    so anything we'd put in `public.*` would be wiped on every dump
    refresh.  `api_overlay` survives.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as PGConnection

DEFAULT_DATABASE_URL = "postgresql://postgres:password@127.0.0.1:5432/postgres"

DDL = """
CREATE SCHEMA IF NOT EXISTS api_overlay;

CREATE TABLE IF NOT EXISTS api_overlay.fetch_state (
    tournament_id      integer PRIMARY KEY,
    last_fetched_at    timestamptz NOT NULL,
    api_last_edit_date timestamptz,
    http_status        integer,
    n_results          integer,
    n_rosters          integer,
    error_message      text
);

CREATE INDEX IF NOT EXISTS fetch_state_last_edit_idx
    ON api_overlay.fetch_state (api_last_edit_date);
"""


def open_conn(database_url: str | None = None) -> PGConnection:
    url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    return psycopg2.connect(url)


@contextmanager
def open_cursor(database_url: str | None = None) -> Iterator[tuple[PGConnection, "psycopg2.extensions.cursor"]]:
    conn = open_conn(database_url)
    try:
        cur = conn.cursor()
        yield conn, cur
    finally:
        conn.close()


def ensure_schema(conn: PGConnection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def discovery_cursor(conn: PGConnection) -> str | None:
    """Pick the lastEditDate cursor for /tournaments?lastEditDate[strictly_after]=…

    We take MAX(public.tournaments.last_edited_at) since:
      - it is filled by the dump restore for every historical tournament,
      - it matches API's lastEditDate semantics (1-to-1 column),
      - it survives a wiped api_overlay (the dump always provides it).

    Returns a full ISO timestamp (``YYYY-MM-DDTHH:MM:SS``); the API
    iterator uses ``strictly_after`` so passing the exact MAX skips the
    record we already have and gives true no-op idempotency.

    Returns None if the table is empty (e.g. before the very first dump
    restore).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(last_edited_at) FROM public.tournaments")
        row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return row[0].isoformat(timespec="seconds")


def record_fetch(
    conn: PGConnection,
    *,
    tournament_id: int,
    api_last_edit_date: str | None,
    http_status: int,
    n_results: int,
    n_rosters: int,
    error_message: str | None,
) -> None:
    """Upsert into api_overlay.fetch_state.  last_fetched_at = now()."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO api_overlay.fetch_state (
                tournament_id, last_fetched_at, api_last_edit_date,
                http_status, n_results, n_rosters, error_message
            ) VALUES (%s, NOW(), %s, %s, %s, %s, %s)
            ON CONFLICT (tournament_id) DO UPDATE SET
                last_fetched_at    = EXCLUDED.last_fetched_at,
                api_last_edit_date = EXCLUDED.api_last_edit_date,
                http_status        = EXCLUDED.http_status,
                n_results          = EXCLUDED.n_results,
                n_rosters          = EXCLUDED.n_rosters,
                error_message      = EXCLUDED.error_message
            """,
            (
                int(tournament_id),
                api_last_edit_date,
                int(http_status),
                int(n_results),
                int(n_rosters),
                error_message,
            ),
        )
    conn.commit()
