"""DuckDB schema and helpers for the venue overlay."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "venue_overlay.duckdb"

DDL = """
CREATE TABLE IF NOT EXISTS venues (
    venue_id INTEGER PRIMARY KEY,
    name TEXT,
    town_id INTEGER,
    town_name TEXT,
    venue_type_id INTEGER,
    venue_type_name TEXT,
    is_online BOOLEAN,
    fetched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS team_tournament_venue (
    tournament_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    venue_id INTEGER NOT NULL,
    synch_request_id INTEGER,
    fetched_at TIMESTAMP,
    PRIMARY KEY (tournament_id, team_id)
);

CREATE TABLE IF NOT EXISTS tournament_venues (
    tournament_id INTEGER NOT NULL,
    venue_id INTEGER NOT NULL,
    teams_played INTEGER NOT NULL,
    is_mono BOOLEAN NOT NULL,
    approx_teams_declared INTEGER,
    date_start TIMESTAMP,
    synch_request_id INTEGER,
    fetched_at TIMESTAMP,
    PRIMARY KEY (tournament_id, venue_id)
);

CREATE TABLE IF NOT EXISTS synch_requests (
    synch_request_id INTEGER PRIMARY KEY,
    tournament_id INTEGER,
    venue_id INTEGER,
    date_start TIMESTAMP,
    status TEXT,
    approximate_teams_count INTEGER,
    fetched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS venue_fetch_state (
    tournament_id INTEGER PRIMARY KEY,
    http_status INTEGER,
    n_results INTEGER,
    n_with_venue INTEGER,
    error_message TEXT,
    fetched_at TIMESTAMP
);
"""


def open_db(path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(DDL)
    for stmt in (
        "ALTER TABLE tournament_venues ADD COLUMN IF NOT EXISTS date_start TIMESTAMP",
        "ALTER TABLE tournament_venues ADD COLUMN IF NOT EXISTS synch_request_id INTEGER",
    ):
        try:
            con.execute(stmt)
        except duckdb.Error:
            pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def delete_tournament_rows(con: duckdb.DuckDBPyConnection, tournament_id: int) -> None:
    con.execute(
        "DELETE FROM team_tournament_venue WHERE tournament_id = ?",
        [tournament_id],
    )
    con.execute(
        "DELETE FROM tournament_venues WHERE tournament_id = ?",
        [tournament_id],
    )


def upsert_fetch_state(
    con: duckdb.DuckDBPyConnection,
    *,
    tournament_id: int,
    http_status: int,
    n_results: int,
    n_with_venue: int,
    error_message: str | None,
    fetched_at: datetime,
) -> None:
    con.execute(
        """
        INSERT INTO venue_fetch_state (
            tournament_id, http_status, n_results, n_with_venue,
            error_message, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (tournament_id) DO UPDATE SET
            http_status = excluded.http_status,
            n_results = excluded.n_results,
            n_with_venue = excluded.n_with_venue,
            error_message = excluded.error_message,
            fetched_at = excluded.fetched_at
        """,
        [
            tournament_id,
            http_status,
            n_results,
            n_with_venue,
            error_message,
            fetched_at,
        ],
    )
