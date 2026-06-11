"""Fetch venue overlay from rating API into DuckDB."""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import duckdb

from venue_overlay.api import (
    DEFAULT_API_BASE,
    RatingApiError,
    SynchRequestDetail,
    fetch_synch_requests_for_tournament,
    fetch_tournament_results,
)
from venue_overlay.store import (
    DEFAULT_DB_PATH,
    delete_tournament_rows,
    ensure_schema,
    open_db,
    upsert_fetch_state,
    utc_now,
)


@dataclass(frozen=True)
class VenueRow:
    venue_id: int
    name: str
    town_id: int | None
    town_name: str | None
    venue_type_id: int | None
    venue_type_name: str | None
    is_online: bool


@dataclass(frozen=True)
class TeamVenueRow:
    tournament_id: int
    team_id: int
    venue_id: int
    synch_request_id: int | None


def _is_online(town_name: str | None, venue_name: str | None) -> bool:
    tn = (town_name or "").strip().lower()
    vn = (venue_name or "").strip().lower()
    return tn == "онлайн" or vn.startswith("онлайн")


def _parse_venue_blob(venue: dict[str, Any]) -> VenueRow | None:
    vid = venue.get("id")
    if vid is None:
        return None
    town = venue.get("town") or {}
    vtype = venue.get("type") or {}
    name = (venue.get("name") or "").strip()
    town_name = (town.get("name") or "").strip() or None
    return VenueRow(
        venue_id=int(vid),
        name=name,
        town_id=int(town["id"]) if town.get("id") is not None else None,
        town_name=town_name,
        venue_type_id=int(vtype["id"]) if vtype.get("id") is not None else None,
        venue_type_name=(vtype.get("name") or "").strip() or None,
        is_online=_is_online(town_name, name),
    )


def parse_results_rows(
    tournament_id: int,
    rows: list[dict[str, Any]],
) -> tuple[list[TeamVenueRow], dict[int, VenueRow], int]:
    """Return team-venue rows, venue dimension rows, count of rows without venue."""
    team_rows: list[TeamVenueRow] = []
    venues: dict[int, VenueRow] = {}
    missing = 0

    for row in rows:
        team = row.get("team") or {}
        team_id = team.get("id")
        if team_id is None:
            missing += 1
            continue
        sr = row.get("synchRequest")
        if not isinstance(sr, dict):
            missing += 1
            continue
        venue = sr.get("venue")
        if not isinstance(venue, dict):
            missing += 1
            continue
        parsed = _parse_venue_blob(venue)
        if parsed is None:
            missing += 1
            continue
        venues[parsed.venue_id] = parsed
        team_rows.append(
            TeamVenueRow(
                tournament_id=int(tournament_id),
                team_id=int(team_id),
                venue_id=parsed.venue_id,
                synch_request_id=int(sr["id"]) if sr.get("id") is not None else None,
            )
        )
    return team_rows, venues, missing


def aggregate_tournament_venues(
    team_rows: list[TeamVenueRow],
    synch_by_venue: dict[int, SynchRequestDetail],
    *,
    fetched_at: datetime,
) -> list[tuple[int, int, int, bool, int | None, datetime | None, int | None, datetime]]:
    """Build tournament_venues tuples."""
    by_venue: dict[int, set[int]] = defaultdict(set)
    srid_by_venue: dict[int, int | None] = {}
    tid = team_rows[0].tournament_id if team_rows else None
    if tid is None:
        return []
    for tr in team_rows:
        by_venue[tr.venue_id].add(tr.team_id)
        if tr.synch_request_id is not None:
            srid_by_venue.setdefault(tr.venue_id, tr.synch_request_id)
    out: list[tuple[int, int, int, bool, int | None, datetime | None, int | None, datetime]] = []
    for vid, teams in sorted(by_venue.items()):
        n = len(teams)
        meta = synch_by_venue.get(vid)
        approx = meta.approximate_teams_count if meta else None
        date_start = meta.date_start if meta else None
        srid = (meta.synch_request_id if meta else None) or srid_by_venue.get(vid)
        out.append(
            (
                int(tid),
                int(vid),
                n,
                n == 1,
                approx,
                date_start,
                srid,
                fetched_at,
            )
        )
    return out


def load_tournament_ids_from_db(
    *,
    database_url: str | None = None,
    min_tournament_date: str | None = "2015-01-01",
    sync_only: bool = True,
    limit: int | None = None,
) -> list[int]:
    import psycopg2

    url = database_url or os.environ.get(
        "DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres"
    )
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    date_cond = ""
    params: list[Any] = []
    if min_tournament_date:
        date_cond = " AND (t.start_datetime IS NULL OR t.start_datetime >= %s::timestamp)"
        params.append(min_tournament_date)
    type_cond = ""
    if sync_only:
        type_cond = """
          AND (
            (LOWER(COALESCE(t.type, '')) LIKE '%%синхрон%%'
             OR LOWER(COALESCE(t.type, '')) LIKE '%%sync%%')
            AND LOWER(COALESCE(t.type, '')) NOT LIKE '%%асинхрон%%'
            AND LOWER(COALESCE(t.type, '')) NOT LIKE '%%async%%'
          )
        """
    cur.execute(
        f"""
        SELECT t.id
        FROM public.tournaments t
        WHERE COALESCE(t.questions_count, 0) >= 10
          AND EXISTS (
            SELECT 1 FROM public.tournament_results r
            WHERE r.tournament_id = t.id AND r.points_mask IS NOT NULL
          )
        {date_cond}
        {type_cond}
        ORDER BY t.id
        """,
        tuple(params),
    )
    ids = [int(r[0]) for r in cur.fetchall() if r[0] is not None]
    conn.close()
    if limit is not None:
        ids = ids[: int(limit)]
    return ids


def load_tournament_ids_from_cache(
    cache_path: Path,
    *,
    sync_only: bool = True,
    limit: int | None = None,
) -> list[int]:
    from data import load_cached

    _arrays, maps = load_cached(cache_path)
    ids: list[int] = []
    for g_idx, tid in enumerate(maps.idx_to_game_id):
        if sync_only and maps.game_type is not None:
            gt = str(maps.game_type[g_idx]).lower()
            if gt != "sync":
                continue
        ids.append(int(tid))
    if limit is not None:
        ids = ids[: int(limit)]
    return ids


def pending_tournament_ids(
    con: duckdb.DuckDBPyConnection,
    all_ids: Iterable[int],
    *,
    resume: bool,
) -> list[int]:
    if not resume:
        return list(all_ids)
    done = {
        int(r[0])
        for r in con.execute(
            "SELECT tournament_id FROM venue_fetch_state WHERE http_status = 200"
        ).fetchall()
    }
    return [tid for tid in all_ids if tid not in done]


def write_tournament_overlay(
    con: duckdb.DuckDBPyConnection,
    tournament_id: int,
    team_rows: list[TeamVenueRow],
    venues: dict[int, VenueRow],
    tour_venue_rows: list[tuple[int, int, int, bool, int | None, datetime | None, int | None, datetime]],
    synch_by_venue: dict[int, SynchRequestDetail],
    *,
    fetched_at: datetime,
) -> None:
    for v in venues.values():
        con.execute(
            """
            INSERT INTO venues (
                venue_id, name, town_id, town_name,
                venue_type_id, venue_type_name, is_online, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (venue_id) DO UPDATE SET
                name = excluded.name,
                town_id = excluded.town_id,
                town_name = excluded.town_name,
                venue_type_id = excluded.venue_type_id,
                venue_type_name = excluded.venue_type_name,
                is_online = excluded.is_online,
                fetched_at = excluded.fetched_at
            """,
            [
                v.venue_id,
                v.name,
                v.town_id,
                v.town_name,
                v.venue_type_id,
                v.venue_type_name,
                v.is_online,
                fetched_at,
            ],
        )
    for tr in team_rows:
        con.execute(
            """
            INSERT INTO team_tournament_venue (
                tournament_id, team_id, venue_id, synch_request_id, fetched_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                tr.tournament_id,
                tr.team_id,
                tr.venue_id,
                tr.synch_request_id,
                fetched_at,
            ],
        )
    for meta in synch_by_venue.values():
        con.execute(
            """
            INSERT INTO synch_requests (
                synch_request_id, tournament_id, venue_id, date_start,
                status, approximate_teams_count, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (synch_request_id) DO UPDATE SET
                tournament_id = excluded.tournament_id,
                venue_id = excluded.venue_id,
                date_start = excluded.date_start,
                status = excluded.status,
                approximate_teams_count = excluded.approximate_teams_count,
                fetched_at = excluded.fetched_at
            """,
            [
                meta.synch_request_id,
                meta.tournament_id,
                meta.venue_id,
                meta.date_start,
                meta.status,
                meta.approximate_teams_count,
                fetched_at,
            ],
        )
    for row in tour_venue_rows:
        con.execute(
            """
            INSERT INTO tournament_venues (
                tournament_id, venue_id, teams_played, is_mono,
                approx_teams_declared, date_start, synch_request_id, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            list(row),
        )


def fetch_one_tournament(
    con: duckdb.DuckDBPyConnection,
    tournament_id: int,
    *,
    api_base: str = DEFAULT_API_BASE,
    fetch_approx: bool = True,
    sleep_sec: float = 0.25,
    timeout_sec: float = 60.0,
) -> None:
    fetched_at = utc_now()
    try:
        status, rows = fetch_tournament_results(
            tournament_id,
            api_base=api_base,
            sleep_sec=sleep_sec,
            timeout_sec=timeout_sec,
        )
        team_rows, venues, missing = parse_results_rows(tournament_id, rows)
        synch_by_venue: dict[int, SynchRequestDetail] = {}
        if fetch_approx and team_rows:
            synch_by_venue = fetch_synch_requests_for_tournament(
                tournament_id,
                api_base=api_base,
                sleep_sec=sleep_sec,
                timeout_sec=timeout_sec,
            )
        tour_venue_rows = aggregate_tournament_venues(
            team_rows, synch_by_venue, fetched_at=fetched_at
        )
        delete_tournament_rows(con, tournament_id)
        if team_rows:
            write_tournament_overlay(
                con,
                tournament_id,
                team_rows,
                venues,
                tour_venue_rows,
                synch_by_venue,
                fetched_at=fetched_at,
            )
        else:
            for v in venues.values():
                con.execute(
                    """
                    INSERT INTO venues (
                        venue_id, name, town_id, town_name,
                        venue_type_id, venue_type_name, is_online, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (venue_id) DO UPDATE SET
                        name = excluded.name,
                        town_id = excluded.town_id,
                        town_name = excluded.town_name,
                        venue_type_id = excluded.venue_type_id,
                        venue_type_name = excluded.venue_type_name,
                        is_online = excluded.is_online,
                        fetched_at = excluded.fetched_at
                    """,
                    [
                        v.venue_id,
                        v.name,
                        v.town_id,
                        v.town_name,
                        v.venue_type_id,
                        v.venue_type_name,
                        v.is_online,
                        fetched_at,
                    ],
                )
        upsert_fetch_state(
            con,
            tournament_id=tournament_id,
            http_status=status,
            n_results=len(rows),
            n_with_venue=len(team_rows),
            error_message=None,
            fetched_at=fetched_at,
        )
    except RatingApiError as e:
        upsert_fetch_state(
            con,
            tournament_id=tournament_id,
            http_status=e.status or 0,
            n_results=0,
            n_with_venue=0,
            error_message=str(e),
            fetched_at=fetched_at,
        )
        raise


def run_fetch(
    tournament_ids: list[int],
    db_path: Path = DEFAULT_DB_PATH,
    *,
    api_base: str = DEFAULT_API_BASE,
    resume: bool = True,
    fetch_approx: bool = True,
    sleep_sec: float = 0.25,
    timeout_sec: float = 60.0,
    show_progress: bool = True,
) -> dict[str, int]:
    con = open_db(db_path)
    ensure_schema(con)
    todo = pending_tournament_ids(con, tournament_ids, resume=resume)
    stats = {"ok": 0, "err": 0, "skipped": len(tournament_ids) - len(todo)}

    iterator: Iterator[int] = iter(todo)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(todo, desc="venue overlay", unit="tournament")
        except ImportError:
            pass

    for tid in iterator:
        try:
            fetch_one_tournament(
                con,
                tid,
                api_base=api_base,
                fetch_approx=fetch_approx,
                sleep_sec=sleep_sec,
                timeout_sec=timeout_sec,
            )
            stats["ok"] += 1
        except RatingApiError:
            stats["err"] += 1
            if show_progress and hasattr(iterator, "write"):
                iterator.write(f"  WARN: tournament {tid} failed")  # type: ignore[union-attr]

    con.close()
    return stats


def print_summary(db_path: Path = DEFAULT_DB_PATH) -> None:
    con = open_db(db_path)
    ensure_schema(con)
    n_fetch = con.execute("SELECT COUNT(*) FROM venue_fetch_state").fetchone()[0]
    n_ok = con.execute(
        "SELECT COUNT(*) FROM venue_fetch_state WHERE http_status = 200"
    ).fetchone()[0]
    n_team = con.execute("SELECT COUNT(*) FROM team_tournament_venue").fetchone()[0]
    n_mono = con.execute(
        "SELECT COUNT(*) FROM tournament_venues WHERE is_mono"
    ).fetchone()[0]
    n_tv = con.execute("SELECT COUNT(*) FROM tournament_venues").fetchone()[0]
    con.close()
    print(f"venue_fetch_state: {n_ok}/{n_fetch} OK")
    print(f"team_tournament_venue rows: {n_team}")
    print(f"tournament_venues: {n_tv} ({n_mono} mono)")
