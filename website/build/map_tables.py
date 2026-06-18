"""Bake ЧГКарта tables into chgk.duckdb."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import duckdb

from venue_overlay.api import DEFAULT_API_BASE, DEFAULT_USER_AGENT

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_GEO = REPO_ROOT / "website" / "data" / "map_geo.json"
DEFAULT_VENUE_DB = REPO_ROOT / "data" / "venue_overlay.duckdb"
DEFAULT_TOURNAMENT_TOWNS = REPO_ROOT / "website" / "data" / "tournament_towns.json"
DEFAULT_SYNC_VENUE_EXTRA = REPO_ROOT / "website" / "data" / "sync_venue_extra.json"

# Rating API omits country on some RF regions (e.g. Крым → region_id=71 only).
ORPHAN_REGION_COUNTRY: dict[int, tuple[int, str]] = {
    71: (21, "Россия"),
}


def _normalize_map_location(
    region_id: int | None,
    region_name: str | None,
    country_id: int | None,
    country_name: str | None,
) -> tuple[int | None, str | None, int | None, str | None]:
    if country_id is not None or region_id is None:
        return region_id, region_name, country_id, country_name
    fill = ORPHAN_REGION_COUNTRY.get(int(region_id))
    if not fill:
        return region_id, region_name, country_id, country_name
    return int(region_id), region_name, int(fill[0]), fill[1]


def _normalize_town_dict(t: dict) -> dict:
    out = dict(t)
    rid, rn, cid, cn = _normalize_map_location(
        out.get("region_id"),
        out.get("region_name"),
        out.get("country_id"),
        out.get("country_name"),
    )
    out["region_id"] = rid
    out["region_name"] = rn
    out["country_id"] = cid
    out["country_name"] = cn
    return out


MAP_DDL = """
CREATE TABLE IF NOT EXISTS map_venues (
    venue_id INTEGER PRIMARY KEY,
    name TEXT,
    town_id INTEGER,
    town_name TEXT,
    region_id INTEGER,
    region_name TEXT,
    country_id INTEGER,
    country_name TEXT,
    lat DOUBLE,
    lon DOUBLE,
    is_online BOOLEAN
);

CREATE TABLE IF NOT EXISTS map_venue_stats (
    venue_id INTEGER PRIMARY KEY,
    n_team_games INTEGER NOT NULL,
    n_teams INTEGER NOT NULL,
    n_tournaments INTEGER NOT NULL,
    last_game_date DATE,
    n_team_games_60d INTEGER NOT NULL,
    n_teams_60d INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS map_player_regions (
    player_id INTEGER NOT NULL,
    region_id INTEGER NOT NULL,
    region_name TEXT,
    country_id INTEGER NOT NULL,
    country_name TEXT,
    n_games INTEGER NOT NULL,
    PRIMARY KEY (player_id, country_id, region_id)
);

CREATE TABLE IF NOT EXISTS map_player_towns (
    player_id INTEGER NOT NULL,
    town_id INTEGER NOT NULL,
    town_name TEXT,
    region_id INTEGER,
    region_name TEXT,
    country_id INTEGER,
    country_name TEXT,
    lat DOUBLE,
    lon DOUBLE,
    n_games INTEGER NOT NULL,
    PRIMARY KEY (player_id, town_id)
);
"""


def _log(msg: str) -> None:
    print(msg, flush=True)


def _towns_from_geo(geo: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for raw in (geo.get("towns") or {}).values():
        tid = int(raw["town_id"])
        out[tid] = raw
    return out


def _fetch_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def _fetch_venue_meta_from_api(venue_id: int) -> dict | None:
    data = _fetch_json(f"{DEFAULT_API_BASE}/venues/{venue_id}")
    if not data:
        return None
    town = data.get("town") or {}
    country = town.get("country") or {}
    region = town.get("region") or {}
    return {
        "venue_id": int(venue_id),
        "name": (data.get("name") or "").strip() or None,
        "town_id": town.get("id"),
        "town_name": (town.get("name") or "").strip() or None,
        "region_id": region.get("id"),
        "region_name": (region.get("name") or "").strip() or None,
        "country_id": country.get("id"),
        "country_name": (country.get("name") or "").strip() or None,
        "lat": None,
        "lon": None,
        "is_online": False,
    }


def _supplement_map_venues(con: duckdb.DuckDBPyConnection, geo: dict) -> int:
    """Insert map_venues rows for overlay/extra venue_ids missing from map_geo."""
    towns = geo.get("towns") or {}
    venues_geo = geo.get("venues") or {}
    missing = con.execute(
        """
        SELECT DISTINCT venue_id FROM (
            SELECT venue_id FROM vo.team_tournament_venue
            UNION
            SELECT venue_id FROM _sync_venue_extra
        )
        WHERE venue_id IS NOT NULL
          AND venue_id NOT IN (SELECT venue_id FROM map_venues)
        """
    ).fetchall()
    rows: list[tuple] = []
    for (vid,) in missing:
        vid = int(vid)
        meta = venues_geo.get(str(vid)) or _fetch_venue_meta_from_api(vid)
        if not meta:
            continue
        tid = meta.get("town_id")
        if tid is not None:
            town = towns.get(str(tid)) or towns.get(int(tid))
            if town:
                meta.setdefault("lat", town.get("lat"))
                meta.setdefault("lon", town.get("lon"))
        if meta.get("lat") is None:
            for other in venues_geo.values():
                if other.get("town_id") == tid and other.get("lat") is not None:
                    meta["lat"] = other["lat"]
                    meta["lon"] = other["lon"]
                    break
        rid, rn, cid, cn = _normalize_map_location(
            meta.get("region_id"),
            meta.get("region_name"),
            meta.get("country_id"),
            meta.get("country_name"),
        )
        rows.append(
            (
                vid,
                meta.get("name"),
                tid,
                meta.get("town_name"),
                rid,
                rn,
                cid,
                cn,
                meta.get("lat"),
                meta.get("lon"),
                bool(meta.get("is_online")),
            )
        )
        time.sleep(0.03)
    if rows:
        con.executemany(
            """
            INSERT INTO map_venues (
                venue_id, name, town_id, town_name, region_id, region_name,
                country_id, country_name, lat, lon, is_online
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def _fetch_town_meta_from_api(town_id: int) -> dict | None:
    data = _fetch_json(f"{DEFAULT_API_BASE}/towns/{town_id}")
    if not data:
        return None
    country = data.get("country") or {}
    region = data.get("region") or {}
    return {
        "town_id": town_id,
        "town_name": (data.get("name") or "").strip() or None,
        "region_id": region.get("id"),
        "region_name": (region.get("name") or "").strip() or None,
        "country_id": country.get("id"),
        "country_name": (country.get("name") or "").strip() or None,
        "lat": None,
        "lon": None,
    }


def _ensure_tournament_towns(
    tournament_ids: list[int],
    *,
    cache_path: Path = DEFAULT_TOURNAMENT_TOWNS,
) -> dict[int, int]:
    """Map offline tournament_id → host town_id (rating API ``idtown``)."""
    cache: dict[str, int | None] = {}
    if cache_path.exists():
        cache = {
            str(k): v for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()
        }
    missing = [tid for tid in tournament_ids if str(tid) not in cache]
    if missing:
        _log(f"  fetching idtown for {len(missing):,} offline tournaments…")
        for i, tid in enumerate(missing):
            if i and i % 200 == 0:
                _log(f"    idtown {i:,}/{len(missing):,}…")
            data = _fetch_json(f"{DEFAULT_API_BASE}/tournaments/{tid}")
            idtown = data.get("idtown") if data else None
            cache[str(tid)] = int(idtown) if idtown is not None else None
            time.sleep(0.03)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    want = set(tournament_ids)
    return {
        int(k): int(v)
        for k, v in cache.items()
        if v is not None and int(k) in want
    }


def _fetch_pre2015_offline_rosters(
    con: duckdb.DuckDBPyConnection, database_url: str | None
) -> int:
    """Rosters for pre-2015 offline tournaments (absent from player_games)."""
    con.execute(
        "CREATE TEMP TABLE _pre2015_offline_rosters "
        "(player_id INTEGER, tournament_id INTEGER)"
    )
    url = database_url or os.environ.get(
        "DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres"
    )
    try:
        import psycopg2

        pg = psycopg2.connect(url)
        cur = pg.cursor()
        cur.execute(
            """
            SELECT tr.player_id, tr.tournament_id
            FROM public.tournament_rosters tr
            JOIN public.tournaments t ON t.id = tr.tournament_id
            WHERE tr.player_id IS NOT NULL
              AND t.start_datetime < DATE '2015-01-01'
              AND COALESCE(LOWER(t.type), '') NOT LIKE '%синхрон%'
              AND COALESCE(LOWER(t.type), '') NOT LIKE '%асинхрон%'
              AND COALESCE(LOWER(t.type), '') NOT LIKE '%sync%'
              AND COALESCE(LOWER(t.type), '') NOT LIKE '%async%'
            """
        )
        rows = cur.fetchall()
        pg.close()
    except Exception as e:
        _log(f"  pre-2015 offline rosters skipped (PG unavailable: {e})")
        return 0
    if rows:
        con.executemany(
            "INSERT INTO _pre2015_offline_rosters VALUES (?, ?)", rows
        )
    return len(rows)


def _ensure_sync_venue_extra(
    pairs: list[tuple[int, int]],
    *,
    cache_path: Path = DEFAULT_SYNC_VENUE_EXTRA,
) -> list[tuple[int, int, int]]:
    """Resolve team→venue for sync games missing from venue_overlay (API synchRequest)."""
    cache: dict[str, dict[str, int | None]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    by_tid: dict[int, set[int]] = {}
    for tid, team_id in pairs:
        tid_i, team_i = int(tid), int(team_id)
        tid_s, team_s = str(tid_i), str(team_i)
        tid_cache = cache.get(tid_s) or {}
        if team_s in tid_cache:
            continue
        by_tid.setdefault(tid_i, set()).add(team_i)

    if by_tid:
        _log(f"  fetching synchRequest venues for {len(by_tid):,} tournaments…")
        for i, (tid, team_ids) in enumerate(by_tid.items()):
            if i and i % 50 == 0:
                _log(f"    sync venues {i:,}/{len(by_tid):,}…")
            tid_s = str(tid)
            cache.setdefault(tid_s, {})
            data = _fetch_json(f"{DEFAULT_API_BASE}/tournaments/{tid}/results")
            found: set[int] = set()
            if isinstance(data, list):
                for item in data:
                    team = (item.get("team") or {}).get("id")
                    if team is None or int(team) not in team_ids:
                        continue
                    venue = ((item.get("synchRequest") or {}).get("venue") or {}).get(
                        "id"
                    )
                    cache[tid_s][str(int(team))] = (
                        int(venue) if venue is not None else None
                    )
                    found.add(int(team))
            for team_i in team_ids:
                if team_i not in found:
                    cache[tid_s].setdefault(str(team_i), None)
            time.sleep(0.05)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    rows: list[tuple[int, int, int]] = []
    for tid, team_id in pairs:
        venue = (cache.get(str(int(tid))) or {}).get(str(int(team_id)))
        if venue is not None:
            rows.append((int(tid), int(team_id), int(venue)))
    return rows


def _fetch_sync_venue_gaps(con: duckdb.DuckDBPyConnection) -> int:
    """team→venue for sync/async games missing from venue_overlay (API synchRequest)."""
    con.execute(
        "CREATE TEMP TABLE _sync_venue_extra ("
        "tournament_id INTEGER, team_id INTEGER, venue_id INTEGER)"
    )
    missing = con.execute(
        """
        SELECT DISTINCT tg.tournament_id, tg.team_id
        FROM team_games tg
        JOIN tournaments t ON t.tournament_id = tg.tournament_id
        WHERE t.type IN ('sync', 'async')
          AND NOT EXISTS (
            SELECT 1 FROM vo.team_tournament_venue ttv
            WHERE ttv.tournament_id = tg.tournament_id
              AND ttv.team_id = tg.team_id
          )
        """
    ).fetchall()
    if not missing:
        return 0
    rows = _ensure_sync_venue_extra([(int(t), int(team)) for t, team in missing])
    if rows:
        con.executemany(
            "INSERT INTO _sync_venue_extra VALUES (?, ?, ?)", rows
        )
    return len(rows)


def _prepare_offline_town_tables(
    con: duckdb.DuckDBPyConnection,
    geo: dict,
) -> int:
    """Temp tables ``_tournament_towns`` and ``_offline_town_meta`` for scratch map."""
    offline_tids = [
        int(r[0])
        for r in con.execute(
            """
            SELECT DISTINCT tournament_id FROM (
                SELECT pg.tournament_id
                FROM player_games pg
                JOIN tournaments t ON t.tournament_id = pg.tournament_id
                WHERE t.type = 'offline'
                UNION
                SELECT tournament_id FROM _pre2015_offline_rosters
            )
            """
        ).fetchall()
    ]
    if not offline_tids:
        con.execute(
            "CREATE TEMP TABLE _tournament_towns "
            "(tournament_id INTEGER, town_id INTEGER)"
        )
        con.execute(
            "CREATE TEMP TABLE _offline_town_meta ("
            "town_id INTEGER, town_name TEXT, region_id INTEGER, region_name TEXT, "
            "country_id INTEGER, country_name TEXT, lat DOUBLE, lon DOUBLE)"
        )
        return 0

    tid_to_town = _ensure_tournament_towns(offline_tids)
    geo_towns = _towns_from_geo(geo)
    town_meta: dict[int, dict] = {}
    for town_id in set(tid_to_town.values()):
        if town_id in geo_towns:
            town_meta[town_id] = _normalize_town_dict(geo_towns[town_id])
        else:
            meta = _fetch_town_meta_from_api(town_id)
            if meta:
                meta = _normalize_town_dict(meta)
                if meta.get("country_id") is not None:
                    town_meta[town_id] = meta
            time.sleep(0.03)

    con.execute(
        "CREATE TEMP TABLE _tournament_towns "
        "(tournament_id INTEGER, town_id INTEGER)"
    )
    if tid_to_town:
        con.executemany(
            "INSERT INTO _tournament_towns VALUES (?, ?)",
            list(tid_to_town.items()),
        )
    con.execute(
        "CREATE TEMP TABLE _offline_town_meta ("
        "town_id INTEGER, town_name TEXT, region_id INTEGER, region_name TEXT, "
        "country_id INTEGER, country_name TEXT, lat DOUBLE, lon DOUBLE)"
    )
    if town_meta:
        con.executemany(
            "INSERT INTO _offline_town_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    int(t["town_id"]),
                    t.get("town_name"),
                    t.get("region_id"),
                    t.get("region_name"),
                    t.get("country_id"),
                    t.get("country_name"),
                    t.get("lat"),
                    t.get("lon"),
                )
                for t in town_meta.values()
            ],
        )
    return len(tid_to_town)


def _fetch_maskless_rosters(
    con: duckdb.DuckDBPyConnection, database_url: str | None
) -> int:
    """
    Load rosters for mask-less (pre-2015) team_games into the temp table
    ``_maskless_rosters`` so the scratch map covers games without a
    points_mask. Returns the number of rows loaded (0 if PG unreachable).
    """
    pairs = con.execute(
        """
        SELECT DISTINCT tg.tournament_id
        FROM team_games tg
        JOIN tournaments t ON t.tournament_id = tg.tournament_id
        WHERE t.type = 'sync'
          AND NOT COALESCE(tg.has_breakdown, TRUE)
        """
    ).fetchall()
    tids = [int(r[0]) for r in pairs]
    con.execute(
        "CREATE TEMP TABLE _maskless_rosters "
        "(tournament_id INTEGER, team_id INTEGER, player_id INTEGER)"
    )
    if not tids:
        return 0
    url = database_url or os.environ.get(
        "DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/postgres"
    )
    try:
        import psycopg2

        pg = psycopg2.connect(url)
        cur = pg.cursor()
        cur.execute(
            """
            SELECT tournament_id, team_id, player_id
            FROM public.tournament_rosters
            WHERE tournament_id = ANY(%s)
              AND team_id IS NOT NULL AND player_id IS NOT NULL
            """,
            (tids,),
        )
        rows = cur.fetchall()
        pg.close()
    except Exception as e:
        _log(f"  maskless rosters skipped (PG unavailable: {e})")
        return 0
    if rows:
        con.executemany(
            "INSERT INTO _maskless_rosters VALUES (?, ?, ?)", rows
        )
    return len(rows)


def bake_map_tables(
    con: duckdb.DuckDBPyConnection,
    *,
    map_geo_path: Path = DEFAULT_MAP_GEO,
    venue_db_path: Path = DEFAULT_VENUE_DB,
    active_days: int = 60,
    database_url: str | None = None,
) -> bool:
    """
    Populate map_* tables. Returns False if inputs are missing (non-fatal).
    """
    if not map_geo_path.exists() or not venue_db_path.exists():
        _log(
            f"  map tables skipped (need {map_geo_path.name} and venue_overlay.duckdb)"
        )
        return False

    _log("Building map tables…")
    geo = json.loads(map_geo_path.read_text(encoding="utf-8"))
    venues = geo.get("venues") or {}

    con.execute(MAP_DDL)
    con.execute("DELETE FROM map_venues")
    con.execute("DELETE FROM map_venue_stats")
    con.execute("DELETE FROM map_player_regions")
    con.execute("DELETE FROM map_player_towns")

    rows = []
    for v in venues.values():
        if v.get("is_online"):
            continue
        rid, rn, cid, cn = _normalize_map_location(
            v.get("region_id"),
            v.get("region_name"),
            v.get("country_id"),
            v.get("country_name"),
        )
        rows.append(
            (
                int(v["venue_id"]),
                v.get("name"),
                v.get("town_id"),
                v.get("town_name"),
                rid,
                rn,
                cid,
                cn,
                v.get("lat"),
                v.get("lon"),
                bool(v.get("is_online")),
            )
        )
    if rows:
        con.executemany(
            """
            INSERT INTO map_venues (
                venue_id, name, town_id, town_name, region_id, region_name,
                country_id, country_name, lat, lon, is_online
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    con.execute(f"ATTACH '{venue_db_path}' AS vo (READ_ONLY)")

    n_sync_venues = _fetch_sync_venue_gaps(con)
    if n_sync_venues:
        _log(f"  sync/async venue gaps filled from API: {n_sync_venues:,}")

    n_supplement = _supplement_map_venues(con, geo)
    if n_supplement:
        _log(f"  map_venues supplemented from API: {n_supplement:,}")

    cutoff = date.today() - timedelta(days=active_days)
    con.execute(
        """
        INSERT INTO map_venue_stats (
            venue_id, n_team_games, n_teams, n_tournaments,
            last_game_date, n_team_games_60d, n_teams_60d
        )
        WITH team_venue AS (
            SELECT tg.tournament_id, tg.team_id, ttv.venue_id
            FROM team_games tg
            JOIN tournaments t ON t.tournament_id = tg.tournament_id
            JOIN vo.team_tournament_venue ttv
              ON ttv.tournament_id = tg.tournament_id AND ttv.team_id = tg.team_id
            WHERE t.type IN ('sync', 'async')
            UNION
            SELECT tournament_id, team_id, venue_id FROM _sync_venue_extra
        )
        SELECT
            mv.venue_id,
            count(*)::INTEGER AS n_team_games,
            count(DISTINCT tg.team_id)::INTEGER AS n_teams,
            count(DISTINCT tg.tournament_id)::INTEGER AS n_tournaments,
            max(t.start_date) AS last_game_date,
            count(*) FILTER (
                WHERE t.start_date >= ?
            )::INTEGER AS n_team_games_60d,
            count(DISTINCT tg.team_id) FILTER (
                WHERE t.start_date >= ?
            )::INTEGER AS n_teams_60d
        FROM team_games tg
        JOIN tournaments t ON t.tournament_id = tg.tournament_id
        JOIN team_venue tv
          ON tv.tournament_id = tg.tournament_id AND tv.team_id = tg.team_id
        JOIN map_venues mv ON mv.venue_id = tv.venue_id
        WHERE t.type IN ('sync', 'async')
          AND NOT coalesce(mv.is_online, false)
          AND mv.lat IS NOT NULL
          AND mv.lon IS NOT NULL
        GROUP BY mv.venue_id
        """,
        [cutoff, cutoff],
    )

    # Pre-2015 tournaments have no points_mask, hence no player_games rows;
    # pull their rosters straight from Postgres so they still count.
    n_maskless = _fetch_maskless_rosters(con, database_url)
    if n_maskless:
        _log(f"  maskless roster rows: {n_maskless:,}")

    n_pre2015 = _fetch_pre2015_offline_rosters(con, database_url)
    if n_pre2015:
        _log(f"  pre-2015 offline roster rows: {n_pre2015:,}")

    n_offline_towns = _prepare_offline_town_tables(con, geo)
    if n_offline_towns:
        _log(f"  offline tournament towns: {n_offline_towns:,}")

    con.execute(
        """
        INSERT INTO map_player_regions (
            player_id, region_id, region_name, country_id, country_name, n_games
        )
        WITH coverage AS (
            SELECT pg.player_id, pg.tournament_id, pg.team_id
            FROM player_games pg
            UNION
            SELECT mr.player_id, mr.tournament_id, mr.team_id
            FROM _maskless_rosters mr
        ),
        sync_venue AS (
            SELECT c.player_id, c.tournament_id, c.team_id, ttv.venue_id
            FROM coverage c
            JOIN tournaments t ON t.tournament_id = c.tournament_id
            JOIN vo.team_tournament_venue ttv
              ON ttv.tournament_id = c.tournament_id AND ttv.team_id = c.team_id
            WHERE t.type IN ('sync', 'async')
            UNION
            SELECT c.player_id, c.tournament_id, c.team_id, sve.venue_id
            FROM coverage c
            JOIN tournaments t ON t.tournament_id = c.tournament_id
            JOIN _sync_venue_extra sve
              ON sve.tournament_id = c.tournament_id AND sve.team_id = c.team_id
            WHERE t.type IN ('sync', 'async')
        ),
        map_events AS (
            SELECT
                sv.player_id,
                coalesce(mv.region_id, -mv.country_id) AS region_id,
                mv.region_name,
                mv.country_id,
                mv.country_name
            FROM sync_venue sv
            JOIN map_venues mv ON mv.venue_id = sv.venue_id
            WHERE NOT coalesce(mv.is_online, false)
              AND mv.country_id IS NOT NULL
            UNION ALL
            SELECT
                pg.player_id,
                coalesce(tm.region_id, -tm.country_id) AS region_id,
                tm.region_name,
                tm.country_id,
                tm.country_name
            FROM player_games pg
            JOIN tournaments t ON t.tournament_id = pg.tournament_id
            JOIN _tournament_towns tt ON tt.tournament_id = pg.tournament_id
            JOIN _offline_town_meta tm ON tm.town_id = tt.town_id
            WHERE t.type = 'offline'
              AND tm.country_id IS NOT NULL
            UNION ALL
            SELECT
                pr.player_id,
                coalesce(tm.region_id, -tm.country_id) AS region_id,
                tm.region_name,
                tm.country_id,
                tm.country_name
            FROM _pre2015_offline_rosters pr
            JOIN _tournament_towns tt ON tt.tournament_id = pr.tournament_id
            JOIN _offline_town_meta tm ON tm.town_id = tt.town_id
            WHERE tm.country_id IS NOT NULL
        )
        SELECT
            player_id,
            region_id,
            max(region_name),
            country_id,
            max(country_name),
            count(*)::INTEGER AS n_games
        FROM map_events
        GROUP BY player_id, region_id, country_id
        """
    )
    con.execute(
        """
        INSERT INTO map_player_towns (
            player_id, town_id, town_name, region_id, region_name,
            country_id, country_name, lat, lon, n_games
        )
        WITH coverage AS (
            SELECT pg.player_id, pg.tournament_id, pg.team_id
            FROM player_games pg
            UNION
            SELECT mr.player_id, mr.tournament_id, mr.team_id
            FROM _maskless_rosters mr
        ),
        sync_venue AS (
            SELECT c.player_id, c.tournament_id, c.team_id, ttv.venue_id
            FROM coverage c
            JOIN tournaments t ON t.tournament_id = c.tournament_id
            JOIN vo.team_tournament_venue ttv
              ON ttv.tournament_id = c.tournament_id AND ttv.team_id = c.team_id
            WHERE t.type IN ('sync', 'async')
            UNION
            SELECT c.player_id, c.tournament_id, c.team_id, sve.venue_id
            FROM coverage c
            JOIN tournaments t ON t.tournament_id = c.tournament_id
            JOIN _sync_venue_extra sve
              ON sve.tournament_id = c.tournament_id AND sve.team_id = c.team_id
            WHERE t.type IN ('sync', 'async')
        ),
        town_events AS (
            SELECT
                sv.player_id,
                mv.town_id,
                mv.town_name,
                mv.region_id,
                mv.region_name,
                mv.country_id,
                mv.country_name,
                mv.lat,
                mv.lon
            FROM sync_venue sv
            JOIN map_venues mv ON mv.venue_id = sv.venue_id
            WHERE NOT coalesce(mv.is_online, false)
              AND mv.town_id IS NOT NULL
              AND mv.lat IS NOT NULL
            UNION ALL
            SELECT
                pg.player_id,
                tm.town_id,
                tm.town_name,
                tm.region_id,
                tm.region_name,
                tm.country_id,
                tm.country_name,
                tm.lat,
                tm.lon
            FROM player_games pg
            JOIN tournaments t ON t.tournament_id = pg.tournament_id
            JOIN _tournament_towns tt ON tt.tournament_id = pg.tournament_id
            JOIN _offline_town_meta tm ON tm.town_id = tt.town_id
            WHERE t.type = 'offline'
              AND tm.town_id IS NOT NULL
              AND tm.lat IS NOT NULL
              AND tm.lon IS NOT NULL
            UNION ALL
            SELECT
                pr.player_id,
                tm.town_id,
                tm.town_name,
                tm.region_id,
                tm.region_name,
                tm.country_id,
                tm.country_name,
                tm.lat,
                tm.lon
            FROM _pre2015_offline_rosters pr
            JOIN _tournament_towns tt ON tt.tournament_id = pr.tournament_id
            JOIN _offline_town_meta tm ON tm.town_id = tt.town_id
            WHERE tm.town_id IS NOT NULL
              AND tm.lat IS NOT NULL
              AND tm.lon IS NOT NULL
        )
        SELECT
            player_id,
            town_id,
            max(town_name),
            region_id,
            max(region_name),
            country_id,
            max(country_name),
            max(lat),
            max(lon),
            count(*)::INTEGER
        FROM town_events
        GROUP BY player_id, town_id, region_id, country_id
        """
    )

    con.execute("DROP TABLE IF EXISTS _maskless_rosters")
    con.execute("DROP TABLE IF EXISTS _pre2015_offline_rosters")
    con.execute("DROP TABLE IF EXISTS _sync_venue_extra")
    con.execute("DROP TABLE IF EXISTS _tournament_towns")
    con.execute("DROP TABLE IF EXISTS _offline_town_meta")

    n_venues = con.execute("SELECT count(*) FROM map_venue_stats").fetchone()[0]
    n_players = con.execute(
        "SELECT count(DISTINCT player_id) FROM map_player_regions"
    ).fetchone()[0]
    _log(f"  map: {n_venues:,} venues with stats, {n_players:,} players with regions")
    return True
