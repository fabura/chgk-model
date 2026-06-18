"""Data access for ЧГКарта (/map)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

from . import db

_GEO_META_PATH = Path(__file__).resolve().parents[1] / "data" / "map_geo.json"
_geo_meta_cache: dict | None = None

ACTIVE_DAYS_DEFAULT = 60
ACTIVE_MIN_RADIUS = 4.0
ACTIVE_MAX_RADIUS = 14.0
INACTIVE_MIN_RADIUS = 2.0
INACTIVE_MAX_RADIUS = 5.0


def _load_geo_meta() -> dict:
    global _geo_meta_cache
    if _geo_meta_cache is not None:
        return _geo_meta_cache
    defaults: dict = {"detail_country_ids": [21, 26], "country_iso": {}}
    try:
        row = db.query_one("SELECT map_scratch_meta FROM site_meta LIMIT 1")
        raw = row.get("map_scratch_meta") if row else None
        if raw:
            meta = json.loads(raw) if isinstance(raw, str) else raw
            _geo_meta_cache = {
                "detail_country_ids": meta.get("detail_country_ids")
                or defaults["detail_country_ids"],
                "country_iso": meta.get("country_iso") or {},
            }
            return _geo_meta_cache
    except Exception:
        pass
    if _GEO_META_PATH.exists():
        _geo_meta_cache = json.loads(_GEO_META_PATH.read_text(encoding="utf-8"))
    else:
        _geo_meta_cache = defaults
    return _geo_meta_cache


def clear_geo_meta_cache() -> None:
    global _geo_meta_cache
    _geo_meta_cache = None


def _marker_radius(
    n_games: int,
    n_teams: int,
    *,
    min_r: float,
    max_r: float,
) -> float:
    weight = math.sqrt(max(1, n_games) * max(1, max(n_teams, 1)))
    t = min(1.0, math.log1p(weight) / math.log1p(500))
    return min_r + t * (max_r - min_r)


def get_venue_markers(*, active_days: int = ACTIVE_DAYS_DEFAULT) -> list[dict[str, Any]]:
    """One marker per city (town_id); stats summed across venues in that city."""
    rows = db.query(
        """
        SELECT
            mv.town_id,
            max(mv.town_name) AS town_name,
            max(mv.region_name) AS region_name,
            max(mv.country_name) AS country_name,
            max(mv.lat) AS lat,
            max(mv.lon) AS lon,
            count(DISTINCT mv.venue_id) AS n_venues,
            sum(mvs.n_team_games) AS n_team_games,
            sum(mvs.n_teams) AS n_teams,
            sum(mvs.n_tournaments) AS n_tournaments,
            max(mvs.last_game_date) AS last_game_date,
            sum(mvs.n_team_games_60d) AS n_team_games_60d,
            sum(mvs.n_teams_60d) AS n_teams_60d
        FROM map_venue_stats mvs
        JOIN map_venues mv ON mv.venue_id = mvs.venue_id
        WHERE mv.lat IS NOT NULL
          AND mv.lon IS NOT NULL
          AND mv.town_id IS NOT NULL
        GROUP BY mv.town_id
        ORDER BY n_team_games DESC
        """
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        recent = int(r.get("n_team_games_60d") or 0)
        n_games = int(r["n_team_games"] or 0)
        n_teams = int(r["n_teams"] or 0)
        active = recent > 0
        if active:
            radius = _marker_radius(
                recent,
                int(r.get("n_teams_60d") or 0),
                min_r=ACTIVE_MIN_RADIUS,
                max_r=ACTIVE_MAX_RADIUS,
            )
        else:
            # Inactive: small dots; lifetime volume only nudges size slightly.
            radius = _marker_radius(
                min(n_games, 300),
                min(n_teams, 150),
                min_r=INACTIVE_MIN_RADIUS,
                max_r=INACTIVE_MAX_RADIUS,
            )
        out.append(
            {
                "town_id": int(r["town_id"]),
                "town_name": r.get("town_name"),
                "region_name": r.get("region_name"),
                "country_name": r.get("country_name"),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "n_venues": int(r["n_venues"] or 1),
                "n_team_games": n_games,
                "n_teams": n_teams,
                "n_tournaments": int(r["n_tournaments"] or 0),
                "n_recent": recent,
                "active": active,
                "radius": radius,
                "last_game_date": (
                    str(r["last_game_date"]) if r.get("last_game_date") else None
                ),
            }
        )
    return out


def get_map_summary() -> dict[str, Any]:
    row = db.query_one(
        """
        SELECT
            count(DISTINCT mv.town_id) AS n_towns,
            count(DISTINCT mv.town_id) FILTER (
                WHERE EXISTS (
                    SELECT 1 FROM map_venue_stats mvs2
                    WHERE mvs2.venue_id = mv.venue_id
                      AND mvs2.n_team_games_60d > 0
                )
            ) AS n_active
        FROM map_venues mv
        WHERE mv.lat IS NOT NULL AND mv.town_id IS NOT NULL
        """
    ) or {"n_towns": 0, "n_active": 0}
    return {
        "n_towns": int(row["n_towns"] or 0),
        "n_active": int(row["n_active"] or 0),
        "active_days": ACTIVE_DAYS_DEFAULT,
    }


def get_player_scratch(player_id: int) -> Optional[dict[str, Any]]:
    player = db.query_one(
        "SELECT player_id, last_name, first_name, games "
        "FROM players WHERE player_id = ?",
        [player_id],
    )
    if not player:
        return None

    regions = db.query(
        """
        SELECT region_id, region_name, country_id, country_name, n_games
        FROM map_player_regions
        WHERE player_id = ?
        ORDER BY n_games DESC
        """,
        [player_id],
    )

    meta = _load_geo_meta()
    detail_ids = set(meta.get("detail_country_ids") or [21, 26])
    iso_map = meta.get("country_iso") or {}

    region_ids: list[int] = []
    countries: list[dict[str, Any]] = []
    country_agg: dict[int, dict[str, Any]] = {}

    for r in regions:
        cid = int(r["country_id"])
        rid = r.get("region_id")
        n = int(r["n_games"] or 0)
        if cid in detail_ids and rid is not None and int(rid) > 0:
            region_ids.append(int(rid))
        ca = country_agg.setdefault(
            cid,
            {
                "country_id": cid,
                "country_name": r.get("country_name"),
                "iso_a2": iso_map.get(str(cid)),
                "n_games": 0,
            },
        )
        ca["n_games"] += n

    for cid, ca in sorted(country_agg.items(), key=lambda x: -x[1]["n_games"]):
        if cid not in detail_ids:
            countries.append(ca)

    region_list = [
        {
            "region_id": int(r["region_id"]) if r.get("region_id") is not None else None,
            "region_name": r.get("region_name"),
            "country_id": int(r["country_id"]),
            "country_name": r.get("country_name"),
            "n_games": int(r["n_games"] or 0),
        }
        for r in regions
    ]

    # Label positions for regions without a GeoJSON polygon (e.g. Москва).
    region_labels: list[dict[str, Any]] = []
    try:
        label_rows = db.query(
            """
            WITH top_town AS (
                SELECT player_id, region_id, lat, lon,
                       row_number() OVER (
                           PARTITION BY player_id, region_id ORDER BY n_games DESC
                       ) AS rn
                FROM map_player_towns
                WHERE player_id = ?
            )
            SELECT mpr.region_id, mpr.region_name, mpr.n_games,
                   tt.lat, tt.lon
            FROM map_player_regions mpr
            LEFT JOIN top_town tt
              ON tt.player_id = mpr.player_id
             AND tt.region_id = mpr.region_id
             AND tt.rn = 1
            WHERE mpr.player_id = ?
              AND mpr.region_id > 0
            """,
            [player_id, player_id],
        )
        for lbl in label_rows:
            region_labels.append(
                {
                    "region_id": int(lbl["region_id"]),
                    "region_name": lbl.get("region_name"),
                    "n_games": int(lbl["n_games"] or 0),
                    "lat": float(lbl["lat"]) if lbl.get("lat") is not None else None,
                    "lon": float(lbl["lon"]) if lbl.get("lon") is not None else None,
                }
            )
    except Exception:
        pass

    scratch_games = sum(int(r["n_games"] or 0) for r in regions)

    return {
        "player_id": player_id,
        "name": _player_name(player),
        "games": int(player.get("games") or 0),
        "regions": region_list,
        "region_ids": sorted(set(region_ids)),
        "countries": countries,
        "region_labels": region_labels,
        "scratch_games": scratch_games,
        "detail_country_ids": sorted(detail_ids),
        "country_iso": iso_map,
    }


def search_players(q: str, *, limit: int = 15) -> list[dict[str, Any]]:
    q = q.strip()
    if len(q) < 2:
        return []
    if q.isdigit():
        rows = db.query(
            """
            SELECT p.player_id, p.last_name, p.first_name, p.games,
                   count(mpr.region_id) AS n_regions
            FROM players p
            LEFT JOIN map_player_regions mpr ON mpr.player_id = p.player_id
            WHERE p.player_id = ?
            GROUP BY p.player_id, p.last_name, p.first_name, p.games
            LIMIT 1
            """,
            [int(q)],
        )
    else:
        rows = db.query(
            """
            SELECT p.player_id, p.last_name, p.first_name, p.games,
                   count(mpr.region_id) AS n_regions
            FROM players p
            JOIN map_player_regions mpr ON mpr.player_id = p.player_id
            WHERE p.last_name ILIKE ? || '%'
               OR p.first_name ILIKE ? || '%'
               OR (p.last_name || ' ' || p.first_name) ILIKE '%' || ? || '%'
            GROUP BY p.player_id, p.last_name, p.first_name, p.games
            ORDER BY p.games DESC
            LIMIT ?
            """,
            [q, q, q, limit],
        )
    return [
        {
            "player_id": int(r["player_id"]),
            "name": _player_name(r),
            "games": int(r.get("games") or 0),
            "n_regions": int(r.get("n_regions") or 0),
        }
        for r in rows
    ]


def _player_name(p: dict) -> str:
    last = (p.get("last_name") or "").strip()
    first = (p.get("first_name") or "").strip()
    return f"{last} {first}".strip() or f"#{p.get('player_id')}"


def map_tables_present() -> bool:
    try:
        row = db.query_one("SELECT count(*) AS n FROM map_venue_stats")
        return bool(row and int(row["n"] or 0) > 0)
    except Exception:
        return False
