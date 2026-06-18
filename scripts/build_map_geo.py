#!/usr/bin/env python3
"""Build map geography cache for ЧГКарта.

Reads venue_overlay.duckdb, enriches each venue via GET /venues/{id},
geocodes towns (static table + Nominatim fallback), and writes:
  - website/data/map_geo.json
  - website/app/static/geo/{countries,ru_regions,ua_regions}.geojson

Run after fetch_venue_overlay.py and before build_db.py (or standalone).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402

from venue_overlay.api import DEFAULT_API_BASE, DEFAULT_USER_AGENT  # noqa: E402
from venue_overlay.store import DEFAULT_DB_PATH  # noqa: E402

GEO_DIR = REPO_ROOT / "website" / "app" / "static" / "geo"
OUT_JSON = REPO_ROOT / "website" / "data" / "map_geo.json"

# Countries where scratch-map shows admin-1 regions instead of whole country.
DETAIL_COUNTRY_IDS = {21, 26}  # Россия, Украина

# Rating API omits country on some RF regions (e.g. Крым → region_id=71 only).
ORPHAN_REGION_COUNTRY: dict[int, tuple[int, str]] = {
    71: (21, "Россия"),
}


def _fill_orphan_region_country(d: dict) -> None:
    rid = d.get("region_id")
    if d.get("country_id") is not None or rid is None:
        return
    fill = ORPHAN_REGION_COUNTRY.get(int(rid))
    if fill:
        d["country_id"] = fill[0]
        d["country_name"] = fill[1]

# Rating API country names (Russian) → ISO-3166-1 alpha-2 for choropleth matching.
RU_COUNTRY_NAME_TO_ISO: dict[str, str] = {
    "Австралия": "AU",
    "Австрия": "AT",
    "Азербайджан": "AZ",
    "Армения": "AM",
    "Беларусь": "BY",
    "Белоруссия": "BY",
    "Бельгия": "BE",
    "Болгария": "BG",
    "Великобритания": "GB",
    "Венгрия": "HU",
    "Вьетнам": "VN",
    "Германия": "DE",
    "Греция": "GR",
    "Грузия": "GE",
    "Дания": "DK",
    "Израиль": "IL",
    "Индия": "IN",
    "Ирландия": "IE",
    "Исландия": "IS",
    "Испания": "ES",
    "Италия": "IT",
    "Казахстан": "KZ",
    "Канада": "CA",
    "Киргизия": "KG",
    "Кыргызстан": "KG",
    "Китай": "CN",
    "Кипр": "CY",
    "Андорра": "AD",
    "Марокко": "MA",
    "Тунис": "TN",
    "Мальта": "MT",
    "Люксембург": "LU",
    "Северная Македония": "MK",
    "Бразилия": "BR",
    "Шри-Ланка": "LK",
    "Аргентина": "AR",
    "Колумбия": "CO",
    "Саудовская Аравия": "SA",
    "Монголия": "MN",
    "Корея": "KR",
    "Латвия": "LV",
    "Литва": "LT",
    "Молдова": "MD",
    "Нидерланды": "NL",
    "Норвегия": "NO",
    "ОАЭ": "AE",
    "Польша": "PL",
    "Португалия": "PT",
    "Россия": "RU",
    "Румыния": "RO",
    "Сербия": "RS",
    "Сингапур": "SG",
    "Словакия": "SK",
    "Словения": "SI",
    "США": "US",
    "Таджикистан": "TJ",
    "Таиланд": "TH",
    "Туркменистан": "TM",
    "Турция": "TR",
    "Узбекистан": "UZ",
    "Украина": "UA",
    "Финляндия": "FI",
    "Франция": "FR",
    "Хорватия": "HR",
    "Чехия": "CZ",
    "Черногория": "ME",
    "Малайзия": "MY",
    "Швейцария": "CH",
    "Швеция": "SE",
    "Эстония": "EE",
    "Южная Корея": "KR",
    "Япония": "JP",
}


def _build_country_iso(venues: dict[str, dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for v in venues.values():
        cid = v.get("country_id")
        cname = (v.get("country_name") or "").strip()
        if cid is None or not cname:
            continue
        iso = RU_COUNTRY_NAME_TO_ISO.get(cname)
        if iso:
            out[str(int(cid))] = iso
    return out

# Rough bounding boxes (lat_min, lat_max, lon_min, lon_max) for geocode validation.
COUNTRY_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "Россия": (41.0, 82.0, 19.0, 190.0),
    "Украина": (44.0, 52.5, 22.0, 41.5),
    "Беларусь": (51.0, 56.5, 23.0, 33.0),
    "Казахстан": (40.0, 56.0, 46.0, 88.0),
    "Швейцария": (45.7, 47.9, 5.8, 10.6),
    "Кипр": (34.4, 35.8, 32.0, 34.9),
    "Германия": (47.0, 55.5, 5.5, 15.5),
    "Израиль": (29.0, 33.6, 34.0, 36.0),
    "Грузия": (41.0, 43.7, 39.9, 46.8),
    "Армения": (38.7, 41.4, 43.3, 46.7),
    "Азербайджан": (38.3, 42.0, 44.5, 51.0),
    "Латвия": (55.6, 58.2, 20.5, 28.4),
    "Литва": (53.8, 56.5, 20.5, 27.0),
    "Эстония": (57.5, 59.8, 21.5, 28.3),
    "Польша": (49.0, 55.0, 14.0, 24.5),
    "Чехия": (48.5, 51.2, 12.0, 19.0),
    "США": (24.0, 50.0, -125.0, -66.0),
    "Вьетнам": (8.0, 24.0, 102.0, 110.0),
}


# Disambiguation for towns Nominatim often misses or confuses.
MANUAL_TOWN_COORDS: dict[tuple[str, str | None, str | None], tuple[float, float]] = {
    ("Зеленогорск (К)", "Красноярский край", "Россия"): (56.1089, 94.5985),
    ("Зеленогорск (СПб)", "Санкт-Петербург", "Россия"): (60.1971, 29.7072),
}


def _coords_match_country(
    country_name: str | None, lat: float, lon: float
) -> bool:
    if not country_name:
        return True
    box = COUNTRY_BOUNDS.get(country_name.strip())
    if not box:
        return True
    lat_min, lat_max, lon_min, lon_max = box
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

# Russian → Ukrainian oblast names for GeoJSON matching.
UA_REGION_RU_TO_UK: dict[str, str] = {
    "автономная республика крым": "Автономна Республіка Крим",
    "крым": "Автономна Республіка Крим",
    "севастополь": "Автономна Республіка Крим",
    "винницкая область": "Вінницька область",
    "волынская область": "Волинська область",
    "днепропетровская область": "Дніпропетровська область",
    "донецкая область": "Донецька область",
    "житомирская область": "Житомирська область",
    "закарпатская область": "Закарпатська область",
    "запорожская область": "Запорізька область",
    "ивано-франковская область": "Івано-Франківська область",
    "киев": "Київ",
    "киевская область": "Київська область",
    "кировоградская область": "Кіровоградська область",
    "луганская область": "Луганська область",
    "львовская область": "Львівська область",
    "николаевская область": "Миколаївська область",
    "одесская область": "Одеська область",
    "полтавская область": "Полтавська область",
    "ровенская область": "Рівненська область",
    "сумская область": "Сумська область",
    "тернопольская область": "Тернопільська область",
    "харьковская область": "Харківська область",
    "херсонская область": "Херсонська область",
    "хмельницкая область": "Хмельницька область",
    "черкасская область": "Черкаська область",
    "черниговская область": "Чернігівська область",
    "черновицкая область": "Чернівецька область",
}


def _norm_region(name: str | None) -> str:
    if not name:
        return ""
    s = name.lower().replace("ё", "е").strip()
    for prefix in ("республика ", "автономная ", "город "):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
    for suffix in (
        " область",
        " край",
        " республика",
        " автономный округ",
        " ао",
        " город",
        " (горсовет)",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s


# Natural Earth name_local → rating region_id (explicit overrides only).
RU_GEO_LOCAL_TO_RATING: dict[str, int] = {
    "республика татарстан": 140,
    "татарстан": 140,
    "санкт-петербург": 128,
    "москва": 90,
    "московская": 91,
    "ленинградская": 75,
    "республика башкортостан": 16,
    "республика мордовия": 89,
    "мордовия": 89,
    "республика адыгея": 1,
    "республика калмыкия": 54,
    "кабардино-балкарская республика": 49,
    "карачаево-черкесская республика": 51,
}

# Natural Earth English ``name`` → rating region_id (NE ``name_local`` is often wrong).
NE_EN_TO_RATING: dict[str, int] = {
    "moskva": 90,
    "moskovskaya": 91,
    "city of st. petersburg": 128,
    "sevastopol": 71,
    "crimea": 71,
    "arkhangel'sk": 10,
    "ingush": 48,
    "karelia": 59,
    "komi": 66,
    "mariy-el": 81,
    "udmurt": 148,
    "chuvash": 162,
    "altay": 8,
    "gorno-altay": 7,
    "chechnya": 161,
    "dagestan": 36,
    "sakha (yakutia)": 166,
    "perm'": 115,
    "kamchatka": 56,
    "north ossetia": 132,
    "chita": 42,
}


def _ru_region_id_for_geo(
    local: str,
    ru_by_norm: dict[str, int],
    *,
    en_name: str | None = None,
    admin_type: str | None = None,
) -> int | None:
    """Match a Natural Earth name_local to a rating region_id."""
    en = (en_name or "").strip().lower()
    typ = (admin_type or "").strip().lower()
    if en in NE_EN_TO_RATING:
        return NE_EN_TO_RATING[en]
    # NE labels Moskva city with the same name_local as the oblast — use EN name.
    if en == "moskva":
        return 90
    if en == "moskovskaya" or (typ == "oblast" and _norm_region(local) == "московская"):
        return 91
    n = _norm_region(local)
    if n in ru_by_norm:
        return ru_by_norm[n]
    raw = local.strip().lower()
    if raw in ru_by_norm:
        return ru_by_norm[raw]
    if raw in RU_GEO_LOCAL_TO_RATING:
        return RU_GEO_LOCAL_TO_RATING[raw]
    if n in RU_GEO_LOCAL_TO_RATING:
        return RU_GEO_LOCAL_TO_RATING[n]
    for api_norm, rid in ru_by_norm.items():
        if len(api_norm) >= 4 and (api_norm in n or n in api_norm):
            if n == "москва" and api_norm == "московская":
                continue
            if n == "московская" and api_norm == "москва":
                continue
            return rid
    return None


def _api_get(path: str, *, api_base: str = DEFAULT_API_BASE) -> dict | list | None:
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def _fetch_venue_detail(venue_id: int, *, cache: dict[int, dict]) -> dict | None:
    if venue_id in cache:
        return cache[venue_id]
    data = _api_get(f"venues/{venue_id}")
    if not isinstance(data, dict):
        return None
    cache[venue_id] = data
    time.sleep(0.15)
    return data


def _geocode_nominatim(
    town_name: str,
    region_name: str | None,
    country_name: str | None,
    *,
    cache: dict[str, tuple[float, float] | None],
) -> tuple[float, float] | None:
    key = f"{town_name}|{region_name}|{country_name}".lower()
    if key in cache:
        return cache[key]
    parts = [p for p in (town_name, region_name, country_name) if p]
    q = ", ".join(parts)
    url = (
        "https://nominatim.openstreetmap.org/search?"
        + urllib.parse.urlencode({"q": q, "format": "json", "limit": 5})
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "chgk-model-map/1 (contact: chgk.quest)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data:
            for hit in data:
                lat = float(hit["lat"])
                lon = float(hit["lon"])
                if _coords_match_country(country_name, lat, lon):
                    cache[key] = (lat, lon)
                    time.sleep(1.1)
                    return lat, lon
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, KeyError, ValueError):
        pass
    cache[key] = None
    time.sleep(1.1)
    return None


def _town_coords(
    town_name: str | None,
    region_name: str | None,
    country_name: str | None,
    *,
    nom_cache: dict[str, tuple[float, float] | None],
) -> tuple[float, float] | None:
    if not town_name:
        return None
    town_name = town_name.strip()
    manual = MANUAL_TOWN_COORDS.get((town_name, region_name, country_name))
    if manual:
        return manual
    # "Кировск (Л)"-style disambiguation suffixes confuse Nominatim.
    base = re.sub(r"\s*\([^)]*\)\s*$", "", town_name)
    attempts = [
        (town_name, region_name, country_name),
        (base, region_name, country_name),
        (base, None, country_name),
    ]
    seen: set[tuple] = set()
    for name, reg, cn in attempts:
        key = (name, reg, cn)
        if not name or key in seen:
            continue
        seen.add(key)
        coords = _geocode_nominatim(name, reg, cn, cache=nom_cache)
        if coords:
            return coords
    return None


def _apply_town_coords(geo: dict, town_coords: dict[int, tuple[float, float]]) -> None:
    towns_out: dict[str, dict] = {}
    for v in geo.get("venues", {}).values():
        tid = v.get("town_id")
        if tid is not None and tid in town_coords:
            lat, lon = town_coords[tid]
            v["lat"] = lat
            v["lon"] = lon
            towns_out[str(tid)] = {
                "town_id": tid,
                "town_name": v.get("town_name"),
                "region_id": v.get("region_id"),
                "region_name": v.get("region_name"),
                "country_id": v.get("country_id"),
                "country_name": v.get("country_name"),
                "lat": lat,
                "lon": lon,
            }
        elif tid is not None:
            v["lat"] = None
            v["lon"] = None
    geo["towns"] = towns_out


def regeocode_map_geo(*, invalid_only: bool = True) -> dict:
    """Re-geocode towns in existing map_geo.json (no venue API round-trip)."""
    geo = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    nom_cache: dict[str, tuple[float, float] | None] = {}
    towns: dict[int, dict] = {}
    for v in geo.get("venues", {}).values():
        tid = v.get("town_id")
        if tid is None:
            continue
        if tid not in towns:
            towns[tid] = {
                "town_name": v.get("town_name"),
                "region_name": v.get("region_name"),
                "country_name": v.get("country_name"),
                "lat": v.get("lat"),
                "lon": v.get("lon"),
            }

    town_coords: dict[int, tuple[float, float]] = {}
    fixed = 0
    for tid, meta in sorted(towns.items()):
        lat, lon = meta.get("lat"), meta.get("lon")
        cn = meta.get("country_name")
        needs_fix = lat is None or lon is None
        if not needs_fix and lat is not None and lon is not None:
            needs_fix = not _coords_match_country(cn, float(lat), float(lon))
        if invalid_only and not needs_fix:
            if lat is not None and lon is not None:
                town_coords[tid] = (float(lat), float(lon))
            continue
        coords = _town_coords(
            meta.get("town_name"),
            meta.get("region_name"),
            cn,
            nom_cache=nom_cache,
        )
        if coords:
            town_coords[tid] = coords
            fixed += 1
            print(f"  town {tid} {meta.get('town_name')!r} → {coords}", flush=True)
        elif lat is not None and lon is not None and _coords_match_country(
            cn, float(lat), float(lon)
        ):
            town_coords[tid] = (float(lat), float(lon))
        elif needs_fix:
            print(f"  town {tid} {meta.get('town_name')!r} — no coords", flush=True)

    _apply_town_coords(geo, town_coords)
    OUT_JSON.write_text(json.dumps(geo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Regeocoded {fixed} towns, wrote {OUT_JSON}")
    return geo


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())


def _circle_polygon(
    lat: float, lon: float, *, radius_km: float = 20.0, points: int = 40
) -> dict:
    """Rough city boundary when admin-1 GeoJSON has no separate polygon."""
    ring: list[list[float]] = []
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    for i in range(points):
        ang = 2.0 * math.pi * i / points
        dlat = (radius_km / 111.0) * math.sin(ang)
        dlon = (radius_km / (111.0 * cos_lat)) * math.cos(ang)
        ring.append([lon + dlon, lat + dlat])
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def _geometry_to_multipolygon_coords(geom: dict | None) -> list:
    if not geom:
        return []
    gtype = geom.get("type")
    if gtype == "Polygon":
        return [geom["coordinates"]]
    if gtype == "MultiPolygon":
        return list(geom["coordinates"])
    return []


def _coords_to_geometry(coords: list) -> dict:
    if len(coords) == 1:
        return {"type": "Polygon", "coordinates": coords[0]}
    return {"type": "MultiPolygon", "coordinates": coords}


def _merge_features_by_region_id(
    features: list[dict],
    regions: dict[int, dict],
) -> list[dict]:
    """One polygon per rating_region_id (e.g. Crimea + Sevastopol → Крым)."""
    unmatched: list[dict] = []
    by_rid: dict[int, list[dict]] = {}
    for f in features:
        rid = (f.get("properties") or {}).get("rating_region_id")
        if rid is None:
            unmatched.append(f)
            continue
        by_rid.setdefault(int(rid), []).append(f)

    merged = list(unmatched)
    for rid, group in by_rid.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        all_coords: list = []
        for f in group:
            all_coords.extend(_geometry_to_multipolygon_coords(f.get("geometry")))
        meta = regions.get(rid) or {}
        merged.append(
            {
                "type": "Feature",
                "properties": {
                    "region": meta.get("region_name"),
                    "rating_region_id": rid,
                    "merged_admin": True,
                },
                "geometry": _coords_to_geometry(all_coords),
            }
        )
    return merged


def _is_city_like_region_name(name: str | None) -> bool:
    n = (name or "").lower().replace("ё", "е")
    return not any(
        token in n
        for token in ("область", "край", "республика", "округ", " автоном")
    )


def _append_country_fallback_polygons(
    features: list[dict],
    iso_map: dict[str, str],
    venues: dict,
    towns: dict[str, dict],
) -> None:
    """Add circle polygons for microstates missing from NE 110m countries."""
    present = {
        str((f.get("properties") or {}).get("ISO_A2") or "").upper()
        for f in features
    }
    present.discard("")
    hardcoded: dict[str, tuple[float, float, str]] = {
        "AD": (42.5063, 1.5218, "Andorra"),
        "SG": (1.3521, 103.8198, "Singapore"),
        "MT": (35.9375, 14.3754, "Malta"),
    }
    for cid_str, iso in iso_map.items():
        iso = iso.upper()
        if iso in present:
            continue
        lat = lon = None
        name = None
        for v in venues.values():
            if str(v.get("country_id")) != cid_str:
                continue
            name = v.get("country_name") or name
            tid = v.get("town_id")
            if tid is not None and str(tid) in towns:
                t = towns[str(tid)]
                if t.get("lat") is not None and t.get("lon") is not None:
                    lat, lon = float(t["lat"]), float(t["lon"])
                    break
        if lat is None and iso in hardcoded:
            lat, lon, name = hardcoded[iso]
        if lat is None:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "ISO_A2": iso,
                    "NAME": name,
                    "country_fallback": True,
                },
                "geometry": _circle_polygon(lat, lon, radius_km=35.0),
            }
        )


def _append_city_fallback_polygons(
    features: list[dict],
    regions: dict[int, dict],
    towns: dict[str, dict],
    *,
    country_id: int,
) -> None:
    """Add circle polygons for city regions missing from the source boundaries."""
    matched = {
        int(f["properties"]["rating_region_id"])
        for f in features
        if f.get("properties", {}).get("rating_region_id") is not None
    }
    for rid, meta in regions.items():
        if int(meta.get("country_id") or 0) != country_id:
            continue
        if int(rid) in matched:
            continue
        if not _is_city_like_region_name(meta.get("region_name")):
            continue
        lat = lon = None
        for town in towns.values():
            if int(town.get("region_id") or -1) != int(rid):
                continue
            if town.get("lat") is not None and town.get("lon") is not None:
                lat, lon = float(town["lat"]), float(town["lon"])
                break
        if lat is None:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "region": meta.get("region_name"),
                    "rating_region_id": int(rid),
                    "city_fallback": True,
                },
                "geometry": _circle_polygon(lat, lon),
            }
        )


def _build_boundary_files(regions: dict[int, dict]) -> None:
    """Download and annotate GeoJSON boundaries with rating region_id."""
    GEO_DIR.mkdir(parents=True, exist_ok=True)

    # NE 50m admin-1 has broken Karelia/Murmansk polygons (nvkelso/natural-earth-vector#929).
    ne_admin1 = REPO_ROOT / "data" / "_ne_admin1_10m.geojson"
    ne_countries = REPO_ROOT / "data" / "_ne_countries.geojson"
    ua_src = REPO_ROOT / "data" / "_ua_regions.geojson"

    if not ne_admin1.exists():
        _download_url(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_10m_admin_1_states_provinces.geojson",
            ne_admin1,
        )
    if not ne_countries.exists():
        _download_url(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_110m_admin_0_countries.geojson",
            ne_countries,
        )
    if not ua_src.exists():
        _download_url(
            "https://raw.githubusercontent.com/slawomirmatuszak/ukrainian_geodata/"
            "main/regiony.geojson",
            ua_src,
        )

    ru_by_norm: dict[str, int] = {}
    ua_by_uk: dict[str, int] = {}
    for rid, meta in regions.items():
        cid = meta.get("country_id")
        name = meta.get("region_name") or ""
        if cid == 21:
            ru_by_norm[_norm_region(name)] = rid
        elif cid == 26:
            uk = UA_REGION_RU_TO_UK.get(name.lower()) or UA_REGION_RU_TO_UK.get(
                _norm_region(name)
            )
            if uk:
                ua_by_uk[uk.lower()] = rid

    geo_extra: dict = {}
    towns: dict[str, dict] = {}
    if OUT_JSON.exists():
        geo_extra = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        towns = geo_extra.get("towns") or {}

    admin1 = json.loads(ne_admin1.read_text(encoding="utf-8"))
    ru_features = []
    for f in admin1.get("features", []):
        p = f.get("properties") or {}
        if p.get("admin") != "Russia":
            continue
        local = (p.get("name_local") or p.get("name") or "").strip()
        rid = _ru_region_id_for_geo(
            local,
            ru_by_norm,
            en_name=p.get("name"),
            admin_type=p.get("type"),
        )
        if rid is not None:
            p = dict(p)
            p["rating_region_id"] = rid
        ru_features.append(
            {"type": "Feature", "properties": p, "geometry": f.get("geometry")}
        )
    ru_features = _merge_features_by_region_id(ru_features, regions)
    _append_city_fallback_polygons(ru_features, regions, towns, country_id=21)
    (GEO_DIR / "ru_regions.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": ru_features}, ensure_ascii=False),
        encoding="utf-8",
    )

    ua_data = json.loads(ua_src.read_text(encoding="utf-8"))
    ua_features = []
    for f in ua_data.get("features", []):
        p = dict(f.get("properties") or {})
        region_uk = (p.get("region") or "").strip()
        rid = ua_by_uk.get(region_uk.lower())
        if rid is not None:
            p["rating_region_id"] = rid
        ua_features.append(
            {"type": "Feature", "properties": p, "geometry": f.get("geometry")}
        )
    _append_city_fallback_polygons(ua_features, regions, towns, country_id=26)
    (GEO_DIR / "ua_regions.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": ua_features}, ensure_ascii=False),
        encoding="utf-8",
    )

    countries = json.loads(ne_countries.read_text(encoding="utf-8"))
    slim = []
    for f in countries.get("features", []):
        p = f.get("properties") or {}
        iso = (p.get("ISO_A2") or "").strip()
        if not iso or iso in ("-99", "-1"):
            iso = (p.get("ISO_A2_EH") or "").strip() or iso
        slim.append(
            {
                "type": "Feature",
                "properties": {
                    "ISO_A2": iso,
                    "NAME": p.get("NAME"),
                    "NAME_RU": p.get("NAME_RU"),
                },
                "geometry": f.get("geometry"),
            }
        )
    _append_country_fallback_polygons(
        slim,
        {str(k): str(v) for k, v in (geo_extra.get("country_iso") or {}).items()},
        geo_extra.get("venues") or {},
        towns,
    )
    (GEO_DIR / "countries.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": slim}, ensure_ascii=False),
        encoding="utf-8",
    )


def build_map_geo(
    venue_db: Path,
    *,
    api_base: str = DEFAULT_API_BASE,
    skip_nominatim: bool = False,
    limit_venues: int | None = None,
    incremental: bool = False,
) -> dict:
    con = duckdb.connect(str(venue_db), read_only=True)
    venue_ids = [
        int(r[0])
        for r in con.execute(
            """
            SELECT DISTINCT venue_id FROM venues
            WHERE NOT coalesce(is_online, false)
            ORDER BY venue_id
            """
        ).fetchall()
    ]
    if limit_venues:
        venue_ids = venue_ids[:limit_venues]

    venue_cache: dict[int, dict] = {}
    nom_cache: dict[str, tuple[float, float] | None] = {}
    venues_raw: list[dict] = []
    regions_meta: dict[int, dict] = {}
    prior_coords: dict[int, tuple[float, float]] = {}

    # Incremental: reuse existing venue rows / town coords, only hit the
    # API + Nominatim for venues and towns not yet in map_geo.json.
    if incremental and OUT_JSON.exists():
        prior = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        known: set[int] = set()
        for v in (prior.get("venues") or {}).values():
            vid = int(v["venue_id"])
            known.add(vid)
            venues_raw.append({k: v[k] for k in (
                "venue_id", "name", "town_id", "town_name", "region_id",
                "region_name", "country_id", "country_name", "is_online",
            )})
            rid = v.get("region_id")
            if rid is not None:
                regions_meta[int(rid)] = {
                    "region_id": int(rid),
                    "region_name": v.get("region_name"),
                    "country_id": v.get("country_id"),
                    "country_name": v.get("country_name"),
                }
        for t in (prior.get("towns") or {}).values():
            if t.get("lat") is not None:
                prior_coords[int(t["town_id"])] = (t["lat"], t["lon"])
        venue_ids = [vid for vid in venue_ids if vid not in known]
        print(f"  incremental: {len(venue_ids)} new venues", flush=True)

    for i, vid in enumerate(venue_ids):
        if i and i % 50 == 0:
            print(f"  venues {i}/{len(venue_ids)}…", flush=True)
        detail = _fetch_venue_detail(vid, cache=venue_cache)
        if not detail:
            row = con.execute(
                "SELECT name, is_online FROM venues WHERE venue_id = ?", [vid]
            ).fetchone()
            if not row:
                continue
            name, is_online = row
            venues_raw.append({
                "venue_id": vid,
                "name": name,
                "town_id": None,
                "town_name": name,
                "region_id": None,
                "region_name": None,
                "country_id": None,
                "country_name": None,
                "is_online": bool(is_online),
            })
            continue

        town = detail.get("town") or {}
        region = town.get("region") or {}
        country = town.get("country") or region.get("country") or {}
        town_id = town.get("id")
        town_name = (town.get("name") or "").strip() or None
        region_id = region.get("id")
        region_name = (region.get("name") or "").strip() or None
        country_id = country.get("id")
        country_name = (country.get("name") or "").strip() or None

        venue_row = {
            "venue_id": vid,
            "name": (detail.get("name") or "").strip(),
            "town_id": int(town_id) if town_id is not None else None,
            "town_name": town_name,
            "region_id": int(region_id) if region_id is not None else None,
            "region_name": region_name,
            "country_id": int(country_id) if country_id is not None else None,
            "country_name": country_name,
            "is_online": False,
        }
        _fill_orphan_region_country(venue_row)
        venues_raw.append(venue_row)

        if region_id is not None:
            region_row = {
                "region_id": int(region_id),
                "region_name": region_name,
                "country_id": venue_row.get("country_id"),
                "country_name": venue_row.get("country_name"),
            }
            regions_meta[int(region_id)] = region_row

    # Geocode each unique town once.
    town_coords: dict[int, tuple[float, float]] = dict(prior_coords)
    towns_out: dict[str, dict] = {}
    if not skip_nominatim:
        seen_towns: set[int] = set(prior_coords)
        for v in venues_raw:
            tid = v.get("town_id")
            if tid is None or tid in seen_towns:
                continue
            seen_towns.add(tid)
            coords = _town_coords(
                v.get("town_name"),
                v.get("region_name"),
                v.get("country_name"),
                nom_cache=nom_cache,
            )
            if coords:
                town_coords[tid] = coords
        print(f"  geocoded {len(town_coords)} towns", flush=True)

    venues_out: dict[str, dict] = {}
    for v in venues_raw:
        tid = v.get("town_id")
        coords = town_coords.get(tid) if tid is not None else None
        venues_out[str(v["venue_id"])] = {
            **v,
            "lat": coords[0] if coords else None,
            "lon": coords[1] if coords else None,
        }
        if tid is not None and coords and str(tid) not in towns_out:
            towns_out[str(tid)] = {
                "town_id": tid,
                "town_name": v.get("town_name"),
                "region_id": v.get("region_id"),
                "region_name": v.get("region_name"),
                "country_id": v.get("country_id"),
                "country_name": v.get("country_name"),
                "lat": coords[0],
                "lon": coords[1],
            }

    payload = {
        "detail_country_ids": sorted(DETAIL_COUNTRY_IDS),
        "country_iso": _build_country_iso(venues_out),
        "regions": {str(k): v for k, v in regions_meta.items()},
        "towns": towns_out,
        "venues": venues_out,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON} ({len(venues_out)} venues, {len(towns_out)} geocoded towns)")
    _build_boundary_files(regions_meta)
    print(f"Wrote GeoJSON to {GEO_DIR}")
    return payload


def rebuild_boundaries_only() -> None:
    """Rebuild ru/ua/countries GeoJSON from existing map_geo.json regions meta."""
    geo = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    regions_meta = {int(k): v for k, v in (geo.get("regions") or {}).items()}
    _build_boundary_files(regions_meta)
    print(f"Rebuilt GeoJSON in {GEO_DIR}")


def refresh_country_iso_only() -> None:
    """Patch country_iso in an existing map_geo.json (no API / Nominatim)."""
    geo = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    venues = geo.get("venues") or {}
    geo["country_iso"] = _build_country_iso(venues)
    OUT_JSON.write_text(json.dumps(geo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Patched country_iso in {OUT_JSON} ({len(geo['country_iso'])} countries)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build map geography cache")
    ap.add_argument("--venue-db", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument(
        "--skip-nominatim",
        action="store_true",
        help="Skip geocoding (venues without coordinates)",
    )
    ap.add_argument("--limit-venues", type=int, default=None)
    ap.add_argument(
        "--refresh-country-iso-only",
        action="store_true",
        help="Only rebuild country_iso in existing map_geo.json",
    )
    ap.add_argument(
        "--rebuild-boundaries-only",
        action="store_true",
        help="Only rebuild GeoJSON boundaries from existing map_geo.json",
    )
    ap.add_argument(
        "--regeocode-only",
        action="store_true",
        help="Re-geocode invalid towns in existing map_geo.json",
    )
    ap.add_argument(
        "--regeocode-all",
        action="store_true",
        help="Re-geocode every town (slow)",
    )
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="Only fetch/geocode venues missing from existing map_geo.json",
    )
    args = ap.parse_args()
    if args.refresh_country_iso_only:
        if not OUT_JSON.exists():
            raise SystemExit(f"missing {OUT_JSON}")
        refresh_country_iso_only()
        return 0
    if args.rebuild_boundaries_only:
        if not OUT_JSON.exists():
            raise SystemExit(f"missing {OUT_JSON}")
        rebuild_boundaries_only()
        return 0
    if args.regeocode_only or args.regeocode_all:
        if not OUT_JSON.exists():
            raise SystemExit(f"missing {OUT_JSON}")
        regeocode_map_geo(invalid_only=not args.regeocode_all)
        return 0
    if not args.venue_db.exists():
        raise SystemExit(f"venue overlay missing: {args.venue_db}")
    build_map_geo(
        args.venue_db,
        api_base=args.api_base,
        skip_nominatim=args.skip_nominatim,
        limit_venues=args.limit_venues,
        incremental=args.incremental,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
