#!/usr/bin/env python3
"""Validate map geography matching before publish.

Checks that countries/regions referenced in map_geo.json have ISO codes and
GeoJSON polygons.  Exit code 0 = OK, 1 = issues found.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GEO_DIR = REPO_ROOT / "website" / "app" / "static" / "geo"
MAP_GEO = REPO_ROOT / "website" / "data" / "map_geo.json"

DETAIL_COUNTRY_IDS = {21, 26}  # Россия, Украина


def _load_geojson(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _region_ids_in_geo(path: Path) -> set[int]:
    data = _load_geojson(path)
    out: set[int] = set()
    for f in data.get("features", []):
        rid = (f.get("properties") or {}).get("rating_region_id")
        if rid is not None:
            out.add(int(rid))
    return out


def _country_isos_in_geo(path: Path) -> set[str]:
    data = _load_geojson(path)
    out: set[str] = set()
    for f in data.get("features", []):
        p = f.get("properties") or {}
        iso = p.get("ISO_A2") or p.get("ISO_A2_EH")
        if iso and iso != "-99":
            out.add(str(iso).upper())
    return out


def _regions_used(geo: dict) -> dict[int, dict]:
    """Regions that appear in venues or towns."""
    regions_meta = {int(k): v for k, v in (geo.get("regions") or {}).items()}
    used: set[int] = set()
    for v in (geo.get("venues") or {}).values():
        rid = v.get("region_id")
        if rid is not None:
            used.add(int(rid))
    for t in (geo.get("towns") or {}).values():
        rid = t.get("region_id")
        if rid is not None:
            used.add(int(rid))
    return {rid: regions_meta[rid] for rid in used if rid in regions_meta}


def _countries_used(geo: dict) -> dict[int, str]:
    """country_id → country_name from venues."""
    out: dict[int, str] = {}
    for v in (geo.get("venues") or {}).values():
        cid = v.get("country_id")
        if cid is None:
            continue
        out[int(cid)] = v.get("country_name") or out.get(int(cid), "")
    return out


def validate(*, strict: bool = True) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not MAP_GEO.exists():
        print(f"ERROR: missing {MAP_GEO}", file=sys.stderr)
        return 1

    geo = json.loads(MAP_GEO.read_text(encoding="utf-8"))
    iso_map: dict[str, str] = {
        str(k): str(v) for k, v in (geo.get("country_iso") or {}).items()
    }
    used_regions = _regions_used(geo)
    used_countries = _countries_used(geo)

    # --- countries ---
    country_geo_path = GEO_DIR / "countries.geojson"
    if not country_geo_path.exists():
        errors.append(f"missing {country_geo_path}")
        geo_isos: set[str] = set()
    else:
        geo_isos = _country_isos_in_geo(country_geo_path)

    for cid, cname in sorted(used_countries.items()):
        if cid in DETAIL_COUNTRY_IDS:
            continue
        iso = iso_map.get(str(cid))
        if not iso:
            errors.append(f"country {cid} ({cname}): no ISO in country_iso")
            continue
        if iso not in geo_isos:
            errors.append(
                f"country {cid} ({cname}): ISO {iso} not in countries.geojson"
            )

    # ISO in map but not in geojson (e.g. Singapore)
    for cid, iso in sorted(iso_map.items(), key=lambda x: int(x[0])):
        if int(cid) in DETAIL_COUNTRY_IDS:
            continue
        if iso and iso not in geo_isos and str(cid) in used_countries:
            errors.append(
                f"country {cid}: ISO {iso} in country_iso but not in countries.geojson"
            )

    # --- RU / UA regions ---
    ru_geo_ids = _region_ids_in_geo(GEO_DIR / "ru_regions.geojson") if (
        GEO_DIR / "ru_regions.geojson"
    ).exists() else set()
    ua_geo_ids = _region_ids_in_geo(GEO_DIR / "ua_regions.geojson") if (
        GEO_DIR / "ua_regions.geojson"
    ).exists() else set()

    by_country: dict[int, list[int]] = defaultdict(list)
    for rid, meta in used_regions.items():
        cid = int(meta.get("country_id") or 0)
        by_country[cid].append(rid)

    for rid in sorted(by_country.get(21, [])):
        if rid not in ru_geo_ids:
            name = used_regions[rid].get("region_name", "?")
            errors.append(f"RU region {rid} ({name}): no polygon in ru_regions.geojson")

    for rid in sorted(by_country.get(26, [])):
        if rid not in ua_geo_ids:
            name = used_regions[rid].get("region_name", "?")
            errors.append(f"UA region {rid} ({name}): no polygon in ua_regions.geojson")

    # Duplicate polygon assignments
    for label, path in [("RU", GEO_DIR / "ru_regions.geojson"), ("UA", GEO_DIR / "ua_regions.geojson")]:
        if not path.exists():
            continue
        data = _load_geojson(path)
        counts: Counter[int] = Counter()
        for f in data.get("features", []):
            rid = (f.get("properties") or {}).get("rating_region_id")
            if rid is not None:
                counts[int(rid)] += 1
        for rid, n in sorted(counts.items()):
            if n > 1:
                warnings.append(f"{label} region {rid}: {n} polygons (possible overlap)")

    # NE features without rating_region_id (informational)
    ru_data = _load_geojson(GEO_DIR / "ru_regions.geojson") if (
        GEO_DIR / "ru_regions.geojson"
    ).exists() else {"features": []}
    unmatched_ne = [
        (f.get("properties") or {}).get("name")
        for f in ru_data.get("features", [])
        if (f.get("properties") or {}).get("rating_region_id") is None
    ]
    if unmatched_ne and not strict:
        warnings.append(
            f"{len(unmatched_ne)} NE Russia features without rating_region_id "
            f"(e.g. {unmatched_ne[:3]})"
        )

    # --- report ---
    print(f"Used countries (non-detail): {len([c for c in used_countries if c not in DETAIL_COUNTRY_IDS])}")
    print(f"Used RU regions: {len(by_country.get(21, []))} / {len(ru_geo_ids)} polygons")
    print(f"Used UA regions: {len(by_country.get(26, []))} / {len(ua_geo_ids)} polygons")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("\nOK — all used countries/regions are matched.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate map geo matching")
    ap.add_argument(
        "--lenient",
        action="store_true",
        help="Do not treat unmatched NE features as issues",
    )
    args = ap.parse_args()
    return validate(strict=not args.lenient)


if __name__ == "__main__":
    raise SystemExit(main())
