"""Probe api.rating.chgk.info to nail down the unknowns before wiring it
into the main pipeline (rating dump → local PG → load_from_db).

Read-only. Hits 4 endpoints + prints structured findings:

1. /tournaments?type=2|3|6|8&itemsPerPage=3   → learn the type-code mapping
2. /tournaments/{id}                          → one full Tournament-read blob
3. /tournaments/{id}/results?includeTeamMembers=1&includeMasksAndControversials=1
                                              → confirm rosters + masks are present
4. /tournaments?lastEditDate[after]=YYYY-MM-DD&itemsPerPage=5
                                              → confirm incremental discovery

After running, we will know enough to spec the upsert layer (F.2).
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta

API_BASE = "https://api.rating.chgk.info"
UA = "chgk-model-probe/1"
TIMEOUT = 30


def get(path: str, params: dict | None = None) -> object:
    q = ("?" + urllib.parse.urlencode(params, doseq=True)) if params else ""
    url = f"{API_BASE}{path}{q}"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def probe_type_mapping() -> dict[int, str]:
    """Hit /tournaments?type=N for each N and grab the first tournament name."""
    section("1. tournament.type integer → text mapping")
    mapping: dict[int, str] = {}
    for code in (2, 3, 6, 8):
        try:
            data = get("/tournaments", {"type": code, "itemsPerPage": 3, "order[id]": "desc"})
        except Exception as e:  # noqa: BLE001
            print(f"  type={code}: ERROR {e}")
            continue
        items = data if isinstance(data, list) else data.get("member") or data.get("hydra:member") or []
        print(f"  type={code}: {len(items)} items returned")
        for it in items[:3]:
            print(
                f"     id={it.get('id'):>6}  "
                f"name={(it.get('name') or '')[:60]!r:<62}  "
                f"dateStart={it.get('dateStart','')[:10]}"
            )
        if items:
            mapping[code] = items[0].get("name", "")
    return mapping


def probe_single_tournament(tid: int) -> dict:
    section(f"2. /tournaments/{tid}  (Tournament-read blob)")
    obj = get(f"/tournaments/{tid}")
    keys = sorted(obj.keys()) if isinstance(obj, dict) else []
    print(f"  top-level keys ({len(keys)}): {keys}")
    for k in ("id", "name", "type", "dateStart", "dateEnd", "lastEditDate", "questionQty", "trueDL"):
        if k in obj:
            print(f"    {k:>15} = {obj[k]!r}")
    eds = obj.get("editors") or []
    print(f"    editors        = {len(eds)} entries; first: {eds[0] if eds else None}")
    return obj if isinstance(obj, dict) else {}


def probe_results(tid: int) -> None:
    section(f"3. /tournaments/{tid}/results  (with teamMembers + masks)")
    rows = get(
        f"/tournaments/{tid}/results",
        {"includeTeamMembers": 1, "includeMasksAndControversials": 1},
    )
    if not isinstance(rows, list):
        rows = rows.get("member") or rows.get("hydra:member") or []
    print(f"  {len(rows)} result rows")
    if not rows:
        return

    n_with_mask = sum(1 for r in rows if r.get("mask") is not None)
    n_with_members = sum(1 for r in rows if r.get("teamMembers"))
    n_with_position = sum(1 for r in rows if r.get("position") is not None)
    print(f"  with mask:        {n_with_mask}/{len(rows)}")
    print(f"  with teamMembers: {n_with_members}/{len(rows)}")
    print(f"  with position:    {n_with_position}/{len(rows)}")

    r0 = rows[0]
    print(f"  first row keys: {sorted(r0.keys())}")
    team = r0.get("team") or {}
    print(f"    team.id={team.get('id')} team.name={team.get('name')!r}")
    print(f"    position={r0.get('position')} questionsTotal={r0.get('questionsTotal')}")
    mask = r0.get("mask") or ""
    print(f"    mask[:40] = {mask[:40]!r}  (len={len(mask)})")
    tm = r0.get("teamMembers") or []
    print(f"    teamMembers: {len(tm)}")
    for m in tm[:3]:
        p = m.get("player") or {}
        print(f"      flag={m.get('flag')!r} player.id={p.get('id')} {p.get('surname')} {p.get('name')}")


def probe_discovery() -> None:
    section("4. /tournaments?lastEditDate[after]=...  (incremental discovery)")
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    print(f"  cutoff: lastEditDate >= {cutoff}")
    data = get(
        "/tournaments",
        {
            "lastEditDate[after]": cutoff,
            "order[lastEditDate]": "asc",
            "itemsPerPage": 5,
            "page": 1,
        },
    )
    items = data if isinstance(data, list) else data.get("member") or data.get("hydra:member") or []
    total = None
    if isinstance(data, dict):
        total = data.get("totalItems") or data.get("hydra:totalItems")
    print(f"  page-1 returned {len(items)} items; totalItems={total}")
    for it in items[:5]:
        print(
            f"    id={it.get('id'):>6}  "
            f"lastEditDate={it.get('lastEditDate','')[:19]}  "
            f"dateEnd={(it.get('dateEnd') or '')[:10]}  "
            f"name={(it.get('name') or '')[:55]!r}"
        )


def main() -> int:
    # Pick 3 well-known tournament ids: an offline (Чемпионат России),
    # a synchron, an async.  Override with `--tid N`.
    sample_tids = [9930, 9893, 9952]
    if len(sys.argv) > 1 and sys.argv[1] == "--tid" and len(sys.argv) > 2:
        sample_tids = [int(sys.argv[2])]

    def _safe(label, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except urllib.error.HTTPError as e:
            print(f"  [{label}] HTTP {e.code} {e.reason} — skipping", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  [{label}] failed: {type(e).__name__}: {e}", file=sys.stderr)

    _safe("type-mapping", probe_type_mapping)
    for tid in sample_tids:
        _safe(f"tournament-{tid}", probe_single_tournament, tid)
        _safe(f"results-{tid}", probe_results, tid)
    _safe("discovery", probe_discovery)
    return 0


if __name__ == "__main__":
    sys.exit(main())
