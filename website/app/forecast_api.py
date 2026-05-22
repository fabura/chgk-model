"""
Lightweight read-only client for the public ``api.rating.chgk.info`` API.

Only the endpoints needed by the /forecast pages are wrapped:

* ``GET /tournaments?dateStart[after]=…&dateStart[before]=…`` — upcoming
  tournament list (paged through with ``itemsPerPage``).
* ``GET /tournaments/{id}`` — tournament metadata (title, type, dates,
  ``questionQty``).
* ``GET /tournaments/{id}/results?includeTeamMembers=1`` — pre-announced
  team rosters (works for *future* tournaments once orgs have collected
  the registrations; ``position == 9999`` is the API sentinel for "not
  played yet").

Everything is fetched via ``urllib.request`` (stdlib only — no extra
runtime deps) and memoised with a small TTL cache shared across
requests.  Failures fall back to a stale entry when one is available so
a flaky CDN does not white-screen the page.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

API_BASE = "https://api.rating.chgk.info"
_USER_AGENT = "chgk-model-forecast/1.0 (+https://github.com)"
_HTTP_TIMEOUT_SEC = 5.0

# Per-endpoint cache TTLs.  Upcoming tournaments and rosters update over
# the course of a day (orgs add teams; new sync requests appear) so a
# 10-minute window is the right balance between freshness and respect
# for the upstream API.
_TTL_UPCOMING_SEC = 600
_TTL_ROSTERS_SEC = 600
_TTL_META_SEC = 3600


_CacheEntry = tuple[float, Any]  # (expires_at, value)
_cache: dict[str, _CacheEntry] = {}
_cache_lock = threading.Lock()


def _http_get_json(url: str) -> Any:
    """Fetch ``url`` and return parsed JSON.  Raises on transport errors."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
        raw = resp.read()
    # The API claims utf-8; decode tolerantly so a stray control char
    # (we have seen them in payloads) does not blow up json.loads.
    text = raw.decode("utf-8", errors="replace")
    return json.loads(text)


def _cached(key: str, ttl_sec: float, fetch):
    """Return cached value for ``key``; on miss/expiry, call ``fetch``.

    On fetch error keep serving the stale entry if we have one (a flaky
    upstream should not break the user-facing page).
    """
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    try:
        value = fetch()
    except Exception:
        if hit is not None:
            return hit[1]
        raise
    with _cache_lock:
        _cache[key] = (now + ttl_sec, value)
    return value


def clear_cache() -> None:
    """Drop every cached API response (useful from /admin/reload-db)."""
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Type / mode mapping
# ---------------------------------------------------------------------------
#
# The API exposes ``type.id`` and ``type.name``; the website internally
# uses three short labels.  ``type.name`` is the source of truth (we have
# seen the same id reused with different names historically).

_TYPE_NAME_TO_MODE = {
    "Обычный": "offline",
    "Чемпионат": "offline",
    "Синхрон": "sync",
    "Асинхрон": "async",
}


def api_type_to_mode(type_obj: Optional[dict]) -> str:
    """Best-effort mapping from API ``type`` payload to ``offline|sync|async``.

    Defaults to ``offline`` for unknown values — that is the safest fallback
    for the calibration kernel (lapse/recal are smallest in offline mode,
    so an unknown event won't get a phantom calibration boost).
    """
    if not isinstance(type_obj, dict):
        return "offline"
    name = (type_obj.get("name") or "").strip()
    return _TYPE_NAME_TO_MODE.get(name, "offline")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def list_upcoming(
    *,
    after_iso: str,
    before_iso: Optional[str] = None,
    items_per_page: int = 50,
    page: int = 1,
) -> list[dict]:
    """List tournaments starting in ``[after_iso, before_iso)``.

    The API returns the items *in start order* by default, which is what
    we want for an "upcoming" listing.
    """
    params: list[tuple[str, str]] = [
        ("dateStart[after]", after_iso),
        ("itemsPerPage", str(int(items_per_page))),
        ("page", str(int(page))),
    ]
    if before_iso is not None:
        params.append(("dateStart[before]", before_iso))
    qs = urllib.parse.urlencode(params)
    url = f"{API_BASE}/tournaments?{qs}"
    return _cached(
        f"upcoming:{qs}", _TTL_UPCOMING_SEC, lambda: _http_get_json(url)
    )


def get_tournament(tournament_id: int) -> dict:
    """Return tournament metadata for ``tournament_id``."""
    url = f"{API_BASE}/tournaments/{int(tournament_id)}"
    return _cached(
        f"meta:{tournament_id}", _TTL_META_SEC, lambda: _http_get_json(url)
    )


def get_rosters(tournament_id: int) -> list[dict]:
    """Return per-team rosters for ``tournament_id`` (may be empty).

    Each item has at least ``team`` (``id``, ``name``, ``town``),
    ``teamMembers`` (list of ``{flag, rating, player: {id, name, ...}}``),
    ``position`` (``9999`` for not-yet-played) and optionally ``mask``
    (post-tournament only).
    """
    url = (
        f"{API_BASE}/tournaments/{int(tournament_id)}/results"
        f"?includeTeamMembers=1&itemsPerPage=200"
    )
    return _cached(
        f"rosters:{tournament_id}",
        _TTL_ROSTERS_SEC,
        lambda: _http_get_json(url),
    )
