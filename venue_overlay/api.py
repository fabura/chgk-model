"""HTTP client for api.rating.chgk.info tournament results."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_API_BASE = "https://api.rating.chgk.info"
DEFAULT_USER_AGENT = "chgk-model-venue-overlay/1"
DEFAULT_TIMEOUT_SEC = 60
DEFAULT_SLEEP_SEC = 0.25
MAX_RETRIES = 5


class RatingApiError(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def fetch_tournament_results(
    tournament_id: int,
    *,
    api_base: str = DEFAULT_API_BASE,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[int, list[dict[str, Any]]]:
    """Return (http_status, parsed JSON list) for GET /tournaments/{id}/results."""
    url = f"{api_base.rstrip('/')}/tournaments/{int(tournament_id)}/results"
    last_err: Exception | None = None

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            time.sleep(sleep_sec * (2**attempt))
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": user_agent,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                status = int(resp.status)
                body = resp.read().decode("utf-8", errors="replace")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            if status != 200:
                raise RatingApiError(f"HTTP {status} for {url}", status=status)
            data = json.loads(body)
            if not isinstance(data, list):
                raise RatingApiError(
                    f"Expected JSON array for tournament {tournament_id}, got {type(data).__name__}"
                )
            return status, data
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                continue
            raise RatingApiError(f"HTTP {e.code} for {url}", status=e.code) from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            continue

    raise RatingApiError(
        f"Failed after {MAX_RETRIES} attempts for tournament {tournament_id}: {last_err}"
    ) from last_err


def fetch_synch_requests_for_tournament(
    tournament_id: int,
    *,
    api_base: str = DEFAULT_API_BASE,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[int, int]:
    """Map venue_id -> approximateTeamsCount from synch requests (best effort)."""
    url = f"{api_base.rstrip('/')}/synch_tournaments/{int(tournament_id)}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": user_agent},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError:
        return {}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}

    if sleep_sec > 0:
        time.sleep(sleep_sec)

    request_iris = obj.get("requests") or []
    out: dict[int, int] = {}
    for iri in request_iris:
        if not isinstance(iri, str) or not iri.startswith("/tournament_synch_requests/"):
            continue
        detail_url = f"{api_base.rstrip('/')}{iri}"
        dreq = urllib.request.Request(
            detail_url,
            headers={"Accept": "application/json", "User-Agent": user_agent},
        )
        try:
            with urllib.request.urlopen(dreq, timeout=timeout_sec) as resp:
                detail = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            continue
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        approx = detail.get("approximateTeamsCount")
        venue = detail.get("venue") or {}
        vid = venue.get("id")
        if vid is not None and approx is not None:
            vid_i = int(vid)
            out[vid_i] = max(out.get(vid_i, 0), int(approx))
    return out
