"""HTTP client for api.rating.chgk.info — the subset we need to mirror
into the local rating Postgres.

Stdlib-only (no requests dependency, like venue_overlay/api.py).

NOTE: venue_overlay/api.py has its own tiny client for the same host.
Kept separate for now to avoid a cross-module dependency before the
rating-api package is proven; consolidating them is in the refactor
backlog.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator

DEFAULT_API_BASE = "https://api.rating.chgk.info"
DEFAULT_USER_AGENT = "chgk-model-rating-api/1"
DEFAULT_TIMEOUT_SEC = 60.0
DEFAULT_SLEEP_SEC = 0.25
MAX_RETRIES = 5
RETRY_STATUSES = {429, 500, 502, 503, 504}


class RatingApiError(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RatingApiClient:
    """Thin wrapper around urllib with retry/throttle.  All methods
    return parsed JSON (already validated to be the expected shape)."""

    def __init__(
        self,
        *,
        api_base: str = DEFAULT_API_BASE,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        sleep_sec: float = DEFAULT_SLEEP_SEC,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.user_agent = user_agent
        self.timeout_sec = timeout_sec
        self.sleep_sec = sleep_sec

    # ------------------------------------------------------------------
    # low-level
    # ------------------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        q = ("?" + urllib.parse.urlencode(params, doseq=True)) if params else ""
        url = f"{self.api_base}{path}{q}"
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            if attempt > 0:
                time.sleep(self.sleep_sec * (2 ** attempt))
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": self.user_agent},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    status = int(resp.status)
                    body = resp.read().decode("utf-8", errors="replace")
                if self.sleep_sec > 0:
                    time.sleep(self.sleep_sec)
                if status != 200:
                    raise RatingApiError(f"HTTP {status} for {url}", status=status)
                return json.loads(body)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in RETRY_STATUSES:
                    continue
                raise RatingApiError(f"HTTP {e.code} for {url}", status=e.code) from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                continue
        raise RatingApiError(
            f"Failed after {MAX_RETRIES} attempts for {url}: {last_err}"
        ) from last_err

    # ------------------------------------------------------------------
    # high-level
    # ------------------------------------------------------------------
    def get_tournament(self, tournament_id: int) -> dict[str, Any]:
        obj = self._get(f"/tournaments/{int(tournament_id)}")
        if not isinstance(obj, dict):
            raise RatingApiError(
                f"Expected dict for tournament {tournament_id}, got {type(obj).__name__}"
            )
        return obj

    def get_results(
        self,
        tournament_id: int,
        *,
        include_team_members: bool = True,
        include_masks: bool = True,
    ) -> list[dict[str, Any]]:
        params = {
            "includeTeamMembers": 1 if include_team_members else 0,
            "includeMasksAndControversials": 1 if include_masks else 0,
        }
        obj = self._get(f"/tournaments/{int(tournament_id)}/results", params)
        # The endpoint returns either a bare JSON list (`application/json`)
        # or a Hydra collection (`application/ld+json`). We always set
        # `Accept: application/json`, so a list is expected; tolerate both.
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            members = obj.get("member") or obj.get("hydra:member") or []
            if isinstance(members, list):
                return members
        raise RatingApiError(
            f"Unexpected /results shape for tournament {tournament_id}: {type(obj).__name__}"
        )

    def iter_tournaments_changed_since(
        self,
        since_iso_date: str,
        *,
        items_per_page: int = 512,
    ) -> Iterator[dict[str, Any]]:
        """Yield Tournament-summary blobs with lastEditDate >= since_iso_date.

        `since_iso_date` is a YYYY-MM-DD string (API accepts date or
        datetime).  Pages are walked in lastEditDate ascending order so
        the cursor can be advanced after each successful upsert.
        """
        page = 1
        while True:
            data = self._get(
                "/tournaments",
                {
                    "lastEditDate[after]": since_iso_date,
                    "order[lastEditDate]": "asc",
                    "itemsPerPage": items_per_page,
                    "page": page,
                },
            )
            items: list[dict[str, Any]] = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("member") or data.get("hydra:member") or []
            if not items:
                return
            for it in items:
                if isinstance(it, dict):
                    yield it
            if len(items) < items_per_page:
                return
            page += 1
