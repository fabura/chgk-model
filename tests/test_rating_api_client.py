"""Unit tests for ``rating_api.client``.

We don't actually talk to api.rating.chgk.info — every test patches
``urllib.request.urlopen`` (the only outbound point) and asserts on
the URLs requested plus the parsed result.
"""
from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from rating_api.client import RatingApiClient, RatingApiError


class _FakeResponse:
    """Minimal urlopen-context-manager stand-in."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _json_response(payload, status: int = 200) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload).encode("utf-8"), status=status)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x", code=code, msg="x", hdrs=None, fp=io.BytesIO(b"")
    )


class _UrlopenSpy:
    """Records every urlopen call (the request URL) and yields the next
    pre-canned response, raising any pre-canned exception."""

    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, req, timeout=None):
        # `req` may be a Request object — pull the URL out for asserts.
        url = req.full_url if hasattr(req, "full_url") else str(req)
        self.calls.append(url)
        if not self.responses:
            raise AssertionError("urlopen called more times than expected")
        out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


# ----------------------------------------------------------------------
# low-level GET
# ----------------------------------------------------------------------


class ClientGetTests(unittest.TestCase):
    def setUp(self) -> None:
        # zero sleep so retries don't slow the suite down
        self.c = RatingApiClient(sleep_sec=0.0)

    def test_get_tournament_returns_parsed_json(self):
        spy = _UrlopenSpy([_json_response({"id": 42, "name": "X"})])
        with patch("urllib.request.urlopen", spy):
            obj = self.c.get_tournament(42)
        self.assertEqual(obj, {"id": 42, "name": "X"})
        self.assertEqual(len(spy.calls), 1)
        self.assertTrue(spy.calls[0].endswith("/tournaments/42"))

    def test_non_dict_tournament_raises(self):
        # /tournaments/{id} should be a dict; anything else is a server
        # bug we want to surface, not silently swallow.
        spy = _UrlopenSpy([_json_response([1, 2, 3])])
        with patch("urllib.request.urlopen", spy):
            with self.assertRaises(RatingApiError):
                self.c.get_tournament(1)

    def test_retries_on_retryable_status_then_succeeds(self):
        # 503 → retry → 200 should return the final body, not raise.
        spy = _UrlopenSpy([
            _http_error(503),
            _http_error(429),
            _json_response({"id": 1}),
        ])
        with patch("urllib.request.urlopen", spy):
            obj = self.c.get_tournament(1)
        self.assertEqual(obj, {"id": 1})
        self.assertEqual(len(spy.calls), 3)

    def test_404_does_not_retry(self):
        # 404 is a real "this id doesn't exist" answer; retrying just
        # wastes API budget.  Must surface immediately.
        spy = _UrlopenSpy([_http_error(404)])
        with patch("urllib.request.urlopen", spy):
            with self.assertRaises(RatingApiError) as cm:
                self.c.get_tournament(99999999)
        self.assertEqual(cm.exception.status, 404)
        self.assertEqual(len(spy.calls), 1)

    def test_gives_up_after_max_retries(self):
        # Pre-canned ones returned 5 retryable failures — after the
        # configured max attempts we must surface a RatingApiError, not
        # loop forever or swallow.
        spy = _UrlopenSpy([_http_error(503)] * 10)
        with patch("urllib.request.urlopen", spy):
            with self.assertRaises(RatingApiError):
                self.c.get_tournament(1)
        # 5 attempts (the module constant MAX_RETRIES); stays bounded.
        self.assertEqual(len(spy.calls), 5)


# ----------------------------------------------------------------------
# get_results
# ----------------------------------------------------------------------


class GetResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.c = RatingApiClient(sleep_sec=0.0)

    def test_accepts_plain_list_response(self):
        spy = _UrlopenSpy([_json_response([{"team": {"id": 1}}])])
        with patch("urllib.request.urlopen", spy):
            rows = self.c.get_results(100)
        self.assertEqual(rows, [{"team": {"id": 1}}])

    def test_accepts_hydra_collection_response(self):
        # If the server upgrades Accept negotiation and starts returning
        # `{"member": [...]}`, we must keep working.
        spy = _UrlopenSpy([_json_response({"member": [{"team": {"id": 2}}]})])
        with patch("urllib.request.urlopen", spy):
            rows = self.c.get_results(100)
        self.assertEqual(rows, [{"team": {"id": 2}}])

    def test_passes_include_flags_to_url(self):
        # The whole point of the wrapper is that callers don't have to
        # remember the magic query params; lock them in.
        spy = _UrlopenSpy([_json_response([])])
        with patch("urllib.request.urlopen", spy):
            self.c.get_results(100, include_team_members=True, include_masks=True)
        q = parse_qs(urlsplit(spy.calls[0]).query)
        self.assertEqual(q.get("includeTeamMembers"), ["1"])
        self.assertEqual(q.get("includeMasksAndControversials"), ["1"])

    def test_unexpected_shape_raises(self):
        spy = _UrlopenSpy([_json_response(42)])  # neither list nor dict
        with patch("urllib.request.urlopen", spy):
            with self.assertRaises(RatingApiError):
                self.c.get_results(100)


# ----------------------------------------------------------------------
# iter_tournaments_changed_since (pagination + cursor semantics)
# ----------------------------------------------------------------------


class IterChangedSinceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.c = RatingApiClient(sleep_sec=0.0)

    def test_uses_strictly_after_filter(self):
        # Tight regression: switching to `[after]` would break
        # idempotency on a day-edge cursor.  Lock the URL param.
        spy = _UrlopenSpy([_json_response([])])
        with patch("urllib.request.urlopen", spy):
            list(self.c.iter_tournaments_changed_since("2026-04-21T11:27:14"))
        q = parse_qs(urlsplit(spy.calls[0]).query)
        self.assertIn("lastEditDate[strictly_after]", q)
        self.assertNotIn("lastEditDate[after]", q)
        self.assertEqual(q["lastEditDate[strictly_after]"], ["2026-04-21T11:27:14"])
        self.assertEqual(q["order[lastEditDate]"], ["asc"])

    def test_stops_on_empty_page(self):
        # Single empty page → terminate, no second request.
        spy = _UrlopenSpy([_json_response([])])
        with patch("urllib.request.urlopen", spy):
            items = list(self.c.iter_tournaments_changed_since("2026-01-01"))
        self.assertEqual(items, [])
        self.assertEqual(len(spy.calls), 1)

    def test_stops_on_partial_page(self):
        # If a page returns fewer items than `itemsPerPage`, that means
        # we've drained the result set — don't burn another request.
        spy = _UrlopenSpy([_json_response([{"id": 1}, {"id": 2}])])
        with patch("urllib.request.urlopen", spy):
            items = list(self.c.iter_tournaments_changed_since(
                "2026-01-01", items_per_page=10
            ))
        self.assertEqual([i["id"] for i in items], [1, 2])
        self.assertEqual(len(spy.calls), 1)

    def test_walks_multiple_full_pages(self):
        # Two full pages followed by an empty one — yields all 4 items.
        page1 = _json_response([{"id": 1}, {"id": 2}])
        page2 = _json_response([{"id": 3}, {"id": 4}])
        page3 = _json_response([])
        spy = _UrlopenSpy([page1, page2, page3])
        with patch("urllib.request.urlopen", spy):
            items = list(self.c.iter_tournaments_changed_since(
                "2026-01-01", items_per_page=2
            ))
        self.assertEqual([i["id"] for i in items], [1, 2, 3, 4])
        # Three pages were requested with page=1, page=2, page=3.
        pages = [parse_qs(urlsplit(u).query)["page"][0] for u in spy.calls]
        self.assertEqual(pages, ["1", "2", "3"])

    def test_drops_non_dict_items(self):
        # Defensive: a stray non-dict in the page must not break iteration.
        spy = _UrlopenSpy([_json_response([{"id": 1}, "weird", {"id": 2}])])
        with patch("urllib.request.urlopen", spy):
            items = list(self.c.iter_tournaments_changed_since(
                "2026-01-01", items_per_page=10
            ))
        self.assertEqual([i["id"] for i in items], [1, 2])


if __name__ == "__main__":
    unittest.main()
