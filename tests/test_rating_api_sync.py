"""Tests for ``rating_api.sync.run_sync`` — the orchestrator.

We stub out every external dependency:
  - the HTTP client (no network),
  - the PG connection + ``upsert_bundle`` + state helpers (no DB).

This isolates the test to the orchestrator's own logic: discovery
walking, per-tournament error containment, the dry-run gate, and the
state-recording contract.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from rating_api.client import RatingApiError
from rating_api.sync import run_sync


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _summary(tid: int, last_edit: str) -> dict:
    return {"id": tid, "lastEditDate": last_edit}


def _meta(tid: int, last_edit: str, *, qcount_total: int = 36) -> dict:
    return {
        "id": tid,
        "name": f"t-{tid}",
        "type": {"id": 3, "name": "Синхрон"},
        "dateStart": "2026-01-01T00:00:00+00:00",
        "dateEnd": "2026-01-02T00:00:00+00:00",
        "lastEditDate": last_edit,
        "questionQty": {"1": qcount_total},
        "editors": [],
    }


def _result_row(team_id: int, *, mask: str = "111111111111111111111111111111111111") -> dict:
    return {
        "team": {"id": team_id, "name": f"team-{team_id}"},
        "mask": mask,
        "position": 1.0,
        "questionsTotal": sum(c == "1" for c in mask),
        "teamMembers": [
            {"flag": "К", "player": {"id": team_id * 10 + 0}},
            {"flag": "Б", "player": {"id": team_id * 10 + 1}},
        ],
    }


class _FakeClient:
    """Stand-in for RatingApiClient — pre-canned discovery + fetches.

    Failures are scheduled by id in ``fail_meta_for`` / ``fail_results_for``.
    """

    def __init__(
        self,
        discovered: list[dict],
        *,
        meta_by_tid: dict[int, dict] | None = None,
        results_by_tid: dict[int, list[dict]] | None = None,
        fail_meta_for: set[int] | None = None,
        fail_results_for: set[int] | None = None,
    ) -> None:
        self.discovered = discovered
        self.meta_by_tid = meta_by_tid or {}
        self.results_by_tid = results_by_tid or {}
        self.fail_meta_for = fail_meta_for or set()
        self.fail_results_for = fail_results_for or set()
        self.calls_get_tournament: list[int] = []
        self.calls_get_results: list[int] = []
        self.iter_args: list[str] = []

    def iter_tournaments_changed_since(self, since: str, *, items_per_page: int = 512):
        self.iter_args.append(since)
        yield from self.discovered

    def get_tournament(self, tid: int) -> dict:
        self.calls_get_tournament.append(int(tid))
        if int(tid) in self.fail_meta_for:
            raise RatingApiError(f"boom-meta-{tid}", status=502)
        return self.meta_by_tid.get(int(tid), _meta(int(tid), "2026-01-02T00:00:00+00:00"))

    def get_results(self, tid: int, *, include_team_members=True, include_masks=True) -> list:
        self.calls_get_results.append(int(tid))
        if int(tid) in self.fail_results_for:
            raise RatingApiError(f"boom-results-{tid}", status=502)
        return self.results_by_tid.get(int(tid), [])


def _patch_db(monkey, *, cursor: str | None = "2026-04-21T11:27:14"):
    """Patch all DB-touching functions inside rating_api.sync."""
    monkey.start("rating_api.sync.open_conn", return_value=MagicMock())
    monkey.start("rating_api.sync.ensure_schema", return_value=None)
    monkey.start("rating_api.sync.discovery_cursor", return_value=cursor)
    return monkey


class _MonkeyPatchSet:
    """Tiny helper to start a batch of patches and stop them all in tearDown."""

    def __init__(self) -> None:
        self._patchers = []
        self.mocks: dict[str, MagicMock] = {}

    def start(self, target: str, **kw) -> MagicMock:
        p = patch(target, **kw)
        mock = p.start()
        self._patchers.append(p)
        self.mocks[target] = mock
        return mock

    def stop_all(self) -> None:
        for p in self._patchers:
            p.stop()


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------


class RunSyncTests(unittest.TestCase):

    def setUp(self) -> None:
        self.mp = _MonkeyPatchSet()

    def tearDown(self) -> None:
        self.mp.stop_all()

    def test_empty_discovery_is_a_clean_noop(self):
        # Walk yields nothing → no upsert, no record_fetch, no error.
        _patch_db(self.mp)
        upsert = self.mp.start("rating_api.sync.upsert_bundle")
        record = self.mp.start("rating_api.sync.record_fetch")
        client = _FakeClient(discovered=[])

        stats = run_sync(client=client, dry_run=False, verbose=False)

        self.assertEqual(stats.discovered, 0)
        self.assertEqual(stats.fetched_ok, 0)
        self.assertEqual(upsert.call_count, 0)
        self.assertEqual(record.call_count, 0)
        # The auto-cursor path was exercised.
        self.assertEqual(client.iter_args, ["2026-04-21T11:27:14"])

    def test_happy_path_calls_upsert_and_records_state(self):
        # 2 tournaments, both fetch + upsert cleanly → 2 upserts, 2
        # record_fetch calls with http_status=200, no errors.
        _patch_db(self.mp)
        upsert = self.mp.start("rating_api.sync.upsert_bundle")
        record = self.mp.start("rating_api.sync.record_fetch")
        client = _FakeClient(
            discovered=[
                _summary(1, "2026-05-01T00:00:00+00:00"),
                _summary(2, "2026-05-02T00:00:00+00:00"),
            ],
            results_by_tid={
                1: [_result_row(101), _result_row(102)],
                2: [_result_row(201)],
            },
        )

        stats = run_sync(client=client, dry_run=False, verbose=False)

        self.assertEqual(stats.discovered, 2)
        self.assertEqual(stats.fetched_ok, 2)
        self.assertEqual(stats.fetched_err, 0)
        self.assertEqual(stats.upsert_err, 0)
        self.assertEqual(upsert.call_count, 2)
        # tournament_id order is preserved — discovery order in,
        # upsert order out.
        upserted_tids = [c.args[1].tournament.id for c in upsert.call_args_list]
        self.assertEqual(upserted_tids, [1, 2])

        self.assertEqual(record.call_count, 2)
        for call in record.call_args_list:
            self.assertEqual(call.kwargs["http_status"], 200)
            self.assertIsNone(call.kwargs["error_message"])

    def test_dry_run_skips_upsert_but_still_records_state(self):
        # The whole point of --dry-run: changes to public.* are gated;
        # api_overlay.fetch_state still gets the observed counts.
        _patch_db(self.mp)
        upsert = self.mp.start("rating_api.sync.upsert_bundle")
        record = self.mp.start("rating_api.sync.record_fetch")
        client = _FakeClient(
            discovered=[_summary(1, "2026-05-01T00:00:00+00:00")],
            results_by_tid={1: [_result_row(101)]},
        )

        stats = run_sync(client=client, dry_run=True, verbose=False)

        self.assertEqual(upsert.call_count, 0)
        self.assertEqual(record.call_count, 1)
        self.assertEqual(stats.fetched_ok, 1)

    def test_fetch_error_is_contained_and_recorded_with_status(self):
        # A bad tournament must not abort the whole run — record the
        # error in state, keep going through the rest of the page.
        _patch_db(self.mp)
        upsert = self.mp.start("rating_api.sync.upsert_bundle")
        record = self.mp.start("rating_api.sync.record_fetch")
        client = _FakeClient(
            discovered=[
                _summary(1, "2026-05-01T00:00:00+00:00"),
                _summary(2, "2026-05-02T00:00:00+00:00"),
                _summary(3, "2026-05-03T00:00:00+00:00"),
            ],
            results_by_tid={1: [_result_row(11)], 3: [_result_row(31)]},
            fail_results_for={2},
        )

        stats = run_sync(client=client, dry_run=False, verbose=False)

        # 1 and 3 succeed; 2 fails on /results — but doesn't tank 3.
        self.assertEqual(stats.fetched_ok, 2)
        self.assertEqual(stats.fetched_err, 1)
        self.assertEqual(upsert.call_count, 2)

        # The recorded state for tid=2 must carry the HTTP status the
        # client raised with (502 — our fake encodes that).
        record_by_tid = {
            c.kwargs["tournament_id"]: c.kwargs for c in record.call_args_list
        }
        self.assertEqual(record_by_tid[2]["http_status"], 502)
        self.assertIsNotNone(record_by_tid[2]["error_message"])
        self.assertEqual(record_by_tid[1]["http_status"], 200)
        self.assertEqual(record_by_tid[3]["http_status"], 200)

    def test_upsert_error_is_contained_and_recorded(self):
        # If the upsert itself raises (e.g. PG hiccup), don't abort —
        # bump upsert_err, record the error, move to the next.
        _patch_db(self.mp)
        upsert = self.mp.start(
            "rating_api.sync.upsert_bundle",
            side_effect=[RuntimeError("pg-down"), None],
        )
        record = self.mp.start("rating_api.sync.record_fetch")
        client = _FakeClient(
            discovered=[
                _summary(1, "2026-05-01T00:00:00+00:00"),
                _summary(2, "2026-05-02T00:00:00+00:00"),
            ],
            results_by_tid={1: [_result_row(11)], 2: [_result_row(22)]},
        )

        stats = run_sync(client=client, dry_run=False, verbose=False)

        self.assertEqual(stats.fetched_ok, 2)   # API calls succeeded
        self.assertEqual(stats.upsert_err, 1)   # one upsert failed
        self.assertEqual(upsert.call_count, 2)

        # The failing tournament's state row carries the error msg.
        record_by_tid = {
            c.kwargs["tournament_id"]: c.kwargs for c in record.call_args_list
        }
        self.assertIsNotNone(record_by_tid[1]["error_message"])
        self.assertIn("upsert failed", record_by_tid[1]["error_message"])
        self.assertIsNone(record_by_tid[2]["error_message"])

    def test_empty_results_payload_is_counted_separately(self):
        # The signal the operator wants to see in the summary: how many
        # tournaments were "not ready yet" so we know what's pending.
        _patch_db(self.mp)
        self.mp.start("rating_api.sync.upsert_bundle")
        self.mp.start("rating_api.sync.record_fetch")
        client = _FakeClient(
            discovered=[
                _summary(1, "2026-05-01T00:00:00+00:00"),
                _summary(2, "2026-05-02T00:00:00+00:00"),
            ],
            # tid=1 has results; tid=2 doesn't (the "not ready" case).
            results_by_tid={1: [_result_row(11)]},
        )

        stats = run_sync(client=client, dry_run=False, verbose=False)

        self.assertEqual(stats.fetched_ok, 2)
        self.assertEqual(stats.n_empty_results, 1)
        self.assertEqual(stats.total_results, 1)

    def test_explicit_since_overrides_auto_cursor(self):
        # If the caller passes --since, the discovery_cursor() lookup
        # must NOT be invoked (avoids hitting PG unnecessarily and
        # gives operators a manual escape hatch).
        _patch_db(self.mp)
        self.mp.start("rating_api.sync.upsert_bundle")
        self.mp.start("rating_api.sync.record_fetch")
        client = _FakeClient(discovered=[])

        run_sync(
            since="2024-01-01",
            client=client,
            dry_run=False,
            verbose=False,
        )
        self.assertEqual(client.iter_args, ["2024-01-01"])
        self.mp.mocks["rating_api.sync.discovery_cursor"].assert_not_called()

    def test_limit_caps_discovery(self):
        # --limit N must stop processing after N items even though the
        # generator has more.  Protects the smoke-test path.
        _patch_db(self.mp)
        upsert = self.mp.start("rating_api.sync.upsert_bundle")
        self.mp.start("rating_api.sync.record_fetch")
        client = _FakeClient(
            discovered=[_summary(i, f"2026-05-{i:02d}T00:00:00+00:00")
                        for i in range(1, 11)],
            results_by_tid={i: [_result_row(i * 100)] for i in range(1, 11)},
        )

        stats = run_sync(client=client, limit=3, dry_run=False, verbose=False)
        self.assertEqual(stats.discovered, 3)
        self.assertEqual(upsert.call_count, 3)

    def test_no_cursor_available_raises_clean(self):
        # Empty PG (no dump restored yet) + no --since must give a
        # clear error, not a cryptic exception inside the iterator.
        _patch_db(self.mp, cursor=None)
        client = _FakeClient(discovered=[])
        with self.assertRaises(RuntimeError) as cm:
            run_sync(client=client, dry_run=False, verbose=False)
        self.assertIn("cursor", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
