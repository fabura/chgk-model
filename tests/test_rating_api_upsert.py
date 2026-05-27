"""Tests for ``rating_api.upsert``.

Two layers:

* Mock tests (always run) — lock in the transaction shape: DELETE+INSERT
  ordering, the empty-payload skip rule, and rollback-on-error.

* Integration tests (skipped if the local rating PG isn't reachable) —
  use a synthetic tournament_id that can't collide with real data,
  exercise the full path against real PG, then clean up.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from rating_api.parse import (
    ParsedEditor,
    ParsedResult,
    ParsedRoster,
    ParsedTournament,
    ParsedTournamentBundle,
)
from rating_api.upsert import upsert_bundle


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _bundle(tid: int = 999_654_321, *, n_results: int, n_editors: int = 2):
    """Build a synthetic bundle.  Roster count = 4 * n_results (a fake
    average team size that varies enough to spot off-by-one bugs)."""
    t = ParsedTournament(
        id=tid,
        title="test-tournament",
        start_datetime="2026-01-01T00:00:00+00:00",
        end_datetime="2026-01-01T01:00:00+00:00",
        last_edited_at="2026-01-02T00:00:00+00:00",
        questions_count=36,
        typeoft_id=3,
        type="Синхрон",
        maii_rating=False,
    )
    results = [
        ParsedResult(
            tournament_id=tid,
            team_id=1000 + i,
            team_title=f"t{i}",
            total=20,
            position=float(i + 1),
            points_mask="1" * 36,
        )
        for i in range(n_results)
    ]
    rosters = [
        ParsedRoster(
            tournament_id=tid,
            team_id=1000 + i,
            player_id=10_000 + i * 4 + k,
            flag="К" if k == 0 else "Б",
            is_captain=(k == 0),
        )
        for i in range(n_results)
        for k in range(4)
    ]
    editors = [
        ParsedEditor(tournament_id=tid, player_id=20_000 + j)
        for j in range(n_editors)
    ]
    return ParsedTournamentBundle(t, results, rosters, editors)


def _make_mock_conn():
    """Build a MagicMock connection whose .cursor() returns a context-
    manager yielding a recording mock cursor.  Exposes:
      - conn.cursor_obj   — the mock cursor (collect .execute calls)
      - conn.commit       — MagicMock
      - conn.rollback     — MagicMock
    """
    conn = MagicMock()
    cur = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = cur
    cm.__exit__.return_value = False
    conn.cursor.return_value = cm
    conn.cursor_obj = cur
    return conn


def _executed_sqls(mock_cursor) -> list[str]:
    """Return the first arg of every cur.execute(...) call as a single
    upper-cased string for easy substring asserts."""
    out = []
    for call in mock_cursor.execute.call_args_list:
        args, _ = call
        if args:
            out.append(args[0].strip())
    return out


# ----------------------------------------------------------------------
# mock-based behavior tests
# ----------------------------------------------------------------------


class UpsertMockTests(unittest.TestCase):

    def test_happy_path_commits_and_writes_all_tables(self):
        b = _bundle(n_results=3, n_editors=2)
        conn = _make_mock_conn()
        with patch("rating_api.upsert.execute_values") as ev:
            stats = upsert_bundle(conn, b)

        # Single commit, no rollback.
        self.assertEqual(conn.commit.call_count, 1)
        self.assertEqual(conn.rollback.call_count, 0)

        sqls = " ".join(_executed_sqls(conn.cursor_obj)).upper()
        self.assertIn("DELETE FROM PUBLIC.TOURNAMENTS WHERE ID", sqls)
        self.assertIn("INSERT INTO PUBLIC.TOURNAMENTS", sqls)
        self.assertIn("DELETE FROM PUBLIC.TOURNAMENT_RESULTS", sqls)
        self.assertIn("DELETE FROM PUBLIC.TOURNAMENT_ROSTERS", sqls)
        self.assertIn("DELETE FROM PUBLIC.TOURNAMENT_EDITORS", sqls)

        # execute_values is the bulk-insert path: results + rosters + editors
        # (3 distinct calls), in that order.
        self.assertEqual(ev.call_count, 3)
        ev_sqls = [c.args[1] for c in ev.call_args_list]
        self.assertIn("public.tournament_results", ev_sqls[0])
        self.assertIn("public.tournament_rosters", ev_sqls[1])
        self.assertIn("public.tournament_editors", ev_sqls[2])

        self.assertEqual(stats["n_results_written"], 3)
        self.assertEqual(stats["n_rosters_written"], 12)
        self.assertEqual(stats["n_editors_written"], 2)
        self.assertEqual(stats["results_skipped_empty"], 0)

    def test_empty_results_does_not_touch_results_or_rosters(self):
        # The whole point of the empty-payload rule: an unfinished
        # tournament must not erase last dump's rows.  We must see
        # NO mention of DELETE FROM tournament_results / _rosters and
        # NO INSERT into them.
        b = _bundle(n_results=0, n_editors=3)
        conn = _make_mock_conn()
        with patch("rating_api.upsert.execute_values") as ev:
            stats = upsert_bundle(conn, b)

        all_sqls = " ".join(_executed_sqls(conn.cursor_obj)).lower()
        self.assertNotIn("tournament_results", all_sqls)
        self.assertNotIn("tournament_rosters", all_sqls)
        # tournaments + editors are still touched.
        self.assertIn("public.tournaments", all_sqls)
        self.assertIn("tournament_editors", all_sqls)

        # execute_values fires once — only for editors.
        ev_targets = [c.args[1] for c in ev.call_args_list]
        self.assertEqual(len(ev_targets), 1)
        self.assertIn("tournament_editors", ev_targets[0])

        self.assertEqual(stats["n_results_written"], 0)
        self.assertEqual(stats["n_rosters_written"], 0)
        self.assertEqual(stats["results_skipped_empty"], 1)

    def test_empty_editors_skips_editor_bulk_insert_but_still_deletes(self):
        # Editors absent in API but the tournament had some before:
        # we MUST still DELETE old editor rows so they don't ghost.
        # We just don't fire an INSERT.
        b = _bundle(n_results=2, n_editors=0)
        conn = _make_mock_conn()
        with patch("rating_api.upsert.execute_values") as ev:
            upsert_bundle(conn, b)

        all_sqls = " ".join(_executed_sqls(conn.cursor_obj)).lower()
        self.assertIn("delete from public.tournament_editors", all_sqls)
        ev_targets = [c.args[1] for c in ev.call_args_list]
        # No execute_values call for editors (only results + rosters).
        self.assertEqual(len(ev_targets), 2)
        self.assertNotIn("editors", " ".join(ev_targets).lower())

    def test_exception_triggers_rollback_and_propagates(self):
        # Verifies the safety net: any DB-level exception during the
        # transaction is rolled back; we never commit a half-written
        # tournament.
        b = _bundle(n_results=2)
        conn = _make_mock_conn()
        with patch(
            "rating_api.upsert.execute_values", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                upsert_bundle(conn, b)
        self.assertEqual(conn.commit.call_count, 0)
        self.assertEqual(conn.rollback.call_count, 1)


# ----------------------------------------------------------------------
# integration tests — real PG, synthetic tournament id
# ----------------------------------------------------------------------


# Use a clearly synthetic id that cannot collide with real rating-db
# tournaments (rating DB ids are < 100_000 as of 2026-05).
TEST_TID = 999_654_321


def _try_connect():
    """Best-effort PG connect; returns conn or None."""
    try:
        import psycopg2
    except ImportError:
        return None
    try:
        url = os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:password@127.0.0.1:5432/postgres",
        )
        # Short timeout: we don't want CI to hang for 30s when PG is down.
        return psycopg2.connect(url, connect_timeout=2)
    except Exception:  # noqa: BLE001
        return None


class UpsertIntegrationTests(unittest.TestCase):
    """Hits the real rating PG.  Skipped if the local docker-compose
    rating-db isn't up."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.conn = _try_connect()
        if cls.conn is None:
            raise unittest.SkipTest("rating PG not reachable on localhost")

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "conn", None) is not None:
            cls.conn.close()

    def setUp(self) -> None:
        self._delete_test_rows()

    def tearDown(self) -> None:
        self._delete_test_rows()

    def _delete_test_rows(self) -> None:
        with self.conn.cursor() as cur:
            for t in (
                "tournament_rosters",
                "tournament_results",
                "tournament_editors",
                "tournaments",
            ):
                cur.execute(
                    f"DELETE FROM public.{t} WHERE "
                    f"{'id' if t == 'tournaments' else 'tournament_id'} = %s",
                    (TEST_TID,),
                )
        self.conn.commit()

    def _count(self, table: str) -> int:
        with self.conn.cursor() as cur:
            col = "id" if table == "tournaments" else "tournament_id"
            cur.execute(
                f"SELECT count(*) FROM public.{table} WHERE {col} = %s",
                (TEST_TID,),
            )
            return int(cur.fetchone()[0])

    # ------------------------------------------------------------------
    def test_first_upsert_inserts_all_rows(self):
        # Tabula-rasa case: no prior rows; all 4 tables must populate.
        b = _bundle(TEST_TID, n_results=3, n_editors=2)
        stats = upsert_bundle(self.conn, b)

        self.assertEqual(self._count("tournaments"), 1)
        self.assertEqual(self._count("tournament_results"), 3)
        self.assertEqual(self._count("tournament_rosters"), 12)
        self.assertEqual(self._count("tournament_editors"), 2)
        self.assertEqual(stats["n_results_written"], 3)
        self.assertEqual(stats["n_rosters_written"], 12)

    def test_repeat_upsert_is_count_stable(self):
        # Idempotency at the row-count level: re-running the same
        # bundle must not duplicate rows.  (This caught the original
        # ON CONFLICT-vs-DELETE design question.)
        b = _bundle(TEST_TID, n_results=3, n_editors=2)
        upsert_bundle(self.conn, b)
        upsert_bundle(self.conn, b)
        self.assertEqual(self._count("tournaments"), 1)
        self.assertEqual(self._count("tournament_results"), 3)
        self.assertEqual(self._count("tournament_rosters"), 12)
        self.assertEqual(self._count("tournament_editors"), 2)

    def test_roster_count_can_shrink_between_upserts(self):
        # The original bug we caught on tid=13556 (330 → 329 rosters):
        # if the API drops a player, the next upsert must reflect that.
        # DELETE-then-INSERT makes this trivial — but we lock it.
        big = _bundle(TEST_TID, n_results=3, n_editors=2)   # 12 rosters
        small = _bundle(TEST_TID, n_results=2, n_editors=2)  #  8 rosters
        upsert_bundle(self.conn, big)
        self.assertEqual(self._count("tournament_rosters"), 12)
        upsert_bundle(self.conn, small)
        self.assertEqual(self._count("tournament_rosters"), 8)
        self.assertEqual(self._count("tournament_results"), 2)

    def test_empty_payload_preserves_existing_results_and_rosters(self):
        # The crown jewel: seed with real-looking rows (as if the dump
        # loaded them), then run an "API still hasn't seen results yet"
        # upsert.  The seeded rows MUST survive — only tournaments
        # metadata and editors get refreshed.
        seed = _bundle(TEST_TID, n_results=3, n_editors=2)
        upsert_bundle(self.conn, seed)
        self.assertEqual(self._count("tournament_results"), 3)
        self.assertEqual(self._count("tournament_rosters"), 12)

        empty = _bundle(TEST_TID, n_results=0, n_editors=5)
        stats = upsert_bundle(self.conn, empty)

        # Untouched:
        self.assertEqual(self._count("tournament_results"), 3)
        self.assertEqual(self._count("tournament_rosters"), 12)
        # Refreshed:
        self.assertEqual(self._count("tournaments"), 1)
        self.assertEqual(self._count("tournament_editors"), 5)
        self.assertEqual(stats["results_skipped_empty"], 1)

    def test_tournaments_columns_round_trip(self):
        # Spot-check that the most important columns survive the
        # round-trip — this is the actual contract with load_from_db.
        b = _bundle(TEST_TID, n_results=1, n_editors=1)
        upsert_bundle(self.conn, b)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT title, type, typeoft_id, questions_count, "
                "       last_edited_at::text "
                "FROM public.tournaments WHERE id = %s",
                (TEST_TID,),
            )
            row = cur.fetchone()
        self.assertEqual(row[0], "test-tournament")
        self.assertEqual(row[1], "Синхрон")
        self.assertEqual(row[2], 3)
        self.assertEqual(row[3], 36)
        # last_edited_at — timestamp stored without tz, value preserved
        # up to seconds (PG cast drops the +00:00 suffix).
        self.assertTrue(row[4].startswith("2026-01-02 00:00:00"))


if __name__ == "__main__":
    unittest.main()
