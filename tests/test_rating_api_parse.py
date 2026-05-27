"""Unit tests for ``rating_api.parse``.

Pure-function tests — no network, no DB.  Each test exercises one
behavior we depend on for the upsert layer to produce correct PG rows.
"""
from __future__ import annotations

import unittest

from rating_api.parse import (
    ParsedEditor,
    ParsedResult,
    ParsedRoster,
    parse_editors,
    parse_results_blob,
    parse_tournament_blob,
)


# ----------------------------------------------------------------------
# parse_tournament_blob
# ----------------------------------------------------------------------


class ParseTournamentBlobTests(unittest.TestCase):
    def _full_blob(self, **overrides):
        blob = {
            "id": 9930,
            "name": "Лайт Нау. Листопад.",
            "type": {"id": 3, "name": "Синхрон", "shortName": "С"},
            "dateStart": "2023-10-28T12:00:00+00:00",
            "dateEnd": "2023-11-04T12:00:00+00:00",
            "lastEditDate": "2025-01-29T11:01:39+00:00",
            "questionQty": {"1": 12, "2": 12, "3": 12},
            "maiiRating": True,
        }
        blob.update(overrides)
        return blob

    def test_happy_path_maps_all_fields(self):
        t = parse_tournament_blob(self._full_blob())
        self.assertEqual(t.id, 9930)
        self.assertEqual(t.title, "Лайт Нау. Листопад.")
        # type expands to BOTH typeoft_id (int) and type (Russian text),
        # because PG dumps store both columns.
        self.assertEqual(t.typeoft_id, 3)
        self.assertEqual(t.type, "Синхрон")
        self.assertEqual(t.start_datetime, "2023-10-28T12:00:00+00:00")
        self.assertEqual(t.end_datetime, "2023-11-04T12:00:00+00:00")
        self.assertEqual(t.last_edited_at, "2025-01-29T11:01:39+00:00")
        # questionQty is a per-tour dict; PG stores the total → sum.
        # 12 + 12 + 12 = 36.
        self.assertEqual(t.questions_count, 36)
        self.assertEqual(t.maii_rating, True)

    def test_type_as_bare_int_keeps_id_drops_name(self):
        # /tournaments listing returns `type` as a bare int (no name).
        # We accept that shape — `type.name` becomes None and the
        # caller can fill it later from a fresh /tournaments/{id} fetch.
        t = parse_tournament_blob(self._full_blob(type=3))
        self.assertEqual(t.typeoft_id, 3)
        self.assertIsNone(t.type)

    def test_type_missing(self):
        blob = self._full_blob()
        del blob["type"]
        t = parse_tournament_blob(blob)
        self.assertIsNone(t.typeoft_id)
        self.assertIsNone(t.type)

    def test_questionQty_as_scalar_int_kept(self):
        # Defensive: some endpoints may simplify questionQty to a scalar.
        t = parse_tournament_blob(self._full_blob(questionQty=36))
        self.assertEqual(t.questions_count, 36)

    def test_questionQty_dict_with_non_numeric_values_skipped(self):
        # If a tour value can't be int()-ified, skip it but keep summing
        # the others.  Defensive against API regressions.
        t = parse_tournament_blob(
            self._full_blob(questionQty={"1": 12, "2": "?", "3": 12})
        )
        self.assertEqual(t.questions_count, 24)

    def test_questionQty_missing_yields_none(self):
        blob = self._full_blob()
        del blob["questionQty"]
        t = parse_tournament_blob(blob)
        self.assertIsNone(t.questions_count)

    def test_questionQty_all_zero_yields_none(self):
        # `or None` collapses 0 → None to match how the dump stores
        # tournaments with unknown qty.
        t = parse_tournament_blob(self._full_blob(questionQty={"1": 0}))
        self.assertIsNone(t.questions_count)

    def test_maiiRating_non_bool_becomes_none(self):
        # We only accept strict booleans; anything else falls back to
        # None so we never write garbage into a BOOLEAN column.
        t = parse_tournament_blob(self._full_blob(maiiRating="yes"))
        self.assertIsNone(t.maii_rating)

    def test_missing_id_raises(self):
        # No id → can't form a PG row.  Loud failure is correct.
        with self.assertRaises(ValueError):
            parse_tournament_blob(self._full_blob(id=None))


# ----------------------------------------------------------------------
# parse_results_blob
# ----------------------------------------------------------------------


class ParseResultsBlobTests(unittest.TestCase):
    def _result_row(self, **overrides):
        row = {
            "team": {"id": 37591, "name": "Флориана"},
            "mask": "111111111111111110110111111111111011",
            "position": 1,
            "questionsTotal": 33,
            "teamMembers": [
                {"flag": "К", "player": {"id": 100, "surname": "Иванов"}},
                {"flag": "Б", "player": {"id": 101, "surname": "Петров"}},
            ],
        }
        row.update(overrides)
        return row

    def test_single_row_produces_result_and_rosters(self):
        results, rosters = parse_results_blob(9930, [self._result_row()])

        self.assertEqual(len(results), 1)
        self.assertEqual(len(rosters), 2)

        r = results[0]
        self.assertEqual(r.tournament_id, 9930)
        self.assertEqual(r.team_id, 37591)
        self.assertEqual(r.team_title, "Флориана")
        self.assertEqual(r.total, 33)
        # position is upcast to float so PG `double precision` accepts
        # both whole-place finishes and tied (1.5) results.
        self.assertIsInstance(r.position, float)
        self.assertEqual(r.position, 1.0)
        self.assertEqual(r.points_mask, "111111111111111110110111111111111011")

    def test_tied_position_kept_as_float(self):
        # Tied teams report position as 1.5 (etc.).  Must not be coerced
        # to int — load_from_db sorts by this.
        results, _ = parse_results_blob(
            9930, [self._result_row(position=1.5)]
        )
        self.assertEqual(results[0].position, 1.5)

    def test_position_none_propagated(self):
        # Position is genuinely nullable in PG (DSQ rows etc.).
        results, _ = parse_results_blob(
            9930, [self._result_row(position=None)]
        )
        self.assertIsNone(results[0].position)

    def test_position_garbage_becomes_none(self):
        # Defensive: never blow up parsing, never write garbage.
        results, _ = parse_results_blob(
            9930, [self._result_row(position="huh")]
        )
        self.assertIsNone(results[0].position)

    def test_questionsTotal_garbage_becomes_none(self):
        results, _ = parse_results_blob(
            9930, [self._result_row(questionsTotal="x")]
        )
        self.assertIsNone(results[0].total)

    def test_mask_null_kept_as_null(self):
        # Tournament with no mask yet → null in PG.  load_from_db
        # already skips these rows.
        results, _ = parse_results_blob(
            9930, [self._result_row(mask=None)]
        )
        self.assertIsNone(results[0].points_mask)

    def test_row_without_team_id_is_dropped(self):
        # Defensive: a row with no team.id is unmappable; skip it
        # entirely instead of producing a row with team_id=None that
        # would then explode in the upsert.
        bad = self._result_row()
        bad["team"] = {"name": "no-id"}
        results, rosters = parse_results_blob(9930, [bad])
        self.assertEqual(results, [])
        self.assertEqual(rosters, [])

    def test_captain_flag_sets_is_captain(self):
        # 'К' is the Russian captain marker; other flags map to False.
        results, rosters = parse_results_blob(9930, [self._result_row()])
        del results  # only rosters matter here
        by_pid = {r.player_id: r for r in rosters}
        self.assertTrue(by_pid[100].is_captain)
        self.assertEqual(by_pid[100].flag, "К")
        self.assertFalse(by_pid[101].is_captain)
        self.assertEqual(by_pid[101].flag, "Б")

    def test_flag_null_kept(self):
        # Newer async rows return flag=null (no roster signoff yet).
        # is_captain must be False but flag itself stays null.
        row = self._result_row(
            teamMembers=[{"flag": None, "player": {"id": 200}}]
        )
        _, rosters = parse_results_blob(9930, [row])
        self.assertEqual(len(rosters), 1)
        self.assertIsNone(rosters[0].flag)
        self.assertFalse(rosters[0].is_captain)

    def test_member_without_player_id_is_dropped(self):
        # tournament_rosters has UNIQUE on (player_id, tournament_id,
        # team_id) and disallows NULL player_id.  Drop the bad member
        # but keep the others.
        row = self._result_row(
            teamMembers=[
                {"flag": "Б", "player": {}},  # no id
                {"flag": "Б", "player": {"id": 102}},
            ]
        )
        _, rosters = parse_results_blob(9930, [row])
        self.assertEqual([r.player_id for r in rosters], [102])

    def test_empty_results_list_returns_empty_lists(self):
        # The "tournament has no /results yet" case.  Caller must check
        # this and apply the empty-payload rule.
        results, rosters = parse_results_blob(9930, [])
        self.assertEqual(results, [])
        self.assertEqual(rosters, [])

    def test_row_without_teamMembers_keeps_result_drops_rosters(self):
        # Some tournaments expose results without recaps yet — we
        # still want the result row (mask, position) for load_from_db.
        row = self._result_row()
        del row["teamMembers"]
        results, rosters = parse_results_blob(9930, [row])
        self.assertEqual(len(results), 1)
        self.assertEqual(rosters, [])


# ----------------------------------------------------------------------
# parse_editors
# ----------------------------------------------------------------------


class ParseEditorsTests(unittest.TestCase):
    def _t(self, editors):
        return parse_tournament_blob({
            "id": 1,
            "name": "x",
            "editors": editors,
        })

    def test_extracts_ids(self):
        t = self._t(editors=[
            {"id": 22331, "name": "Константин", "surname": "Науменко"},
            {"id": 22332, "name": "X", "surname": "Y"},
        ])
        eds = parse_editors(t, {"editors": [
            {"id": 22331}, {"id": 22332}
        ]})
        self.assertEqual([e.player_id for e in eds], [22331, 22332])
        self.assertTrue(all(e.tournament_id == 1 for e in eds))

    def test_empty_editors_returns_empty(self):
        t = self._t(editors=[])
        eds = parse_editors(t, {})
        self.assertEqual(eds, [])

    def test_non_dict_editor_entry_skipped(self):
        # Defensive: malformed blob (string instead of dict) shouldn't
        # crash; just skip the entry.
        t = self._t(editors=[])
        eds = parse_editors(t, {"editors": ["not-a-dict", {"id": 42}]})
        self.assertEqual([e.player_id for e in eds], [42])

    def test_editor_without_id_skipped(self):
        t = self._t(editors=[])
        eds = parse_editors(t, {"editors": [{"name": "no-id"}, {"id": 5}]})
        self.assertEqual([e.player_id for e in eds], [5])


if __name__ == "__main__":
    unittest.main()
