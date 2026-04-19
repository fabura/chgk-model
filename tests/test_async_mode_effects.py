import unittest

import numpy as np

from data import IndexMaps
from rating.engine import Config, run_sequential
from rating.tournaments import TYPE_ASYNC, TYPE_OFFLINE, TYPE_SYNC, TournamentState


class TournamentStateTests(unittest.TestCase):
    def test_total_delta_and_center_by_type(self) -> None:
        tournaments = TournamentState(
            num_games=5,
            game_type=["offline", "sync", "sync", "async", "async"],
        )
        tournaments.mu_type[TYPE_SYNC] = -0.2
        tournaments.mu_type[TYPE_ASYNC] = -0.5
        tournaments.eps[:] = np.array([0.4, 0.1, 0.5, -0.3, 0.7], dtype=np.float64)

        self.assertAlmostEqual(tournaments.total_delta(1), -0.1)
        self.assertAlmostEqual(tournaments.total_delta(3), -0.8)

        tournaments.center([0, 1, 2, 3, 4])

        self.assertAlmostEqual(float(tournaments.eps[0]), 0.0, places=10)
        self.assertAlmostEqual(float(np.mean(tournaments.eps[[1, 2]])), 0.0, places=10)
        self.assertAlmostEqual(float(np.mean(tournaments.eps[[3, 4]])), 0.0, places=10)


class AsyncUpdateTests(unittest.TestCase):
    def _run_single_game(self, game_type: str):
        arrays = {
            "q_idx": np.array([0], dtype=np.int32),
            "taken": np.array([1], dtype=np.float64),
            "team_sizes": np.array([1], dtype=np.int32),
            "player_indices_flat": np.array([0], dtype=np.int32),
            "game_idx": np.array([0], dtype=np.int32),
        }
        maps = IndexMaps(
            player_id_to_idx={100: 0},
            question_id_to_idx={(200, 0): 0},
            idx_to_player_id=[100],
            idx_to_question_id=[(200, 0)],
            idx_to_game_id=[200],
            game_type=np.array([game_type], dtype=object),
            game_date_ordinal=np.array([1], dtype=np.int32),
        )
        cfg = Config(
            eta0=0.1,
            rho=1.0,
            w_online=0.25,
            w_online_questions=1.0,
            w_online_log_a=1.0,
            eta_mu=0.1,
            eta_eps=0.1,
            reg_mu_type=0.0,
            reg_eps=0.0,
            use_tournament_delta=True,
        )
        return run_sequential(arrays, maps, cfg, verbose=False)

    def test_async_player_updates_are_smaller_than_offline(self) -> None:
        offline = self._run_single_game("offline")
        async_result = self._run_single_game("async")

        self.assertGreater(float(offline.players.theta[0]), float(async_result.players.theta[0]))
        self.assertAlmostEqual(float(offline.tournaments.mu_type[TYPE_OFFLINE]), 0.0, places=10)
        self.assertLess(float(async_result.tournaments.mu_type[TYPE_ASYNC]), 0.0)


class TeamSizeEffectTests(unittest.TestCase):
    """Smoke tests for the per-team-size difficulty shift δ_size."""

    def _run_with_size(self, team_size: int, *, use_size_effect: bool, eta_size: float = 0.5):
        # Single tournament, single question, team of given size, all
        # players new (θ=0).  With use_size_effect=True and a single
        # tournament we expect the question-level gradient to be split
        # between b, ε_t and δ_size; with use_size_effect=False, only b
        # and ε_t move.
        team = np.arange(team_size, dtype=np.int32)
        arrays = {
            "q_idx": np.array([0], dtype=np.int32),
            "taken": np.array([0], dtype=np.float64),
            "team_sizes": np.array([team_size], dtype=np.int32),
            "player_indices_flat": team,
            "game_idx": np.array([0], dtype=np.int32),
        }
        maps = IndexMaps(
            player_id_to_idx={i: i for i in range(team_size)},
            question_id_to_idx={(200, 0): 0},
            idx_to_player_id=list(range(team_size)),
            idx_to_question_id=[(200, 0)],
            idx_to_game_id=[200],
            game_type=np.array(["offline"], dtype=object),
            game_date_ordinal=np.array([1], dtype=np.int32),
        )
        cfg = Config(
            eta0=0.1,
            eta_mu=0.0,
            eta_eps=0.1,
            reg_mu_type=0.0,
            reg_eps=0.0,
            use_tournament_delta=True,
            use_team_size_effect=use_size_effect,
            eta_size=eta_size,
            reg_size=0.0,
            team_size_anchor=6,
            team_size_max=8,
        )
        return run_sequential(arrays, maps, cfg, verbose=False)

    def test_size_effect_disabled_leaves_delta_size_none(self) -> None:
        result = self._run_with_size(3, use_size_effect=False)
        self.assertIsNone(result.delta_size)

    def test_size_effect_enabled_updates_only_observed_index(self) -> None:
        result = self._run_with_size(3, use_size_effect=True)
        self.assertIsNotNone(result.delta_size)
        ds = result.delta_size
        self.assertEqual(ds.shape, (9,))
        # Anchor (n=6) must stay at zero.
        self.assertEqual(float(ds[6]), 0.0)
        # The observed team size (n=3) should have moved.  taken=0, so
        # the gradient pushes difficulty up (positive δ).
        self.assertGreater(float(ds[3]), 0.0)
        # Indexes that were never observed must remain at zero.
        for n in (1, 2, 4, 5, 7, 8):
            self.assertEqual(float(ds[n]), 0.0)

    def test_size_effect_anchor_is_never_updated(self) -> None:
        result = self._run_with_size(6, use_size_effect=True)
        self.assertIsNotNone(result.delta_size)
        # Even though team_size==anchor was the only observation,
        # delta_size[anchor] must remain pinned at zero.
        self.assertEqual(float(result.delta_size[6]), 0.0)


class PositionEffectTests(unittest.TestCase):
    """Smoke tests for the per-position-in-tour difficulty shift δ_pos."""

    def _run_with_pos(
        self,
        question_pos: int,
        *,
        use_pos_effect: bool,
        eta_pos: float = 0.5,
        pos_anchor: int = 0,
        tour_len: int = 12,
    ):
        # Single team plays a single question whose within-tournament
        # index encodes the position-in-tour we want to probe.
        team = np.arange(3, dtype=np.int32)
        arrays = {
            "q_idx": np.array([0], dtype=np.int32),
            "taken": np.array([0], dtype=np.float64),
            "team_sizes": np.array([3], dtype=np.int32),
            "player_indices_flat": team,
            "game_idx": np.array([0], dtype=np.int32),
        }
        maps = IndexMaps(
            player_id_to_idx={i: i for i in range(3)},
            question_id_to_idx={(200, question_pos): 0},
            idx_to_player_id=list(range(3)),
            # Position-in-tour comes from the second tuple element.
            idx_to_question_id=[(200, question_pos)],
            idx_to_game_id=[200],
            game_type=np.array(["offline"], dtype=object),
            game_date_ordinal=np.array([1], dtype=np.int32),
        )
        cfg = Config(
            eta0=0.1,
            eta_mu=0.0,
            eta_eps=0.1,
            reg_mu_type=0.0,
            reg_eps=0.0,
            use_tournament_delta=True,
            use_team_size_effect=False,
            use_pos_effect=use_pos_effect,
            eta_pos=eta_pos,
            reg_pos=0.0,
            pos_anchor=pos_anchor,
            tour_len=tour_len,
        )
        return run_sequential(arrays, maps, cfg, verbose=False)

    def test_pos_effect_disabled_leaves_delta_pos_none(self) -> None:
        result = self._run_with_pos(3, use_pos_effect=False)
        self.assertIsNone(result.delta_pos)

    def test_pos_effect_enabled_updates_only_observed_index(self) -> None:
        result = self._run_with_pos(3, use_pos_effect=True, pos_anchor=0)
        self.assertIsNotNone(result.delta_pos)
        dp = result.delta_pos
        self.assertEqual(dp.shape, (12,))
        # Anchor (p=0) must remain zero.
        self.assertEqual(float(dp[0]), 0.0)
        # taken=0 ⇒ gradient pushes difficulty up at the observed
        # position (p=3 % 12 = 3).
        self.assertGreater(float(dp[3]), 0.0)
        # No other position must have moved.
        for p in (1, 2, 4, 5, 6, 7, 8, 9, 10, 11):
            self.assertEqual(float(dp[p]), 0.0)

    def test_pos_effect_anchor_is_never_updated(self) -> None:
        # Observe a question at position == anchor; the anchor entry
        # must stay pinned at zero.
        result = self._run_with_pos(0, use_pos_effect=True, pos_anchor=0)
        self.assertIsNotNone(result.delta_pos)
        self.assertEqual(float(result.delta_pos[0]), 0.0)

    def test_pos_modulo_tour_len(self) -> None:
        # Position 14 in a 12-question tour should map to slot 2.
        result = self._run_with_pos(14, use_pos_effect=True, pos_anchor=0)
        dp = result.delta_pos
        self.assertGreater(float(dp[2]), 0.0)
        self.assertEqual(float(dp[14 % 12]), float(dp[2]))


if __name__ == "__main__":
    unittest.main()
