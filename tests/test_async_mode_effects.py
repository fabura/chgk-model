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


if __name__ == "__main__":
    unittest.main()
