"""Unit tests for ``rating.simulate.simulate_roster_on_pack``."""
from __future__ import annotations

import math
import unittest

import numpy as np

from rating.simulate import (
    apply_probability_calibration,
    simulate_roster_on_pack,
)


class SimulateRosterOnPackTests(unittest.TestCase):
    def test_single_player_no_calibration(self):
        # θ=0, b=0, a=1, no δ → S = e^0 = 1, p = 1 − e^{-1}.
        p = simulate_roster_on_pack(
            thetas=np.array([0.0]),
            b=np.array([0.0]),
            a=np.array([1.0]),
        )
        self.assertEqual(p.shape, (1,))
        self.assertAlmostEqual(float(p[0]), 1.0 - math.exp(-1.0), places=12)

    def test_zero_players_returns_zero_probabilities(self):
        # Empty roster: S = 0 → p_raw = 0; identity calibration leaves zeros.
        p = simulate_roster_on_pack(
            thetas=np.zeros(0),
            b=np.array([0.0, 1.0, -1.0]),
            a=np.array([1.0, 1.0, 1.0]),
        )
        self.assertTrue(np.array_equal(p, np.zeros(3)))

    def test_six_player_team_matches_hand_formula(self):
        # 6 identical players at θ=0.5, single question with b=−0.2, a=1.1.
        # S = 6 · exp(−(−0.2) + 1.1·0.5) = 6 · exp(0.75); p = 1 − exp(−S).
        thetas = np.full(6, 0.5)
        S_expected = 6.0 * math.exp(0.2 + 1.1 * 0.5)
        p_expected = 1.0 - math.exp(-S_expected)
        p = simulate_roster_on_pack(
            thetas=thetas,
            b=np.array([-0.2]),
            a=np.array([1.1]),
        )
        self.assertAlmostEqual(float(p[0]), p_expected, places=10)

    def test_size_shift_anchored_at_anchor(self):
        # δ_size with anchor=6 should subtract δ_size[6] before applying the
        # team-size entry.  For an anchor-sized team, expected output is
        # invariant to δ_size content.
        delta_size = np.array([0.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.2, 0.25, 0.3])
        common = dict(
            thetas=np.zeros(6),
            b=np.array([0.0]),
            a=np.array([1.0]),
            delta_size=delta_size,
            team_size_anchor=6,
        )
        p = simulate_roster_on_pack(**common)
        # Same team, same anchor — δ_size shift cancels out.
        ref = simulate_roster_on_pack(
            thetas=common["thetas"], b=common["b"], a=common["a"]
        )
        self.assertAlmostEqual(float(p[0]), float(ref[0]), places=12)

        # 3-player team should pick up δ_size[3] − δ_size[6] = −0.1 − 0.2 = −0.3
        # added to b (so easier per-question if positive shift is added; here
        # the shift is negative so b_eff is smaller → easier ⇒ higher p).
        p3 = simulate_roster_on_pack(
            thetas=np.zeros(3),
            b=np.array([0.0]),
            a=np.array([1.0]),
            delta_size=delta_size,
            team_size_anchor=6,
        )
        # S = 3 · exp(−(0 + (-0.1 − 0.2)) + 0) = 3 · exp(0.3).
        S = 3.0 * math.exp(0.3)
        self.assertAlmostEqual(float(p3[0]), 1.0 - math.exp(-S), places=10)

    def test_pos_shift_uses_q_in_tour_modulo_tour_len(self):
        # 3-question pack, tour_len=12, q_in_tour=[0, 5, 11] → shifts at those
        # indices minus shift at anchor 0 (which is zero).
        delta_pos = np.array(
            [0.0, 0.05, 0.10, 0.12, 0.15, 0.18, 0.22, 0.20, 0.18, 0.10, 0.05, -0.05]
        )
        thetas = np.zeros(6)
        p = simulate_roster_on_pack(
            thetas=thetas,
            b=np.array([0.0, 0.0, 0.0]),
            a=np.array([1.0, 1.0, 1.0]),
            q_in_tour=np.array([0, 5, 11]),
            delta_pos=delta_pos,
            pos_anchor=0,
        )
        # Expected b_eff_q = 0 + (delta_pos[q] - delta_pos[0]).
        for i, q in enumerate([0, 5, 11]):
            shift = float(delta_pos[q] - delta_pos[0])
            S = 6.0 * math.exp(-shift)
            self.assertAlmostEqual(float(p[i]), 1.0 - math.exp(-S), places=10)

    def test_lapse_caps_probability(self):
        # Strong team: p_raw very close to 1.  With offline-team lapse=0.1 (set
        # via lapse_arr), p_final = 0.9 · p_raw.
        lapse_arr = np.array([[0.1, 0.05], [0.0, 0.0], [0.0, 0.0]])
        thetas = np.full(6, 5.0)  # absurdly strong
        p_no_lapse = simulate_roster_on_pack(
            thetas=thetas, b=np.array([0.0]), a=np.array([1.0])
        )
        p_with_lapse = simulate_roster_on_pack(
            thetas=thetas,
            b=np.array([0.0]),
            a=np.array([1.0]),
            mode="offline",
            lapse_arr=lapse_arr,
        )
        self.assertAlmostEqual(
            float(p_with_lapse[0]), 0.9 * float(p_no_lapse[0]), places=12
        )

    def test_recal_alpha_zero_beta_one_is_identity(self):
        # Identity recal must short-circuit and not pull p through logit.
        recal_arr = np.zeros((3, 2, 2))
        recal_arr[..., 1] = 1.0  # β = 1
        out = apply_probability_calibration(
            np.array([0.1, 0.5, 0.9]), lapse=0.0, recal_alpha=0.0, recal_beta=1.0
        )
        self.assertTrue(np.allclose(out, [0.1, 0.5, 0.9]))

    def test_solo_uses_solo_calibration_bucket(self):
        # Sync solo: lapse[1, 1] = 0.2 should be picked when n=1, mode="sync".
        lapse_arr = np.zeros((3, 2))
        lapse_arr[1, 1] = 0.2
        p_team = simulate_roster_on_pack(
            thetas=np.array([0.0, 0.0]),
            b=np.array([0.0]),
            a=np.array([1.0]),
            mode="sync",
            lapse_arr=lapse_arr,
        )
        p_solo = simulate_roster_on_pack(
            thetas=np.array([0.0]),
            b=np.array([0.0]),
            a=np.array([1.0]),
            mode="sync",
            lapse_arr=lapse_arr,
        )
        # Team uses bucket [1, 0] = 0.0 (no lapse).
        S_team = 2.0 * math.exp(0.0)
        self.assertAlmostEqual(float(p_team[0]), 1.0 - math.exp(-S_team), places=10)
        # Solo uses bucket [1, 1] = 0.2 → p = 0.8 · (1 − e^{-1}).
        S_solo = math.exp(0.0)
        self.assertAlmostEqual(
            float(p_solo[0]), 0.8 * (1.0 - math.exp(-S_solo)), places=10
        )


if __name__ == "__main__":
    unittest.main()
