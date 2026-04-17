"""
Player state: strengths θ and experience tracking.
"""
from __future__ import annotations

import math

import numpy as np


class PlayerState:
    """Mutable store for all player parameters.

    Attributes:
        theta:             player strengths            [num_players]
        games:             tournaments played counter  [num_players]
        seen:              whether a player has been initialised  [num_players]
        last_seen_ordinal: date ordinal of the last tournament the player
                           took part in (-1 = never seen).  Used by
                           calendar-based decay.  [num_players]
    """

    __slots__ = ("num_players", "theta", "games", "seen", "last_seen_ordinal")

    def __init__(self, num_players: int) -> None:
        self.num_players = num_players
        self.theta = np.zeros(num_players, dtype=np.float64)
        self.games = np.zeros(num_players, dtype=np.int64)
        self.seen = np.zeros(num_players, dtype=bool)
        self.last_seen_ordinal = np.full(num_players, -1, dtype=np.int64)

    # ------------------------------------------------------------------
    def learning_rate(self, player_idx: int, eta0: float) -> float:
        """Adaptive learning rate: η_k = η0 / √(1 + games_k)."""
        return eta0 / math.sqrt(1.0 + self.games[player_idx])

    # ------------------------------------------------------------------
    def initialize_new(
        self,
        player_idx: int,
        teammate_indices: list[int],
        cold_factor: float = 1.0,
    ) -> None:
        """Set θ for a first-time player.

        If teammates already have ratings → θ_new = cold_factor · mean(teammate θ).
        Otherwise → θ_new = 0.

        ``cold_factor`` < 1.0 implements partial inheritance: a rookie
        does not immediately get the full team-average rating but is
        shrunk toward zero.  This curbs the "appended player" failure
        mode where a strong roster instantly inflates a newcomer's θ.
        """
        if self.seen[player_idx]:
            return
        known = [i for i in teammate_indices if i != player_idx and self.seen[i]]
        if known:
            self.theta[player_idx] = float(np.mean(self.theta[known])) * cold_factor
        else:
            self.theta[player_idx] = 0.0
        self.seen[player_idx] = True
