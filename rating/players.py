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
    def learning_rate(
        self,
        player_idx: int,
        eta0: float,
        games_offset: float = 1.0,
    ) -> float:
        """Adaptive learning rate: η_k = η0 / √(games_offset + games_k).

        ``games_offset`` < 1.0 produces a chess-Elo-style "rookie boost":
        the very first games get a larger learning rate than the asymptotic
        η0/√games, fading smoothly as games accumulate.
        """
        return eta0 / math.sqrt(games_offset + self.games[player_idx])

    # ------------------------------------------------------------------
    def initialize_new(self, player_idx: int, prior: float = 0.0) -> None:
        """Set θ for a first-time player to ``prior`` (idempotent).

        Combined with a rookie boost (``games_offset`` < 1.0 in the
        learning-rate formula), this gives every newcomer a fixed
        conservative starting θ that the data then moves upward
        quickly.  Breaks the cold-start positive-feedback loop where
        weak rookies pull down team means and subsequent rookies
        inherit ever-lower starting θ.

        The legacy team-mean cold-start path was removed in 2026-05;
        see ``docs/cleanup_2026-05.md``.
        """
        if self.seen[player_idx]:
            return
        self.theta[player_idx] = prior
        self.seen[player_idx] = True
