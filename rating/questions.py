"""
Question state: difficulty b and selectivity a.
"""
from __future__ import annotations

import math

import numpy as np


class QuestionState:
    """Mutable store for question parameters.

    Attributes:
        b:           difficulty       [num_questions]
        log_a:       log selectivity  [num_questions]  (a = exp(log_a))
        initialized: flag             [num_questions]
    """

    __slots__ = ("num_questions", "b", "log_a", "initialized")

    def __init__(self, num_questions: int) -> None:
        self.num_questions = num_questions
        self.b = np.zeros(num_questions, dtype=np.float64)
        self.log_a = np.zeros(num_questions, dtype=np.float64)
        self.initialized = np.zeros(num_questions, dtype=bool)

    # ------------------------------------------------------------------
    def init_from_take_rate(
        self,
        idx: int,
        take_rate: float,
        team_size_avg: float = 1.0,
        theta_bar: float | None = None,
    ) -> None:
        """Initialise b from empirical take rate.

        Three modes (in order of physical correctness):

        * Legacy (``team_size_avg == 1`` and ``theta_bar is None``):
          ``b = −log(p_take)``.  Implicitly assumes a single-player team
          at θ=0, which under-estimates b by ``log(n)`` for n-player
          teams under noisy-OR.

        * Noisy-OR-aware over team size (``Config.noisy_or_init=True``):
          ``b = log(n) − log(−log(1 − p_take))``.  Solves
          ``1 − exp(−n · exp(−b)) = p_take`` for an n-player team at
          θ=0, eliminating the systematic underestimation by team size.

        * Noisy-OR-aware over team size *and* roster strength
          (``Config.theta_bar_init=True``, ``theta_bar`` provided):
          ``b = log(n) + θ̄ − log(−log(1 − p_take))``.  Solves
          ``1 − exp(−n · exp(−b + a·θ̄)) = p_take`` with a=1 (init
          value), where θ̄ is the mean θ of mature players who actually
          played the question.  Removes the residual leak of pack
          hardness into player θ on strong-roster tournaments
          (Высшая лига Москвы, Гран-при etc.).
        """
        r = max(min(take_rate, 1.0 - 1e-6), 1e-6)
        n = max(float(team_size_avg), 1.0)
        neg_log_1mr = -math.log(max(1.0 - r, 1e-12))
        if theta_bar is not None:
            self.b[idx] = (
                math.log(n) + float(theta_bar) - math.log(max(neg_log_1mr, 1e-12))
            )
        elif n <= 1.0:
            self.b[idx] = -math.log(r)
        else:
            self.b[idx] = math.log(n) - math.log(max(neg_log_1mr, 1e-12))
        self.log_a[idx] = 0.0
        self.initialized[idx] = True

    # ------------------------------------------------------------------
    @property
    def a(self) -> np.ndarray:
        """Discrimination values (always > 0)."""
        return np.exp(self.log_a)
