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
        takes_sum: int,
        n_obs: int,
        team_size_avg: float = 1.0,
        theta_bar: float | None = None,
        laplace_alpha: float = 0.0,
        b_clip_lo: float = -10.0,
        b_clip_hi: float = 10.0,
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

        Defensive corrections (see docs/error_structure_2026-04.md):

        * ``laplace_alpha`` — Beta(α, α) shrinkage of the observed
          take rate before the formula: ``r' = (k+α)/(n+2α)``.
          α=0 = legacy (no shrinkage); α=1 = classical Laplace
          (rule of succession).  Pulls extreme rates (0 or 1) away
          from the boundary in proportion to sample size, which
          stops ``b`` from running off to ±∞ on small samples with
          all-0 or all-1 takes.
        * ``b_clip_lo`` / ``b_clip_hi`` — hard clamp on the
          initialised ``b``.  SGD can still escape this range later;
          this only governs the starting point.
        """
        # Laplace shrinkage on the observed take rate.
        k = float(takes_sum)
        n_o = max(int(n_obs), 1)
        a = max(float(laplace_alpha), 0.0)
        r = (k + a) / (n_o + 2.0 * a)
        # Numerical safety after shrinkage.
        r = max(min(r, 1.0 - 1e-6), 1e-6)
        n = max(float(team_size_avg), 1.0)
        neg_log_1mr = -math.log(max(1.0 - r, 1e-12))
        if theta_bar is not None:
            b_val = (
                math.log(n) + float(theta_bar) - math.log(max(neg_log_1mr, 1e-12))
            )
        elif n <= 1.0:
            b_val = -math.log(r)
        else:
            b_val = math.log(n) - math.log(max(neg_log_1mr, 1e-12))
        self.b[idx] = max(min(b_val, float(b_clip_hi)), float(b_clip_lo))
        self.log_a[idx] = 0.0
        self.initialized[idx] = True

    # ------------------------------------------------------------------
    @property
    def a(self) -> np.ndarray:
        """Discrimination values (always > 0)."""
        return np.exp(self.log_a)
