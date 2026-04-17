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
    def init_from_take_rate(self, idx: int, take_rate: float) -> None:
        """Initialise from empirical take rate: b ≈ −log(p_take), a = 1."""
        r = max(min(take_rate, 1.0 - 1e-6), 1e-6)
        self.b[idx] = -math.log(r)
        self.log_a[idx] = 0.0
        self.initialized[idx] = True

    # ------------------------------------------------------------------
    @property
    def a(self) -> np.ndarray:
        """Discrimination values (always > 0)."""
        return np.exp(self.log_a)
