"""Load rating results from compact .npz format."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RatingResults:
    """Loaded results from --results_npz."""

    player_id: np.ndarray
    theta: np.ndarray
    games: np.ndarray
    question_tid: np.ndarray
    question_qi: np.ndarray
    b: np.ndarray
    a: np.ndarray
    canonical_q_idx: Optional[np.ndarray]
    history_player_id: Optional[np.ndarray] = None
    history_game_id: Optional[np.ndarray] = None
    history_theta: Optional[np.ndarray] = None

    def theta_for_player(self, player_id: int) -> float:
        """Get θ for player_id (or nan if not found)."""
        idx = np.where(self.player_id == player_id)[0]
        return float(self.theta[idx[0]]) if len(idx) else float("nan")

    def b_a_for_question(self, raw_idx: int) -> tuple[float, float]:
        """Get (b, a) for raw question index."""
        cq = self.canonical_q_idx
        qi = int(cq[raw_idx]) if cq is not None else raw_idx
        return float(self.b[qi]), float(self.a[qi])


def load_results_npz(path: str | Path) -> RatingResults:
    """Load results from .npz file saved by --results_npz."""
    z = np.load(path, allow_pickle=True)
    return RatingResults(
        player_id=z["player_id"],
        theta=z["theta"],
        games=z["games"],
        question_tid=z["question_tid"],
        question_qi=z["question_qi"],
        b=z["b"],
        a=z["a"],
        canonical_q_idx=z["canonical_q_idx"] if "canonical_q_idx" in z else None,
        history_player_id=z["history_player_id"] if "history_player_id" in z else None,
        history_game_id=z["history_game_id"] if "history_game_id" in z else None,
        history_theta=z["history_theta"] if "history_theta" in z else None,
    )
