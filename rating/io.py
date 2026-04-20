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
    # Tournament-level effects: δ_t = μ_type[type_t] + ε_t.
    mu_type: Optional[np.ndarray] = None      # shape (3,) for offline/sync/async
    eps: Optional[np.ndarray] = None          # shape (num_games,)
    game_type_idx: Optional[np.ndarray] = None  # shape (num_games,) int8 with 0/1/2
    # Team-size effect, indexed by team size; zeroed at anchor.
    delta_size: Optional[np.ndarray] = None
    team_size_anchor: Optional[int] = None
    # Position-in-tour effect, indexed by (q_in_tournament % len(delta_pos)).
    delta_pos: Optional[np.ndarray] = None
    pos_anchor: Optional[int] = None
    # Yearly gauge re-centering events (ord_day, delta_applied).  Used to
    # retroactively shift historical θ rows into a single (final) gauge.
    recenter_ord: Optional[np.ndarray] = None
    recenter_delta: Optional[np.ndarray] = None

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
    def _get(name):
        return z[name] if name in z else None
    def _scalar(name):
        return int(z[name][0]) if name in z else None
    return RatingResults(
        player_id=z["player_id"],
        theta=z["theta"],
        games=z["games"],
        question_tid=z["question_tid"],
        question_qi=z["question_qi"],
        b=z["b"],
        a=z["a"],
        canonical_q_idx=_get("canonical_q_idx"),
        history_player_id=_get("history_player_id"),
        history_game_id=_get("history_game_id"),
        history_theta=_get("history_theta"),
        mu_type=_get("mu_type"),
        eps=_get("eps"),
        game_type_idx=_get("game_type_idx"),
        delta_size=_get("delta_size"),
        team_size_anchor=_scalar("team_size_anchor"),
        delta_pos=_get("delta_pos"),
        pos_anchor=_scalar("pos_anchor"),
        recenter_ord=_get("recenter_ord"),
        recenter_delta=_get("recenter_delta"),
    )
