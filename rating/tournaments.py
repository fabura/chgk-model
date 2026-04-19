"""
Tournament state: type-level effects and tournament residual offsets.

The total tournament shift is decomposed as:

    δ_t = μ_type[type_t] + ε_t

where μ_type captures the systematic effect of the tournament mode
(offline / sync / async) and ε_t is the residual shift for a specific
tournament. Positive values mean a harder event.
"""
from __future__ import annotations

import numpy as np


# Optional type prior for initialization (learned values override)
DELTA_OFFLINE = 0.0
DELTA_SYNC = -0.1
DELTA_ONLINE = -0.2
TYPE_OFFLINE = 0
TYPE_SYNC = 1
TYPE_ASYNC = 2


def game_type_to_idx(game_type: str) -> int:
    """Map string tournament type to a compact integer code."""
    if "async" in game_type:
        return TYPE_ASYNC
    if "sync" in game_type:
        return TYPE_SYNC
    return TYPE_OFFLINE


class TournamentState:
    """Mutable store for μ_type and ε_t."""

    __slots__ = ("num_games", "mu_type", "eps", "game_type", "game_type_idx", "_processed")

    def __init__(
        self,
        num_games: int,
        game_type: list[str] | None = None,
        use_type_prior: bool = False,
    ) -> None:
        self.num_games = num_games
        self.game_type = game_type or ["offline"] * num_games
        self.game_type_idx = np.array(
            [game_type_to_idx(gt) for gt in self.game_type],
            dtype=np.int32,
        )
        self.mu_type = np.zeros(3, dtype=np.float64)
        self.eps = np.zeros(num_games, dtype=np.float64)
        if use_type_prior:
            self.mu_type[TYPE_OFFLINE] = DELTA_OFFLINE
            self.mu_type[TYPE_SYNC] = DELTA_SYNC
            self.mu_type[TYPE_ASYNC] = DELTA_ONLINE
        self._processed: set[int] = set()

    @property
    def delta(self) -> np.ndarray:
        """Compatibility view: total tournament shifts δ_t."""
        return self.eps + self.mu_type[self.game_type_idx]

    def total_delta(self, g: int) -> float:
        """Return μ_type[type_g] + ε_g for a single tournament."""
        type_idx = int(self.game_type_idx[g]) if g < len(self.game_type_idx) else TYPE_OFFLINE
        return float(self.eps[g] + self.mu_type[type_idx])

    def update(
        self,
        g: int,
        dL_ddelta: float,
        *,
        eta_mu: float,
        eta_eps: float,
        weight_mu: float,
        weight_eps: float,
        reg_mu: float = 0.0,
        reg_eps: float = 0.0,
    ) -> None:
        """Update type effect and residual using the same gradient as δ."""
        type_idx = int(self.game_type_idx[g]) if g < len(self.game_type_idx) else TYPE_OFFLINE

        if type_idx != TYPE_OFFLINE:
            mu_val = self.mu_type[type_idx]
            mu_val += weight_mu * eta_mu * dL_ddelta
            if reg_mu > 0.0:
                mu_val *= max(0.0, 1.0 - eta_mu * reg_mu)
            self.mu_type[type_idx] = max(min(mu_val, 10.0), -10.0)

        eps_val = self.eps[g] + weight_eps * eta_eps * dL_ddelta
        if reg_eps > 0.0:
            eps_val *= max(0.0, 1.0 - eta_eps * reg_eps)
        self.eps[g] = max(min(eps_val, 10.0), -10.0)
        self._processed.add(g)

    def center(
        self,
        indices: list[int] | None = None,
        weights: np.ndarray | None = None,
    ) -> None:
        """Center residuals ε_t within each mode over the given indices.

        If ``weights`` is provided (per-game positive weights, e.g. number of
        observations), use a *weighted* mean for centering.  This prevents a
        single small/weak tournament from setting the bar for a week — a
        large/hard tournament with thousands of obs will dominate the
        within-week mean and keep its non-trivial residual.
        """
        if indices is None or len(indices) == 0:
            return
        idx = np.asarray(indices, dtype=np.int64)
        for type_idx in (TYPE_OFFLINE, TYPE_SYNC, TYPE_ASYNC):
            mask = self.game_type_idx[idx] == type_idx
            if not np.any(mask):
                continue
            typed_idx = idx[mask]
            if weights is not None:
                w = np.asarray(weights, dtype=np.float64)[mask]
                wsum = float(w.sum())
                if wsum > 0.0:
                    mean_eps = float((self.eps[typed_idx] * w).sum() / wsum)
                else:
                    mean_eps = float(np.mean(self.eps[typed_idx]))
            else:
                mean_eps = float(np.mean(self.eps[typed_idx]))
            self.eps[typed_idx] -= mean_eps
