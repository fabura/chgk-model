"""
Rating decay: slow multiplicative drift toward zero.

Two flavours are supported:

* ``apply_decay(theta, rho)`` — global per-tournament decay.  Cheap,
  but unfair: a player who is in many tournaments per week gets
  punished as much as a long-inactive one.
* ``apply_calendar_decay(theta, last_seen_ordinal, current_ordinal,
  pids, rho_calendar, period_days)`` — per-player decay proportional to
  the number of days since the player's last appearance.  This matches
  the intuition that θ is a *current strength* estimate and should
  decay only because of real-world time passing, not because the
  dataset happens to contain many parallel tournaments in the same
  week.
"""
from __future__ import annotations

import math

import numpy as np


def apply_decay(theta: np.ndarray, rho: float) -> None:
    """θ ← ρ · θ  (in-place)."""
    theta *= rho


def apply_calendar_decay(
    theta: np.ndarray,
    last_seen_ordinal: np.ndarray,
    current_ordinal: int,
    pids: np.ndarray,
    rho_calendar: float,
    period_days: float = 7.0,
) -> None:
    """Per-player calendar-based decay.

    For each player ``k`` in ``pids`` whose ``last_seen_ordinal[k] >= 0``::

        delta_days = max(0, current_ordinal - last_seen_ordinal[k])
        theta[k]  *= rho_calendar ** (delta_days / period_days)

    Players with ``last_seen_ordinal[k] == -1`` (never seen) are left
    untouched.  ``last_seen_ordinal`` is updated to ``current_ordinal``
    by the caller after the tournament is processed (we keep this side
    effect out of the decay function so the caller can choose when to
    advance it).

    ``period_days`` controls the natural unit of ``rho_calendar``
    (default: per-week).
    """
    if rho_calendar >= 1.0 or current_ordinal < 0:
        return
    log_rho = math.log(rho_calendar)
    for pidx in pids:
        last = int(last_seen_ordinal[pidx])
        if last < 0:
            continue
        delta_days = current_ordinal - last
        if delta_days <= 0:
            continue
        theta[pidx] *= math.exp(log_rho * (delta_days / period_days))
