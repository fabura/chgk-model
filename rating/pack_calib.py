"""Pack-level difficulty calibration for retrospective forecast display.

On elite single-pass offline events (ЧР-scale fields) sequential SGD often
stops with ``b`` ~0.2 nats below the noisy-OR init from in-tournament take
rates.  Rebaking expected takes with init ``b`` fixes the *mean* bias but is
in-sample (uses observed take rates) and must not be applied globally —
see ``scripts/forecast_diagnostic.py`` counterfactuals.

This module is shared by diagnostics and (optionally) ``website/build``.
"""
from __future__ import annotations

import statistics
from typing import Iterable, Sequence

import numpy as np

from rating.questions import QuestionState

# init.mean() − trained.mean(); tuned on II ЧР #12826 / I ЧР #11749.
PACK_B_GAP_THRESHOLD = 0.15

# Retrospective elite-offline gate (mean expected takes from trained model).
ELITE_OFFLINE_EXP_MIN = 36.0
ELITE_OFFLINE_EXP_MAX = 45.0
ELITE_OFFLINE_DELTA_MAX = -3.0


def init_b_from_take_rate(
    take_rate: float,
    *,
    team_size_avg: float = 6.0,
    theta_bar: float = 0.0,
) -> float:
    qs = QuestionState(1)
    qs.init_from_take_rate(
        0, take_rate, team_size_avg=team_size_avg, theta_bar=theta_bar
    )
    return float(qs.b[0])


def theta_bar_for_question(
    takes: Sequence[int],
    team_theta_means: Sequence[float],
) -> float:
    vals = [th for t, th in zip(takes, team_theta_means) if t == 1]
    return float(statistics.mean(vals)) if vals else 0.0


def init_b_pack(
    take_rates: np.ndarray,
    theta_bars: np.ndarray,
    *,
    team_size_avg: float = 6.0,
) -> np.ndarray:
    out = np.empty(len(take_rates), dtype=np.float64)
    for i, (tr, tb) in enumerate(zip(take_rates, theta_bars)):
        out[i] = init_b_from_take_rate(
            float(tr), team_size_avg=team_size_avg, theta_bar=float(tb)
        )
    return out


def pack_b_gap(b_trained: np.ndarray, b_init: np.ndarray) -> float:
    """init.mean() − trained.mean(); positive ⇒ trained ``b`` is too low."""
    return float(np.mean(b_init) - np.mean(b_trained))


def pack_adjust_b(
    b_trained: np.ndarray,
    b_init: np.ndarray,
    *,
    gap_threshold: float = PACK_B_GAP_THRESHOLD,
) -> tuple[np.ndarray, float, bool]:
    gap = pack_b_gap(b_trained, b_init)
    if gap < gap_threshold:
        return np.asarray(b_trained, dtype=np.float64), gap, False
    return b_trained + (b_init - b_trained), gap, True


def should_use_pack_adj_retrospective(
    *,
    mean_expected_trained: float,
    mean_delta_trained: float,
    b_gap: float,
    gap_threshold: float = PACK_B_GAP_THRESHOLD,
) -> bool:
    """Gate for display-only oracle ``b`` on finished elite offline events."""
    if b_gap < gap_threshold:
        return False
    if mean_delta_trained > ELITE_OFFLINE_DELTA_MAX:
        return False
    return ELITE_OFFLINE_EXP_MIN <= mean_expected_trained <= ELITE_OFFLINE_EXP_MAX


def team_theta_means_from_rosters(
    rosters: Iterable[tuple[Sequence[int], dict[int, float]]],
    *,
    default_theta: float = -1.0,
) -> list[float]:
    out: list[float] = []
    for pids, pmap in rosters:
        vals = [pmap.get(int(p), default_theta) for p in pids]
        out.append(float(statistics.mean(vals)) if vals else default_theta)
    return out
