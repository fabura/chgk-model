"""Tournament-type encoding helpers.

The tournament-level shift (μ_type + ε_t) was removed in 2026-04 after
an ablation showed those 8 746 parameters were net-negative for
backtest quality (logloss −0.0043, AUC +0.0044 when removed).  This
module now only exposes the small int code used to weight per-mode
update strength in `rating.engine`.
"""
from __future__ import annotations


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
