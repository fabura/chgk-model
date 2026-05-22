"""ChGK sequential online rating system.

Re-exports are lazy (PEP 562 ``__getattr__``) so that importing a single
lightweight submodule (e.g. ``rating.simulate``, used at website request
time with only numpy) does not pull in the training-only dependency
graph (``numba``, ``sklearn``, ``psycopg2``, …).
"""
from __future__ import annotations

__all__ = [
    "Config",
    "SequentialResult",
    "run_sequential",
    "backtest",
    "RatingResults",
    "load_results_npz",
]


def __getattr__(name: str):
    if name in {"Config", "SequentialResult", "run_sequential"}:
        from rating.engine import Config, SequentialResult, run_sequential

        return {"Config": Config, "SequentialResult": SequentialResult, "run_sequential": run_sequential}[name]
    if name == "backtest":
        from rating.backtest import backtest

        return backtest
    if name in {"RatingResults", "load_results_npz"}:
        from rating.io import RatingResults, load_results_npz

        return {"RatingResults": RatingResults, "load_results_npz": load_results_npz}[name]
    raise AttributeError(f"module 'rating' has no attribute {name!r}")
