"""
Hyperparameter tuning for sequential rating.

Grid or random search over eta0, rho, w_online via time-based backtest.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from rating.backtest import backtest
from rating.engine import Config


@dataclass
class TuneResult:
    """Single trial result."""

    config: Config
    logloss: float
    brier: float
    auc: float
    n_test_obs: int
    n_train_obs: int


def _default_grid() -> list[dict[str, Any]]:
    """Default grid: eta0 × rho × w_online (~36 trials)."""
    eta0_vals = [0.02, 0.05, 0.07, 0.1]
    rho_vals = [0.997, 0.999, 0.9995]
    w_online_vals = [0.5, 0.7, 0.8]
    configs = []
    for eta0 in eta0_vals:
        for rho in rho_vals:
            for w in w_online_vals:
                configs.append({
                    "eta0": eta0,
                    "rho": rho,
                    "w_online": w,
                    "use_tournament_delta": True,
                    "use_delta_type_prior": False,
                })
    return configs


def random_search(
    n_trials: int = 24,
    eta0_range: tuple[float, float] = (0.02, 0.12),
    rho_range: tuple[float, float] = (0.996, 0.9998),
    w_online_range: tuple[float, float] = (0.5, 0.9),
) -> list[dict[str, Any]]:
    """Generate random configs for search."""
    configs = []
    for _ in range(n_trials):
        eta0 = random.uniform(*eta0_range)
        rho = random.uniform(*rho_range)
        w = random.uniform(*w_online_range)
        configs.append({
            "eta0": round(eta0, 4),
            "rho": round(rho, 4),
            "w_online": round(w, 2),
            "use_tournament_delta": True,
            "use_delta_type_prior": False,
        })
    return configs


def tune(
    arrays: dict[str, np.ndarray],
    maps,
    *,
    configs: Optional[list[dict[str, Any]]] = None,
    grid: bool = True,
    n_trials: int = 24,
    test_fraction: float = 0.2,
    metric: str = "logloss",
    verbose: bool = True,
) -> list[TuneResult]:
    """Run hyperparameter search via backtest.

    Parameters
    ----------
    arrays, maps : data
    configs : optional list of dicts
        If provided, use these configs. Else use grid or random.
    grid : bool
        If True and configs is None, use default grid. Else random search.
    n_trials : int
        Number of random trials when grid=False.
    test_fraction : float
        Fraction of tournaments for test set.
    metric : str
        Primary metric: "logloss", "brier", or "auc" (higher is better for auc).
    verbose : bool
        Print progress.

    Returns
    -------
    list[TuneResult]
        Sorted by metric (best first). Lower is better for logloss/brier.
    """
    if configs is None:
        configs = _default_grid() if grid else random_search(n_trials=n_trials)

    results: list[TuneResult] = []
    n = len(configs)

    for i, kw in enumerate(configs):
        cfg = Config(
            eta0=kw.get("eta0", 0.05),
            rho=kw.get("rho", 0.999),
            w_online=kw.get("w_online", 0.7),
            use_tournament_delta=kw.get("use_tournament_delta", True),
            use_delta_type_prior=kw.get("use_delta_type_prior", False),
        )
        if verbose:
            print(
                f"Trial {i + 1}/{n} | "
                f"eta0={cfg.eta0:.4f} rho={cfg.rho:.4f} w_online={cfg.w_online:.2f}",
                end=" ... ",
                flush=True,
            )
        metrics = backtest(
            arrays,
            maps,
            cfg,
            test_fraction=test_fraction,
            verbose=False,
        )
        r = TuneResult(
            config=cfg,
            logloss=metrics["logloss"],
            brier=metrics["brier"],
            auc=metrics["auc"],
            n_test_obs=metrics.get("n_test_obs", 0),
            n_train_obs=metrics.get("n_train_obs", 0),
        )
        results.append(r)
        if verbose:
            print(
                f"logloss={r.logloss:.4f} Brier={r.brier:.4f} AUC={r.auc:.4f}"
            )

    # Sort: lower is better for logloss/brier, higher for auc
    if metric == "auc":
        results.sort(key=lambda x: -x.auc)
    else:
        key_attr = "logloss" if metric == "logloss" else "brier"
        results.sort(key=lambda x: getattr(x, key_attr))

    return results
