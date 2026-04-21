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
    eta0_vals = [0.04, 0.07, 0.10]
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
                })
    return configs


def random_search(
    n_trials: int = 24,
    eta0_range: tuple[float, float] = (0.03, 0.12),
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

    # Identify which fields are actually being swept (varying across
    # configs) so we print a concise summary rather than every default.
    swept_keys: list[str] = []
    if configs:
        all_keys = set().union(*(set(kw.keys()) for kw in configs))
        for k in sorted(all_keys):
            vals = {kw.get(k) for kw in configs}
            if len(vals) > 1:
                swept_keys.append(k)

    for i, kw in enumerate(configs):
        # Build Config from defaults, then override with whatever the
        # trial supplied.  Any field of ``Config`` may be tuned this way
        # (eta0, rho, w_online, w_sync, eta_teammate, recenter_target, …).
        try:
            cfg = Config(**kw)
        except TypeError as e:
            raise TypeError(
                f"Tune config has unknown field(s): {kw!r}.  "
                f"Make sure every key matches a Config attribute. ({e})"
            )
        if verbose:
            if swept_keys:
                summary = " ".join(
                    f"{k}={kw.get(k, getattr(cfg, k))}"
                    for k in swept_keys
                )
            else:
                summary = (
                    f"eta0={cfg.eta0:.4f} rho={cfg.rho:.4f} "
                    f"w_online={cfg.w_online:.2f}"
                )
            print(
                f"Trial {i + 1}/{n} | {summary}",
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
