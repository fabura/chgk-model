"""
Evaluation: logloss, Brier score, AUC, calibration curve.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def logloss(y_true: torch.Tensor, p_pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Binary cross-entropy: -[ y*log(p) + (1-y)*log(1-p) ]."""
    p = p_pred.clamp(min=eps, max=1.0 - eps)
    return -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p)).mean()


def weighted_logloss(
    y_true: torch.Tensor,
    p_pred: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Weighted BCE with normalized weighted mean."""
    p = p_pred.clamp(min=eps, max=1.0 - eps)
    ll = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
    w = weights.clamp(min=0.0)
    w_sum = w.sum().clamp(min=eps)
    return (w * ll).sum() / w_sum


def brier_score(y_true: torch.Tensor, p_pred: torch.Tensor) -> torch.Tensor:
    """Brier: mean (p - y)^2."""
    return ((p_pred - y_true) ** 2).mean()


def auc_roc(y_true: torch.Tensor, p_pred: torch.Tensor) -> float:
    """Area under ROC curve (sklearn)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    y = y_true.detach().cpu().numpy()
    p = p_pred.detach().cpu().numpy()
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def calibration_curve(
    y_true: torch.Tensor,
    p_pred: torch.Tensor,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: bin_centers, mean_predicted_value, fraction_of_positives (per bin).
    """
    try:
        from sklearn.calibration import calibration_curve as sk_calibration_curve
    except ImportError:
        return np.array([]), np.array([]), np.array([])
    y = y_true.detach().cpu().numpy()
    p = p_pred.detach().cpu().numpy()
    frac_pos, mean_pred = sk_calibration_curve(y, p, n_bins=n_bins)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, mean_pred, frac_pos


def plot_calibration(
    y_true: torch.Tensor,
    p_pred: torch.Tensor,
    n_bins: int = 10,
    path: Optional[str] = None,
) -> None:
    """Plot calibration curve (optional matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    bc, mean_pred, frac_pos = calibration_curve(y_true, p_pred, n_bins=n_bins)
    if len(bc) == 0:
        return
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.plot(mean_pred, frac_pos, "s-", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.close()
