
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def opnorm(A: np.ndarray) -> float:
    """Operator norm / spectral norm."""
    return float(np.linalg.norm(A, 2))

def frobnorm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, "fro"))

def safe_inv(A: np.ndarray, rcond: float = 1e-12) -> Tuple[np.ndarray, bool, float]:
    """
    Try to invert A. If numerically singular, return pseudoinverse.
    Returns (inverse_or_pinv, used_pinv, smallest_eigenvalue).
    """
    evals = np.linalg.eigvalsh((A + A.T) / 2)
    lam_min = float(evals[0])
    lam_max = float(evals[-1])
    threshold = rcond * max(1.0, abs(lam_max))

    if lam_min > threshold:
        return np.linalg.inv(A), False, lam_min
    return np.linalg.pinv(A, rcond=rcond), True, lam_min

def validate_probability_matrix(P: np.ndarray, atol: float = 1e-8) -> None:
    """
    Basic checks for a column-stochastic probability matrix.
    """
    if P.ndim != 2:
        raise ValueError("P must be a 2D array.")

    if np.any(P < -atol):
        raise ValueError("P has negative entries beyond tolerance.")

    col_sums = P.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=atol):
        max_err = np.max(np.abs(col_sums - 1.0))
        raise ValueError(f"Columns of P do not sum to 1. Max error: {max_err:.3e}")

def empirical_quantiles(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }

def loglog_slope(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[float, float]:
    """
    Fit log(y) = slope * log(x) + intercept.
    Only uses rows with positive finite x,y.
    """
    x = np.asarray(df[x_col], dtype=float)
    y = np.asarray(df[y_col], dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return np.nan, np.nan

    coeffs = np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    return float(slope), float(intercept)
