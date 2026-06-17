
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def opnorm(A: np.ndarray) -> float:
    """Operator norm / spectral norm."""
    if A.size == 0:
        return 0.0
    return float(np.linalg.norm(A, 2))

def frobnorm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, "fro"))

def symmetrize(A: np.ndarray) -> np.ndarray:
    """Numerically symmetrize a real square matrix."""
    return 0.5 * (A + A.T)

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

def psd_solve(
    A: np.ndarray,
    B: np.ndarray,
    rcond: Optional[float] = None,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve A X = B for symmetric positive-semidefinite A by eigendecomposition.

    Eigenvalues below rcond * lambda_max are dropped unless a positive ridge is
    supplied, in which case all eigendirections are kept with eigenvalues
    shifted by ridge.
    """
    A = symmetrize(A)
    eigvals, Q = np.linalg.eigh(A)

    if ridge > 0:
        inv_eigs = 1.0 / (eigvals + ridge)
        kept = np.ones_like(eigvals, dtype=bool)
    else:
        if rcond is None:
            # this should be the default cutoff used by np.linalg.pinv or lstsq for the given dtype and shape
            rcond_eff = np.finfo(eigvals.dtype).eps * max(A.shape)
        else:
            rcond_eff = float(rcond)
        lam_max = np.max(eigvals)
        cutoff = rcond_eff * lam_max
        kept = eigvals > cutoff
        inv_eigs = np.zeros_like(eigvals)
        inv_eigs[kept] = 1.0 / eigvals[kept]

    X = Q @ (inv_eigs[:, None] * (Q.T @ B))
    return X, eigvals, kept


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

def quantile_suffix(q: float) -> str:
    """
    Stable column suffix for a quantile.

    Examples: 0.25 -> q25, 0.025 -> q2p5.
    """
    pct = 100.0 * float(q)
    if np.isclose(pct, round(pct)):
        return f"q{int(round(pct))}"
    return f"q{pct:g}".replace(".", "p")

def distribution_summary(
    x: np.ndarray,
    quantiles: Sequence[float] = (0.25, 0.75),
) -> Dict[str, float]:
    """
    Mean, median, and user-selected quantiles for finite numeric samples.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        out = {"mean": np.nan, "median": np.nan}
        out.update({quantile_suffix(q): np.nan for q in quantiles})
        return out

    out = {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
    }
    for q in quantiles:
        out[quantile_suffix(q)] = float(np.quantile(x, q))
    return out

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

def loglog_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit log(y) = slope * log(x) + intercept using positive finite entries.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return np.nan, np.nan

    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1)
    return float(slope), float(intercept)
