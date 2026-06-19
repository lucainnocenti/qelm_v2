"""Shot-noise models for probability matrices.

This module distinguishes two related objects:

* ``Xi`` is the scaled fluctuation, ``Xi = sqrt(N) * (P_hat - P)``.  The
  block-diagnostic code in :mod:`qelm.trials` projects ``Xi`` into SVD
  blocks and studies its Schur-complement quantities.
* ``P_hat`` is the noisy empirical probability matrix used as a design matrix
  in actual QELM least-squares training.  Use ``noisy_probability_matrix`` for
  that object.

All probability-matrix shot-noise routing goes through ``sample_shot_noise``.
The public helpers ``shot_noise_matrix`` and ``noisy_probability_matrix`` only
select the requested output representation.
"""

from typing import Dict

import numpy as np

from .blocks import PBlocks
from .quantum import get_rng


def clean_probability_vector(p: np.ndarray) -> np.ndarray:
    """Numerically clean a probability vector before multinomial sampling."""
    p = np.maximum(np.asarray(p, dtype=float), 0.0)
    total = p.sum()
    if total <= 0:
        raise ValueError("Probability vector has non-positive total mass.")
    return p / total


def clean_probability_matrix_columns(P: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Validate and normalize probability-matrix columns."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError("P must be a 2D array.")
    if not np.all(np.isfinite(P)):
        raise ValueError("P contains non-finite entries.")
    if np.any(P < -atol):
        raise ValueError("P has negative entries beyond tolerance.")

    P = np.where((P < 0.0) & (P >= -atol), 0.0, P)
    col_sums = P.sum(axis=0, keepdims=True)
    if np.any(col_sums <= 0):
        raise ValueError("Each column of P must have positive total probability mass.")
    return P / col_sums


def _validate_shot_count(N: int, *, name: str = "N") -> int:
    if int(N) != N or N <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(N)


def _sample_multinomial_probability_matrix(
    P: np.ndarray,
    *,
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    nout, ntr = P.shape
    P_hat = np.empty((nout, ntr), dtype=float)

    for i in range(ntr):
        P_hat[:, i] = rng.multinomial(N, P[:, i]) / N

    return P_hat


def sample_finite_shot_probability_matrix(
    P: np.ndarray,
    N: int,
    rng: np.random.Generator | int | None = None,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """Sample empirical multinomial probabilities from exact columns.

    Returns ``P_hat[:, i] = Multinomial(N, P[:, i]) / N`` for each column.
    This is the single implementation that draws finite-shot multinomial
    counts; scaled multinomial fluctuations are derived from this same draw.
    """
    N = _validate_shot_count(N)
    rng = get_rng(rng)
    P = clean_probability_matrix_columns(P, atol=atol)
    return _sample_multinomial_probability_matrix(P, N=N, rng=rng)


def generate_multinomial_Xi(
    P: np.ndarray,
    N: int,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate scaled finite-shot multinomial fluctuations.

    Returns a matrix ``Xi`` with columns
    ``Xi_i = sqrt(N) * (counts_i / N - p_i)``, where
    ``counts_i ~ Multinomial(N, p_i)``.  This is not the sampled probability
    matrix itself; use ``noisy_probability_matrix`` when the desired output is
    ``P_hat = counts / N``.
    """
    N = _validate_shot_count(N)
    rng = get_rng(rng)
    P = clean_probability_matrix_columns(P)
    P_hat = _sample_multinomial_probability_matrix(P, N=N, rng=rng)
    return np.sqrt(N) * (P_hat - P)


def generate_gaussian_Xi(
    P: np.ndarray,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate scaled Gaussian shot-noise fluctuations.

    Generates columns ``Xi_i ~ N(0, diag(p_i)-p_i p_i^T)``, the large-shot
    limit of ``sqrt(N) * (P_hat_i - p_i)``.  The result is independent of
    ``N`` because the ``sqrt(N)`` scaling is already included.
    """
    rng = get_rng(rng)
    P = clean_probability_matrix_columns(P)
    return _generate_gaussian_Xi_from_clean(P, rng=rng)


def _generate_centered_gaussian_Xi_from_clean(
    P: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    nout, ntr = P.shape
    # Average covariance:
    # Sigma_bar = mean_i [diag(p_i) - p_i p_i^T]
    mean_probs = P.mean(axis=1)
    cov_matrix = np.diag(mean_probs) - (P @ P.T) / ntr
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    return rng.multivariate_normal(
        mean=np.zeros(nout),
        cov=cov_matrix,
        size=ntr,
        check_valid="warn",
    ).T

def _generate_gaussian_Xi_from_clean(
    P: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    nout, ntr = P.shape
    Xi = np.empty_like(P, dtype=float)
    G = rng.standard_normal(size=P.shape)
    for i in range(ntr):
        probs = P[:, i]
        x = np.sqrt(probs) * G[:, i]
        Xi[:, i] = x - probs * np.sum(x)

    return Xi

def sample_shot_noise(
    P: np.ndarray,
    rng: np.random.Generator | int | None = None,
    *,
    Nshots: int = 10_000,
    noise: str = "gaussian",
    output: str = "xi",
    atol: float = 1e-12,
) -> np.ndarray:
    """Route a probability matrix through one supported shot-noise model.

    ``output="xi"`` returns the scaled fluctuation
    ``Xi = sqrt(Nshots) * (P_hat - P)``. ``output="probability"`` returns the
    noisy design matrix ``P_hat``.
    """
    Nshots = _validate_shot_count(Nshots, name="Nshots")
    rng = get_rng(rng)
    P = clean_probability_matrix_columns(P, atol=atol)

    if output not in {"xi", "probability"}:
        raise ValueError("output must be 'xi' or 'probability'.")

    if noise == "multinomial":
        P_hat = _sample_multinomial_probability_matrix(P, N=Nshots, rng=rng)
        if output == "probability":
            return P_hat
        return np.sqrt(Nshots) * (P_hat - P)

    if noise == "gaussian":
        Xi = _generate_gaussian_Xi_from_clean(P, rng=rng)
    elif noise == "centered_gaussian":
        Xi = _generate_centered_gaussian_Xi_from_clean(P, rng=rng)
    else:
        raise ValueError(f"Unknown noise type: {noise!r}")

    if output == "xi":
        return Xi
    return P + Xi / np.sqrt(Nshots)


def shot_noise_matrix(
    P: np.ndarray,
    rng: np.random.Generator | int | None = None,
    Nshots: int = 10_000,
    noise: str = "gaussian",
) -> np.ndarray:
    """Generate the scaled shot-noise matrix ``Xi``."""
    return sample_shot_noise(
        P,
        rng=rng,
        Nshots=Nshots,
        noise=noise,
        output="xi",
    )


def noisy_probability_matrix(
    P: np.ndarray,
    rng: np.random.Generator | int | None = None,
    *,
    Nshots: int,
    noise: str = "gaussian",
) -> np.ndarray:
    """Sample the noisy probability/design matrix used by training."""
    return sample_shot_noise(
        P,
        rng=rng,
        Nshots=Nshots,
        noise=noise,
        output="probability",
    )


def project_noise_blocks(Xi: np.ndarray, blocks: PBlocks) -> Dict[str, np.ndarray]:
    """
    Compute Xi12, Xi21, Xi22 induced by P.

    Xi12 = U1^T Xi V2
    Xi21 = U2^T Xi V1
    Xi22 = U2^T Xi V2
    """
    U1, U2, V1, V2 = blocks.U1, blocks.U2, blocks.V1, blocks.V2

    Xi12 = U1.T @ Xi @ V2
    Xi21 = U2.T @ Xi @ V1
    Xi22 = U2.T @ Xi @ V2

    return {
        "Xi12": Xi12,
        "Xi21": Xi21,
        "Xi22": Xi22,
    }
