
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


def generate_multinomial_Xi(
    P: np.ndarray,
    N: int,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Generate Xi with columns Xi_i = sqrt(N) * (counts_i/N - p_i),
    counts_i ~ Multinomial(N, p_i).
    """
    rng = get_rng(rng)
    nout, ntr = P.shape
    Xi = np.empty_like(P, dtype=float)

    for i in range(ntr):
        probs = P[:, i]
        # Small numerical cleanup.
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()

        counts = rng.multinomial(N, probs)
        Xi[:, i] = np.sqrt(N) * (counts / N - probs)

    return Xi

def generate_gaussian_Xi(
    P: np.ndarray,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Generate Gaussian Xi_i ~ N(0, diag(p_i)-p_i p_i^T).

    Efficient construction:
    x = sqrt(p_i) * g,
    y = x - p_i * sum(x).
    Then Cov(y)=diag(p_i)-p_i p_i^T.
    """
    rng = get_rng(rng)
    nout, ntr = P.shape
    Xi = np.empty_like(P, dtype=float)

    G = rng.standard_normal(size=P.shape)

    for i in range(ntr):
        probs = P[:, i]
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()

        x = np.sqrt(probs) * G[:, i]
        Xi[:, i] = x - probs * np.sum(x)

    return Xi

def shot_noise_matrix(
    P: np.ndarray,
    rng: np.random.Generator | int | None = None,
    Nshots: int = 10_000,
    noise: str = "gaussian",
) -> np.ndarray:
    """
    Generate the scaled shot-noise matrix Xi.

    Xi_i = sqrt(Nshots) * (p_hat_i - p_i). The Gaussian option samples the
    large-shot scaled limit and ignores Nshots.

    For noise == "centered_gaussian", each column is sampled independently from
    N(0, Sigma_bar), where Sigma_bar is the average of the column covariance
    matrices diag(p_i) - p_i p_i^T.
    """
    rng = get_rng(rng)
    P = np.maximum(np.asarray(P, dtype=float), 0.0)

    col_sums = P.sum(axis=0, keepdims=True)
    if np.any(col_sums <= 0):
        raise ValueError("Each column of P must have positive total probability mass.")

    P = P / col_sums

    if noise == "gaussian":
        return generate_gaussian_Xi(P, rng=rng)

    if noise == "multinomial":
        return generate_multinomial_Xi(P, N=Nshots, rng=rng)

    if noise == "centered_gaussian":
        nout, ntr = P.shape

        # Average covariance:
        # Sigma_bar = mean_i [diag(p_i) - p_i p_i^T]
        mean_probs = P.mean(axis=1)
        cov_matrix = np.diag(mean_probs) - (P @ P.T) / ntr

        # Symmetrize to remove tiny numerical asymmetries.
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

        # Sample ntr independent columns from N(0, cov_matrix).
        Xi = rng.multivariate_normal(
            mean=np.zeros(nout),
            cov=cov_matrix,
            size=ntr,
            check_valid="warn",
        ).T

        return Xi

    raise ValueError(f"Unknown noise type: {noise!r}")

# def shot_noise_matrix(
#     P: np.ndarray,
#     rng: np.random.Generator,
#     Nshots: int = 10_000,
#     noise: str = "gaussian",
# ) -> np.ndarray:
#     """
#     Generate the scaled shot-noise matrix Xi.

#     Xi_i = sqrt(Nshots) * (p_hat_i - p_i). The Gaussian option samples the
#     large-shot scaled limit and ignores Nshots.
#     """
#     P = np.maximum(np.asarray(P, dtype=float), 0.0)
#     P = P / P.sum(axis=0, keepdims=True)

#     if noise == "gaussian":
#         return generate_gaussian_Xi(P, rng=rng)

#     if noise == "multinomial":
#         return generate_multinomial_Xi(P, N=Nshots, rng=rng)

#     if noise == 'centered_gaussian':
#         # in this case sample each column as a centered Gaussian distribution with covariance matrix
#         # that's the haar average of the covariance matrices, that is,
#         # \bar\Sigma_{ab} = tr(\mu_a)/d \delta_{ab} - \frac{tr(\mu_a)tr(\mu_b)+tr(\mu_a \mu_b)}{d(d+1)}
#         # where \mu_a is the a-th measurement operator, and d is the dimension of the Hilbert space.
#         nout, ntr = P.shape
#         Xi = np.empty_like(P, dtype=float)
#         for i in range(ntr):
#             probs = P[:, i]
#             probs = np.maximum(probs, 0.0)
#             probs = probs / probs.sum()

#             # Compute the covariance matrix for the centered Gaussian noise
#             d = nout  # Assuming the dimension of the Hilbert space is equal to nout
#             cov_matrix = np.diag(probs) - np.outer(probs, probs)

#             # Sample from the centered Gaussian distribution
#             Xi[:, i] = rng.multivariate_normal(mean=np.zeros(nout), cov=cov_matrix)

#     raise ValueError("noise must be either 'gaussian' or 'multinomial'.")

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
