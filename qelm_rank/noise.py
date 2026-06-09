
from typing import Dict

import numpy as np

from .blocks import PBlocks


def generate_multinomial_Xi(
    P: np.ndarray,
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate Xi with columns Xi_i = sqrt(N) * (counts_i/N - p_i),
    counts_i ~ Multinomial(N, p_i).
    """
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
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate Gaussian Xi_i ~ N(0, diag(p_i)-p_i p_i^T).

    Efficient construction:
    x = sqrt(p_i) * g,
    y = x - p_i * sum(x).
    Then Cov(y)=diag(p_i)-p_i p_i^T.
    """
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
