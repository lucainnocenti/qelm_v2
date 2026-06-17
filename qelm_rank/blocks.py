
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .linalg import opnorm, safe_inv, symmetrize, validate_probability_matrix


@dataclass
class PBlocks:
    P: np.ndarray
    r: int
    p_dim: int
    q_dim: int
    nout: int
    ntr: int

    U1: np.ndarray
    U2: np.ndarray
    V1: np.ndarray
    V2: np.ndarray

    singular_values: np.ndarray
    w: np.ndarray
    pbar: np.ndarray

    C11: np.ndarray
    C12: np.ndarray
    C22: np.ndarray
    Q_limit: np.ndarray

    C22_inv: np.ndarray
    C22_used_pinv: bool

    lambda_min_C22: float
    lambda_max_C22: float
    cond_C22: float
    c_p: float
    trace_C22: float
    pbar_inf: float
    pbar_uniform_inf: float
    q_lambda: float
    q_c: float
    q_pbar_inf: float
    q_pbar_uniform_inf: float

def svd_probability_blocks(
    P: np.ndarray,
    rank: int,
    tol: float = 1e-10,
) -> dict:
    """
    Build the SVD block basis used by the Schur-correction scaling diagnostic.

    Returns U1, U2, V1, V2, singular values, diagonal entries of
    Pi2 = I - V1 V1^T, and dimension metadata. The returned key names match
    the original notebook diagnostics for stable downstream DataFrame columns.
    """
    P = np.asarray(P, dtype=float)
    nout, ntr = P.shape
    r = int(rank)

    if r <= 0:
        raise ValueError(f"Need positive rank. Got r={r}.")
    if nout <= r:
        raise ValueError(f"Need nout > r. Got nout={nout}, r={r}.")
    if ntr <= r:
        raise ValueError(f"Need ntr > r. Got ntr={ntr}, r={r}.")

    U, s, Vt = np.linalg.svd(P, full_matrices=True)

    U1 = U[:, :r]
    U2 = U[:, r:]
    V1 = Vt.T[:, :r]
    V2 = Vt.T[:, r:]

    pi2_diag = 1.0 - np.sum(V1 * V1, axis=1)
    pi2_diag = np.clip(pi2_diag, 0.0, 1.0)

    numerical_rank = int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0

    return {
        "U1": U1,
        "U2": U2,
        "V1": V1,
        "V2": V2,
        "singular_values": s,
        "numerical_rank": numerical_rank,
        "Pi2_diag": pi2_diag,
        "r": r,
        "q": nout - r,
        "p_kernel": ntr - r,
    }


def schur_covariance_blocks(
    P: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    pi2_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute C12 and C22 without explicitly forming every Sigma_i.

    C_ab = (1/p_kernel) sum_i (Pi2)_{ii} U_a^T Sigma_i U_b, where
    Sigma_i = diag(p_i) - p_i p_i^T.
    """
    P = np.asarray(P, dtype=float)
    pi2_diag = np.asarray(pi2_diag, dtype=float)
    if pi2_diag.shape != (P.shape[1],):
        raise ValueError(
            f"pi2_diag must have shape ({P.shape[1]},). Got {pi2_diag.shape}."
        )

    p_kernel = np.sum(pi2_diag)
    if p_kernel <= 0:
        raise ValueError("The V2 projector diagonal must have positive trace.")
    w = pi2_diag / p_kernel
    sqrtw = np.sqrt(w)

    mean_probs = P @ w

    C12_diag = (U1.T * mean_probs[None, :]) @ U2
    C22_diag = (U2.T * mean_probs[None, :]) @ U2

    Y1 = U1.T @ P
    Y2 = U2.T @ P

    C12_outer = (Y1 * sqrtw[None, :]) @ (Y2 * sqrtw[None, :]).T
    C22_outer = (Y2 * sqrtw[None, :]) @ (Y2 * sqrtw[None, :]).T

    C12 = C12_diag - C12_outer
    C22 = symmetrize(C22_diag - C22_outer)
    return C12, C22


def deterministic_blocks_from_P(
    P: np.ndarray,
    r: int,
    validate: bool = True,
    inv_rcond: float = 1e-12,
) -> PBlocks:
    """
    Compute the P-induced block decomposition and deterministic covariance blocks.

    P has shape (nout, ntr).
    r should be rank(P), typically r=d^2.
    p_dim = ntr-r.
    q_dim = nout-r.
    """
    P = np.asarray(P, dtype=float)

    if validate:
        validate_probability_matrix(P)

    nout, ntr = P.shape
    svd_blocks = svd_probability_blocks(P, rank=r)
    p_dim = svd_blocks["p_kernel"]
    q_dim = svd_blocks["q"]

    U1 = svd_blocks["U1"]
    U2 = svd_blocks["U2"]
    V1 = svd_blocks["V1"]
    V2 = svd_blocks["V2"]
    s = svd_blocks["singular_values"]

    # w_i = (Pi2)_{ii} = (V2 V2^T)_{ii}. Its trace is p_dim=ntr-r.
    w = svd_blocks["Pi2_diag"]
    pbar = P @ w / p_dim

    C12, C22 = schur_covariance_blocks(P, U1, U2, w)

    A = U1.T @ P
    C11_diag = U1.T @ (pbar[:, None] * U1)
    C11_outer = (A * w[None, :]) @ A.T / p_dim
    C11 = symmetrize(C11_diag - C11_outer)

    C22_inv, C22_used_pinv, _ = safe_inv(C22, rcond=inv_rcond)

    trace_C22 = np.trace(C22)
    evals_C22 = np.linalg.eigvalsh((C22 + C22.T) / 2)
    lambda_min_C22 = float(evals_C22[0])
    lambda_max_C22 = float(evals_C22[-1])
    cond_C22 = float(lambda_max_C22 / lambda_min_C22) if lambda_min_C22 > 0 else np.inf

    c_p = opnorm(C12)

    Q_limit = C11 - C12 @ C22_inv @ C12.T

    uniform = np.ones(nout) / nout
    pbar_inf = float(np.max(np.abs(pbar)))
    pbar_uniform_inf = float(np.max(np.abs(pbar - uniform)))

    return PBlocks(
        P=P,
        r=r,
        p_dim=p_dim,
        q_dim=q_dim,
        nout=nout,
        ntr=ntr,

        U1=U1,
        U2=U2,
        V1=V1,
        V2=V2,

        singular_values=s,
        w=w,
        pbar=pbar,

        C11=C11,
        C12=C12,
        C22=C22,
        Q_limit=Q_limit,

        C22_inv=C22_inv,
        C22_used_pinv=C22_used_pinv,

        lambda_min_C22=lambda_min_C22,
        lambda_max_C22=lambda_max_C22,
        trace_C22=trace_C22,
        cond_C22=cond_C22,
        c_p=c_p,

        pbar_inf=pbar_inf,
        pbar_uniform_inf=pbar_uniform_inf,
        q_lambda=q_dim * lambda_min_C22,
        q_c=q_dim * c_p,
        q_pbar_inf=q_dim * pbar_inf,
        q_pbar_uniform_inf=q_dim * pbar_uniform_inf,
    )

def block_report(blocks: PBlocks) -> pd.DataFrame:
    """
    Compact deterministic diagnostics for one P.
    """
    r = blocks.r
    p = blocks.p_dim
    q = blocks.q_dim
    M = r + q + np.log(max(p, 2))

    delta_shape = np.sqrt(M / p) + M / p

    rows = {
        "nout": blocks.nout,
        "ntr": blocks.ntr,
        "r": r,
        "q": q,
        "p": p,
        "M=r+q+log(p)": M,
        "delta_shape": delta_shape,
        "lambda_min_C22": blocks.lambda_min_C22,
        "lambda_max_C22": blocks.lambda_max_C22,
        "cond_C22": blocks.cond_C22,
        "c_p=||C12||": blocks.c_p,
        "q*lambda_min_C22": blocks.q_lambda,
        "q*c_p": blocks.q_c,
        "||pbar||_inf": blocks.pbar_inf,
        "q*||pbar||_inf": blocks.q_pbar_inf,
        "||pbar-uniform||_inf": blocks.pbar_uniform_inf,
        "q*||pbar-uniform||_inf": blocks.q_pbar_uniform_inf,
        "C22_used_pinv": blocks.C22_used_pinv,
        "trace_C22": blocks.trace_C22
    }

    return pd.DataFrame([rows])
