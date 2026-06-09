
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .linalg import opnorm, safe_inv, validate_probability_matrix


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
    p_dim = ntr - r
    q_dim = nout - r

    if p_dim <= 0:
        raise ValueError("Need ntr > r, so p_dim=ntr-r must be positive.")
    if q_dim <= 0:
        raise ValueError("Need nout > r, so q_dim=nout-r must be positive.")

    U, s, Vt = np.linalg.svd(P, full_matrices=True)

    U1 = U[:, :r]
    U2 = U[:, r:]
    V1 = Vt.T[:, :r]
    V2 = Vt.T[:, r:]

    # Diagonal of Pi2 = V2 V2^T. Equivalent to 1 - rownorm(V1)^2,
    # but computing from V2 is explicit.
    # here w_i = (Pi2)_{ii} = (V2 V2^T)_{ii} = sum_j V2_{ij}^2.
    w = np.sum(V2**2, axis=1)
    # here pbar_i = (P Pi2)_{ii} = sum_j P_{ij} w_j.
    pbar = P @ w / p_dim

    # Dbar times a matrix X is pbar[:, None] * X.
    Dbar_U1 = pbar[:, None] * U1
    Dbar_U2 = pbar[:, None] * U2

    C22 = U2.T @ Dbar_U2
    C12 = U1.T @ Dbar_U2

    # C11 = (1/p) sum_i w_i U1^T Sigma_i U1.
    # Sigma_i = diag(p_i)-p_i p_i^T.
    #
    # First term:
    # (1/p) sum_i w_i U1^T diag(p_i) U1
    # = U1^T diag(pbar) U1.
    #
    # Second term:
    # (1/p) sum_i w_i (U1^T p_i)(U1^T p_i)^T.
    A = U1.T @ P  # shape (r, ntr)
    second = (A * w[None, :]) @ A.T / p_dim
    C11 = U1.T @ Dbar_U1 - second

    C22_inv, C22_used_pinv, lam_min_numerical = safe_inv(C22, rcond=inv_rcond)

    trace_C22 = np.trace(C22)
    evals_C22 = np.linalg.eigvalsh((C22 + C22.T) / 2)
    lambda_min_C22 = float(evals_C22[0])
    lambda_max_C22 = float(evals_C22[-1])
    # print the rank of C22 for debugging; should be q_dim.
    rank_C22 = np.sum(np.linalg.eigvalsh(C22) > inv_rcond * np.max(np.abs(np.linalg.eigvalsh(C22))))
    print('rank_C22:', rank_C22, 'q_dim:', q_dim, 'lam_max:', lambda_max_C22, 'bound:', rank_C22 * lambda_max_C22, 'trace:', trace_C22)
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
