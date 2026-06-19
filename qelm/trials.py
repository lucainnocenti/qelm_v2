"""Monte Carlo diagnostics for a fixed low-rank probability matrix.

The routines here start after a probability matrix ``P`` has already been
constructed.  They do not build POVMs, sample quantum states, or fit QELM
models.  Instead, they take deterministic SVD/covariance blocks from
``qelm.blocks``, sample scaled shot-noise matrices ``Xi`` from
``qelm.noise``, project that noise into the block basis, and summarize the
Schur-complement quantities used in the rank proof.

For actual QELM training simulations, use ``qelm.training``.  That module
works with noisy design matrices ``P_hat`` rather than directly with projected
``Xi`` blocks.
"""

from typing import Dict

import numpy as np
import pandas as pd

from .blocks import PBlocks
from .linalg import empirical_quantiles, opnorm, safe_inv
from .noise import project_noise_blocks, shot_noise_matrix
from .quantum import get_rng


def theoretical_predictors(blocks: PBlocks) -> Dict[str, float]:
    """Compute dimensionless proof predictors for one block decomposition.

    delta_shape = sqrt(M/p)+M/p, M=r+q+log(p).
    """
    r = blocks.r
    q = blocks.q_dim
    p = blocks.p_dim
    lam = blocks.lambda_min_C22
    c = blocks.c_p

    M = r + q + np.log(max(p, 2))
    delta = np.sqrt(M / p) + M / p

    out = {
        "M": M,
        "delta_shape": delta,

        # Basic invertibility predictor.
        "E_inv_delta_over_lambda": delta / lam if lam > 0 else np.inf,

        # Markov-free shape, missing the a_p factor.
        "E_Y_shape": r / (p * lam) if lam > 0 else np.inf,

        # Q residual predictor.
        "E_Q_shape": delta * (
            1.0
            + c / lam
            + (c**2) / (lam**2)
        ) if lam > 0 else np.inf,

        # Schur remainder predictor, missing Markov a_p.
        "E_R_schur_shape": np.sqrt(r) * delta * (
            1.0 / lam
            + c / (lam**2)
        ) if lam > 0 else np.inf,

        # Full Schur leading-term predictor, missing Markov a_p.
        "E_full_schur_lead_shape": np.sqrt(r) * c / lam if lam > 0 else np.inf,

        # Generic collapse variables.
        "B_general_worst_cp1": r * (q**4) * M / p,
        "B_cp_qminus1": r * (q**2) * M / p,
        "kappa_q5": p / (r * (q**5)) if r > 0 and q > 0 else np.nan,
        "kappa_q3": p / (r * (q**3)) if r > 0 and q > 0 else np.nan,
    }

    return out

def one_trial_diagnostics(
    blocks: PBlocks,
    rng: np.random.Generator | int | None = None,
    N: int = 1000,
    noise_model: str = "multinomial",
    inv_rcond: float = 1e-12,
) -> Dict[str, float]:
    """Run one scaled-noise trial and compute block diagnostics.

    ``blocks`` contains the deterministic decomposition of the exact
    probability matrix ``P``.  This function samples a scaled fluctuation
    matrix ``Xi = sqrt(N) * (P_hat - P)``, projects it into the SVD block basis,
    and records operator-norm diagnostics for the empirical Schur terms.

    noise_model:
        "multinomial" = exact finite-N multinomial normalized by sqrt(N)
        "gaussian" = large-N Gaussian approximation
    """
    rng = get_rng(rng)
    P = blocks.P
    p_dim = blocks.p_dim

    Xi = shot_noise_matrix(P, rng=rng, Nshots=N, noise=noise_model)

    noise_blocks = project_noise_blocks(Xi, blocks)
    Xi12 = noise_blocks["Xi12"]
    Xi21 = noise_blocks["Xi21"]
    Xi22 = noise_blocks["Xi22"]

    S11 = Xi12 @ Xi12.T / p_dim
    S12 = Xi12 @ Xi22.T / p_dim
    S21 = S12.T
    S22 = Xi22 @ Xi22.T / p_dim

    S22_inv, S22_used_pinv, lambda_min_S22 = safe_inv(S22, rcond=inv_rcond)

    Y_minus_I = Xi21.T @ S22_inv @ Xi21 / p_dim

    Q_term = S11 - S12 @ S22_inv @ S21
    Q_limit = blocks.Q_limit

    R_schur = (S12 @ S22_inv - blocks.C12 @ blocks.C22_inv) @ Xi21

    full_schur = S12 @ S22_inv @ Xi21
    lead_schur = blocks.C12 @ blocks.C22_inv @ Xi21

    predictors = theoretical_predictors(blocks)

    out = {
        "D_Y": opnorm(Y_minus_I),
        "D_Q": opnorm(Q_term - Q_limit),
        "D_R_schur": opnorm(R_schur),
        "D_full_schur": opnorm(full_schur),
        "D_lead_schur": opnorm(lead_schur),

        "norm_Xi21": opnorm(Xi21),
        "norm_Xi21_over_sqrt_r": opnorm(Xi21) / np.sqrt(blocks.r),

        "S22_error": opnorm(S22 - blocks.C22),
        "S12_error": opnorm(S12 - blocks.C12),
        "S11_error": opnorm(S11 - blocks.C11),

        "lambda_min_S22": lambda_min_S22,
        "S22_used_pinv": S22_used_pinv,

        "lambda_min_C22": blocks.lambda_min_C22,
        "c_p": blocks.c_p,
        "q_lambda": blocks.q_lambda,
        "q_c": blocks.q_c,

        "r": blocks.r,
        "q": blocks.q_dim,
        "p": blocks.p_dim,
        "nout": blocks.nout,
        "ntr": blocks.ntr,
        "N": N,
        "noise_model": noise_model,
    }

    out.update(predictors)
    return out

def run_trials(
    blocks: PBlocks,
    trials: int = 100,
    N: int = 1000,
    noise_model: str = "multinomial",
    seed: int | None = None,
    inv_rcond: float = 1e-12,
    progress: bool = True,
) -> pd.DataFrame:
    """Run repeated ``one_trial_diagnostics`` calls for a fixed block object."""
    rng = get_rng(seed)
    rows = []

    for k in range(trials):
        rows.append(
            one_trial_diagnostics(
                blocks=blocks,
                rng=rng,
                N=N,
                noise_model=noise_model,
                inv_rcond=inv_rcond,
            )
        )

        if progress and ((k + 1) % max(1, trials // 10) == 0):
            print(f"trial {k + 1}/{trials}")

    return pd.DataFrame(rows)

def summarize_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize trial diagnostics by dimensions, shot count, and noise model."""
    group_cols = ["r", "q", "p", "nout", "ntr", "N", "noise_model"]

    metric_cols = [
        "D_Y",
        "D_Q",
        "D_R_schur",
        "D_full_schur",
        "D_lead_schur",
        "norm_Xi21_over_sqrt_r",
        "S22_error",
        "S12_error",
        "S11_error",
        "lambda_min_S22",
    ]

    fixed_cols = [
        "lambda_min_C22",
        "c_p",
        "q_lambda",
        "q_c",
        "delta_shape",
        "E_inv_delta_over_lambda",
        "E_Y_shape",
        "E_Q_shape",
        "E_R_schur_shape",
        "E_full_schur_lead_shape",
        "B_general_worst_cp1",
        "B_cp_qminus1",
        "kappa_q5",
        "kappa_q3",
    ]

    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))

        for col in fixed_cols:
            if col in g.columns:
                row[col] = float(g[col].iloc[0])

        for col in metric_cols:
            if col in g.columns:
                qs = empirical_quantiles(g[col].values)
                for name, val in qs.items():
                    row[f"{col}_{name}"] = val

        row["S22_pinv_rate"] = float(np.mean(g["S22_used_pinv"].astype(float)))

        rows.append(row)

    return pd.DataFrame(rows)
