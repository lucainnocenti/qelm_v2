
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .blocks import PBlocks, block_report, deterministic_blocks_from_P
from .plotting import plot_metric_vs_predictors
from .quantum import (
    generate_haar_random_pure_states,
    generate_random_rank1_povm,
    probability_matrix_from_povm_states,
)
from .trials import run_trials, summarize_trials

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - plain Python fallback
    def display(obj):
        print(obj)


def run_single_P_workflow(
    P: np.ndarray,
    r: int,
    N: int = 1000,
    trials: int = 200,
    noise_model: str = "multinomial",
    seed: int = 12345,
) -> Tuple[PBlocks, pd.DataFrame, pd.DataFrame]:
    """
    Full deterministic + stochastic diagnostics for one fixed P.
    """
    blocks = deterministic_blocks_from_P(P, r=r)

    print("Deterministic block report:")
    display(block_report(blocks))

    print("Running shot-noise trials...")
    trial_df = run_trials(
        blocks=blocks,
        trials=trials,
        N=N,
        noise_model=noise_model,
        seed=seed,
        progress=True,
    )

    summary = summarize_trials(trial_df)

    print("Trial summary:")
    display(summary)

    plot_metric_vs_predictors(summary, quantile="p90")

    return blocks, trial_df, summary

def make_toy_low_rank_probability_matrix(
    nout: int,
    ntr: int,
    r: int,
    rng: np.random.Generator,
    basis_concentration: float = 1.0,
    coeff_concentration: float = 1.0,
) -> np.ndarray:
    """
    Generate a rank <= r column-stochastic probability matrix P = B W.

    B has r probability columns in R^nout.
    W has ntr probability columns in R^r.
    Then each P[:, i] is a convex combination of the columns of B.
    """
    if r >= min(nout, ntr):
        raise ValueError("Need r < min(nout,ntr).")

    B = rng.dirichlet(
        alpha=basis_concentration * np.ones(nout),
        size=r,
    ).T  # shape (nout, r)

    W = rng.dirichlet(
        alpha=coeff_concentration * np.ones(r),
        size=ntr,
    ).T  # shape (r, ntr)

    P = B @ W
    P = np.maximum(P, 0.0)
    P = P / P.sum(axis=0, keepdims=True)
    return P

def run_toy_low_rank_sweep(
    q_values: Tuple[int, ...] = (4,),
    p_values: Tuple[int, ...] = (500, 1000, 2500, 5000),
    d: int = 2,
    N: int = 1000,
    trials: int = 30,
    noise_model: str = "gaussian",
    seed: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run enough toy configurations to make the summary table and log-log plots meaningful.
    """
    r = d**2
    trial_parts = []
    summary_parts = []

    for q in q_values:
        for p_dim in p_values:
            nout = r + q
            ntr = r + p_dim
            pair_seed = seed + 1000 * q + p_dim
            rng = np.random.default_rng(pair_seed)
            P_toy = make_toy_low_rank_probability_matrix(
                nout=nout,
                ntr=ntr,
                r=r,
                rng=rng,
            )
            blocks = deterministic_blocks_from_P(P_toy, r=r)

            print(f"q={q:>2}, p={p_dim:>5}: running {trials} trials")
            trial_df = run_trials(
                blocks=blocks,
                trials=trials,
                N=N,
                noise_model=noise_model,
                seed=pair_seed + 17,
                progress=False,
            )
            summary = summarize_trials(trial_df)

            trial_parts.append(trial_df)
            summary_parts.append(summary)

    trial_df = pd.concat(trial_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    return trial_df, summary.sort_values(["q", "p"]).reset_index(drop=True)

def run_dimension_sweep(
    make_P_fn: Callable[[int, int, int, np.random.Generator], np.ndarray],
    d_values: List[int],
    q_values: List[int],
    kappa_values: List[float],
    scaling: str = "q3",
    trials: int = 50,
    N: int = 1000,
    noise_model: str = "gaussian",
    seed: int = 12345,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sweep dimensions.

    scaling:
        "q3": set p = ceil(kappa * r * q^3)
              relevant when c_p ~ 1/q and lambda_min(C22) ~ 1/q.

        "q5": set p = ceil(kappa * r * q^5)
              conservative worst-case when c_p = O(1).

        "custom": kappa_values are interpreted directly as p values.
    """
    rng = np.random.default_rng(seed)

    all_trials = []
    all_summaries = []
    all_det = []

    for d in d_values:
        r = d**2

        for q in q_values:
            nout = r + q

            for kappa in kappa_values:
                if scaling == "q3":
                    p_dim = int(np.ceil(kappa * r * q**3))
                elif scaling == "q5":
                    p_dim = int(np.ceil(kappa * r * q**5))
                elif scaling == "custom":
                    p_dim = int(np.ceil(kappa))
                else:
                    raise ValueError("scaling must be 'q3', 'q5', or 'custom'.")

                ntr = r + p_dim

                if progress:
                    print(
                        f"d={d}, r={r}, q={q}, p={p_dim}, "
                        f"nout={nout}, ntr={ntr}, scaling={scaling}, kappa={kappa}"
                    )

                P = make_P_fn(d, nout, ntr, rng)
                blocks = deterministic_blocks_from_P(P, r=r)

                det = block_report(blocks)
                det["d"] = d
                det["kappa_input"] = kappa
                det["scaling"] = scaling
                all_det.append(det)

                trial_df = run_trials(
                    blocks=blocks,
                    trials=trials,
                    N=N,
                    noise_model=noise_model,
                    seed=int(rng.integers(0, 2**32 - 1)),
                    progress=False,
                )

                trial_df["d"] = d
                trial_df["kappa_input"] = kappa
                trial_df["scaling"] = scaling
                all_trials.append(trial_df)

                summary = summarize_trials(trial_df)
                summary["d"] = d
                summary["kappa_input"] = kappa
                summary["scaling"] = scaling
                all_summaries.append(summary)

    trials_df = pd.concat(all_trials, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    det_df = pd.concat(all_det, ignore_index=True)

    return trials_df, summary_df, det_df

def make_toy_P_for_sweep(
    d: int,
    nout: int,
    ntr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r = d**2
    return make_toy_low_rank_probability_matrix(
        nout=nout,
        ntr=ntr,
        r=r,
        rng=rng,
        basis_concentration=1.0,
        coeff_concentration=1.0,
    )


def make_random_quantum_probability_matrix(
    nout: int,
    ntr: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build P from a random rank-1 POVM and Haar random pure states.

    The resulting matrix has shape (nout, ntr) and entries
        P[a, i] = Tr(mu_a rho_i).
    """
    effects = generate_random_rank1_povm(nout=nout, dim=dim, rng=rng)
    states = generate_haar_random_pure_states(num_states=ntr, dim=dim, rng=rng)
    return probability_matrix_from_povm_states(effects, states)


def run_random_quantum_scaling_sweep(
    d_values: Sequence[int],
    nout_values: Sequence[int],
    ntr_values: Sequence[int],
    *,
    repetitions: int = 1,
    seed: int = 12345,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Sweep deterministic block quantities for random quantum P matrices.

    For each d, nout, ntr, and repetition, this generates a random rank-1 POVM
    with nout effects, generates ntr Haar random pure states in C^d, builds P,
    and records lambda_min_C22, lambda_max_C22, delta_shape, c_p, and related
    deterministic diagnostics.

    The block rank is r=d^2, so this routine requires nout>d^2 and ntr>d^2.
    The POVM isometry construction also requires nout>=d.
    """
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    rng = np.random.default_rng(seed)
    rows = []

    for d in d_values:
        if d <= 0:
            raise ValueError("All d values must be positive.")

        r = d**2

        for nout in nout_values:
            if nout <= r:
                raise ValueError(f"Need nout > d^2 for deterministic blocks; got d={d}, nout={nout}.")

            for ntr in ntr_values:
                if ntr <= r:
                    raise ValueError(f"Need ntr > d^2 for deterministic blocks; got d={d}, ntr={ntr}.")

                for repetition in range(repetitions):
                    if progress:
                        print(
                            f"d={d}, nout={nout}, ntr={ntr}, "
                            f"rep={repetition + 1}/{repetitions}"
                        )

                    P = make_random_quantum_probability_matrix(
                        nout=nout,
                        ntr=ntr,
                        dim=d,
                        rng=rng,
                    )
                    blocks = deterministic_blocks_from_P(P, r=r)

                    row = block_report(blocks).iloc[0].to_dict()
                    row.update(
                        {
                            "d": d,
                            "d2": r,
                            "repetition": repetition,
                            "nout_over_d2": nout / r,
                            "ntr_over_d2": ntr / r,
                            "ntr_over_nout": ntr / nout,
                            "c_p": blocks.c_p,
                            "rank_estimate": int(np.sum(blocks.singular_values > 1e-10)),
                            "sigma_r": float(blocks.singular_values[r - 1]),
                        }
                    )
                    rows.append(row)

    return pd.DataFrame(rows)


def fit_random_quantum_scaling_laws(
    df: pd.DataFrame,
    quantities: Sequence[str] = (
        "lambda_min_C22",
        "lambda_max_C22",
        "delta_shape",
        "c_p",
    ),
    group_cols: Sequence[str] = ("d",),
) -> pd.DataFrame:
    """
    Fit two-variable power laws y ~= const * nout^alpha * ntr^beta.

    Fits are performed in log space within each group in group_cols.
    """
    required = {"nout", "ntr", *group_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    grouped = df.groupby(list(group_cols), dropna=False) if group_cols else [((), df)]

    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_tuple))

        for quantity in quantities:
            if quantity not in group.columns:
                continue

            temp = group[["nout", "ntr", quantity]].dropna()
            temp = temp[(temp["nout"] > 0) & (temp["ntr"] > 0) & (temp[quantity] > 0)]

            if len(temp) < 3:
                rows.append(
                    {
                        **base,
                        "quantity": quantity,
                        "nout_power": np.nan,
                        "ntr_power": np.nan,
                        "log_intercept": np.nan,
                        "num_rows": len(temp),
                    }
                )
                continue

            X = np.column_stack(
                [
                    np.ones(len(temp)),
                    np.log(np.asarray(temp["nout"], dtype=float)),
                    np.log(np.asarray(temp["ntr"], dtype=float)),
                ]
            )
            y = np.log(np.asarray(temp[quantity], dtype=float))
            intercept, nout_power, ntr_power = np.linalg.lstsq(X, y, rcond=None)[0]

            rows.append(
                {
                    **base,
                    "quantity": quantity,
                    "nout_power": float(nout_power),
                    "ntr_power": float(ntr_power),
                    "log_intercept": float(intercept),
                    "num_rows": len(temp),
                }
            )

    return pd.DataFrame(rows)
