"""Notebook-oriented workflows that compose the lower-level QELM modules.

This module is intentionally orchestration-heavy.  It builds parameter grids,
constructs toy or quantum probability matrices, calls ``qelm_rank.trials`` for
fixed-``P`` shot-noise diagnostics, and calls ``qelm_rank.training`` for actual
QELM training and tilde-U approximation studies.  The numerical primitives live
in ``blocks.py``, ``noise.py``, ``quantum.py``, and ``training.py``; functions
here mainly package those pieces into repeatable experiments and summary
tables for notebooks.
"""

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, List, Sequence, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .blocks import (
    PBlocks,
    block_report,
    deterministic_blocks_from_P,
    schur_covariance_blocks,
    svd_probability_blocks,
)
from .linalg import (
    distribution_summary,
    frobnorm,
    loglog_fit,
    opnorm,
    psd_solve,
    quantile_suffix,
    symmetrize,
)
from .noise import shot_noise_matrix
from .plotting import (
    plot_mean_median_quantile_summary,
    plot_grouped_mean_median_quantile_summary,
    plot_metric_vs_predictors,
    plot_summary_series,
)
from .quantum import (
    POVMEffects,
    QuantumStateBatch,
    get_rng,
)
from .training import (
    QELMRun,
    QELMTrainingSpec,
    ResolvedTarget,
    ResolvedTest,
    TildeUDiagnostics,
    TildeUTrainingApproxStudySpec,
    _povm_kind_from_spec,
    _required_noise_N,
    _resolve_test_state_request,
    _training_state_count_from_spec,
    estimate_actual_training_mse,
    estimate_actual_training_mse_target_average,
    leading_training_bias_variance_terms,
    leading_training_bias_variance_terms_target_average,
    tilde_u_correction_operator_diagnostics,
    with_training_sweep_value,
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
    seed: int | None = None,
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
    rng: np.random.Generator | int | None = None,
    basis_concentration: float = 1.0,
    coeff_concentration: float = 1.0,
) -> np.ndarray:
    """
    Generate a rank <= r column-stochastic probability matrix P = B W.

    B has r probability columns in R^nout.
    W has ntr probability columns in R^r.
    Then each P[:, i] is a convex combination of the columns of B.
    """
    rng = get_rng(rng)
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
    seed: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run enough toy configurations to make the summary table and log-log plots meaningful.
    """
    r = d**2
    trial_parts = []
    summary_parts = []

    rng = get_rng(seed)

    for q in q_values:
        for p_dim in p_values:
            nout = r + q
            ntr = r + p_dim
            pair_seed = int(rng.integers(0, 2**32 - 1))
            pair_rng = np.random.default_rng(pair_seed)
            P_toy = make_toy_low_rank_probability_matrix(
                nout=nout,
                ntr=ntr,
                r=r,
                rng=pair_rng,
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
    seed: int | None = None,
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
    rng = get_rng(seed)

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
    rng: np.random.Generator | int | None = None,
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


def run_random_quantum_scaling_sweep(
    d_values: Sequence[int],
    nout_values: Sequence[int],
    ntr_values: Sequence[int],
    *,
    repetitions: int = 1,
    seed: int | None = None,
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

    rng = get_rng(seed)
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

                    povm = POVMEffects.random_rank1(nout=nout, dim=d, rng=rng)
                    states = QuantumStateBatch.haar_pure(num_states=ntr, dim=d, rng=rng)
                    P = povm.probability_matrix(states)
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


SCHUR_COMPLEMENT_METRIC_COLS = [
    "empirical_schur_op",
    "limit_approx_op",
    "xi11_op",
    "limit_approx_error_op",
    "xi11_approx_error_op",
    "limit_relative_error",
    "xi11_relative_error",
    "empirical_correction_op",
    "limit_correction_op",
    "empirical_correction_over_xi11",
    "limit_correction_over_xi11",
    "sample_vs_limit_correction_relative_error",
    "C22_lambda_min",
    "C22_cond",
    "S22_lambda_min",
    "S22_cond",
    "C12_op",
]

SCHUR_COMPLEMENT_SUMMARY_COLS = [
    "ntr",
    "p_kernel",
    "q",
    "empirical_schur_op_median",
    "limit_approx_error_op_median",
    "xi11_approx_error_op_median",
    "limit_relative_error_median",
    "xi11_relative_error_median",
    "empirical_correction_over_xi11_median",
    "limit_approx_ok_median",
    "xi11_approx_ok_median",
    "C22_lambda_min_median",
    "S22_lambda_min_median",
]


def run_repeated_trials(
    *,
    trial_fn,
    configs: Sequence[dict],
    trials: int,
    seed: int | None,
    shared_kwargs: dict | None = None,
    rng_arg: str | None = None,
    seed_arg: str | None = "seed",
    verbose: bool = True,
    progress_fields: Sequence[str] = ("nout", "ntr"),
    fail_soft: bool = False,
) -> pd.DataFrame:
    """
    Run a trial function repeatedly over a list of parameter configurations.

    The trial function can receive either a generated integer seed or a fresh
    NumPy RNG per trial. This keeps sweep mechanics separate from the quantity
    being measured.
    """
    master_rng = get_rng(seed)
    shared_kwargs = shared_kwargs or {}
    rows = []

    for config in configs:
        if verbose:
            parts = []
            for field in progress_fields:
                if field in config:
                    value = config[field]
                elif field in shared_kwargs:
                    value = shared_kwargs[field]
                else:
                    continue
                parts.append(f"{field}={value}")
            print(", ".join([*parts, f"trials={trials}"]))

        for trial in range(trials):
            trial_seed = int(master_rng.integers(0, 2**32 - 1))
            kwargs = {**shared_kwargs, **config}

            if rng_arg is not None:
                kwargs[rng_arg] = np.random.default_rng(trial_seed)
            elif seed_arg is not None:
                kwargs[seed_arg] = trial_seed

            try:
                row = trial_fn(**kwargs)
                row["failed"] = False
                row["error"] = ""
            except Exception as exc:
                if not fail_soft:
                    raise
                row = {
                    **shared_kwargs,
                    **config,
                    "failed": True,
                    "error": repr(exc),
                }

            row["trial"] = trial
            row["trial_seed"] = trial_seed
            rows.append(row)

    return pd.DataFrame(rows)


def one_schur_complement_approx_trial(
    *,
    d,
    nout,
    ntr,
    rng=None,
    noise,
    Nshots,
    rcond,
    ridge,
    rank,
) -> dict:
    """
    Run one Schur-complement approximation trial.

    Compares the empirical Schur term Xi11 - S12 S22^+ Xi21 with the
    limit-covariance approximation Xi11 - C12 C22^+ Xi21 and Xi11 alone.
    """
    rng = get_rng(rng)
    r = d * d if rank is None else int(rank)
    povm = POVMEffects.random_isometry(nout=nout, dim=d, rng=rng)
    states = QuantumStateBatch.haar_pure_from_columns(num_states=ntr, dim=d, rng=rng)
    P = povm.probability_matrix(states)

    blocks = svd_probability_blocks(P, rank=r)
    U1 = blocks["U1"]
    U2 = blocks["U2"]
    V1 = blocks["V1"]
    V2 = blocks["V2"]
    pi2_diag = blocks["Pi2_diag"]

    C12, C22 = schur_covariance_blocks(P, U1, U2, pi2_diag)

    Xi = shot_noise_matrix(P, rng, Nshots=Nshots, noise=noise)
    Xi11 = U1.T @ Xi @ V1
    Xi12 = U1.T @ Xi @ V2
    Xi21 = U2.T @ Xi @ V1
    Xi22 = U2.T @ Xi @ V2

    p_kernel = blocks["p_kernel"]
    S12 = Xi12 @ Xi22.T / p_kernel
    S22 = Xi22 @ Xi22.T / p_kernel

    S22_inv_Xi21, eigvals_S22, kept_S22 = psd_solve(
        S22,
        Xi21,
        rcond=rcond,
        ridge=ridge,
    )
    C22_inv_Xi21, eigvals_C22, kept_C22 = psd_solve(
        C22,
        Xi21,
        rcond=rcond,
        ridge=ridge,
    )

    empirical_correction = S12 @ S22_inv_Xi21
    limit_correction = C12 @ C22_inv_Xi21

    empirical_schur = Xi11 - empirical_correction
    limit_approx = Xi11 - limit_correction
    xi11_only = Xi11

    eps = 1e-15
    empirical_schur_op = opnorm(empirical_schur)
    xi11_op = opnorm(xi11_only)
    empirical_correction_op = opnorm(empirical_correction)
    limit_correction_op = opnorm(limit_correction)

    limit_error_op = opnorm(empirical_schur - limit_approx)
    xi11_error_op = opnorm(empirical_schur - xi11_only)

    return {
        "d": d,
        "r": blocks["r"],
        "nout": nout,
        "ntr": ntr,
        "q": blocks["q"],
        "p_kernel": p_kernel,
        "noise": noise,
        "empirical_schur_op": empirical_schur_op,
        "limit_approx_op": opnorm(limit_approx),
        "xi11_op": xi11_op,
        "limit_approx_error_op": limit_error_op,
        "xi11_approx_error_op": xi11_error_op,
        "limit_relative_error": limit_error_op / max(empirical_schur_op, eps),
        "xi11_relative_error": xi11_error_op / max(empirical_schur_op, eps),
        "empirical_correction_op": empirical_correction_op,
        "limit_correction_op": limit_correction_op,
        "empirical_correction_over_xi11": empirical_correction_op / max(xi11_op, eps),
        "limit_correction_over_xi11": limit_correction_op / max(xi11_op, eps),
        "sample_vs_limit_correction_relative_error": (
            opnorm(empirical_correction - limit_correction)
            / max(empirical_correction_op, eps)
        ),
        "C22_lambda_min": float(np.min(eigvals_C22)),
        "C22_cond": (
            float(np.max(eigvals_C22) / np.min(eigvals_C22))
            if np.min(eigvals_C22) > 0
            else np.inf
        ),
        "C22_kept_rank": int(np.sum(kept_C22)),
        "S22_lambda_min": float(np.min(eigvals_S22)),
        "S22_cond": (
            float(np.max(eigvals_S22) / np.min(eigvals_S22))
            if np.min(eigvals_S22) > 0
            else np.inf
        ),
        "S22_kept_rank": int(np.sum(kept_S22)),
        "C12_op": opnorm(C12),
    }


def summarize_schur_complement_approx(
    raw_df: pd.DataFrame,
    relative_error_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Summarize Schur-complement approximation trials by parameter setting.
    """
    group_cols = ["d", "r", "nout", "ntr", "q", "p_kernel", "noise"]
    rows = []

    for key, group in raw_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["trials"] = len(group)
        row["C22_kept_rank_min"] = int(group["C22_kept_rank"].min())
        row["S22_kept_rank_min"] = int(group["S22_kept_rank"].min())

        for col in SCHUR_COMPLEMENT_METRIC_COLS:
            values = group[col].to_numpy(dtype=float)
            row[f"{col}_q25"] = float(np.quantile(values, 0.25))
            row[f"{col}_median"] = float(np.median(values))
            row[f"{col}_q75"] = float(np.quantile(values, 0.75))
            row[f"{col}_p90"] = float(np.quantile(values, 0.90))

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("p_kernel").reset_index(drop=True)
    summary["limit_approx_ok_median"] = (
        summary["limit_relative_error_median"] <= relative_error_threshold
    )
    summary["xi11_approx_ok_median"] = (
        summary["xi11_relative_error_median"] <= relative_error_threshold
    )
    return summary


def run_schur_complement_approx_experiment(
    *,
    d=2,
    nout=32,
    ntr_values=(128, 256, 512, 1024, 2048, 4096),
    trials=20,
    noise="gaussian",
    Nshots=10_000,
    seed=None,
    rcond=1e-12,
    ridge=0.0,
    rank=None,
    relative_error_threshold=0.10,
    verbose=True,
    show_summary=True,
    make_plots=True,
    plot_specs=(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run, summarize, display, and plot the Schur-complement approximation check.
    """
    raw_df = run_repeated_trials(
        trial_fn=one_schur_complement_approx_trial,
        configs=[{"nout": nout, "ntr": ntr} for ntr in ntr_values],
        trials=trials,
        seed=seed,
        shared_kwargs={
            "d": d,
            "noise": noise,
            "Nshots": Nshots,
            "rcond": rcond,
            "ridge": ridge,
            "rank": rank,
        },
        rng_arg="rng",
        verbose=verbose,
        progress_fields=("d", "nout", "ntr"),
    )
    summary_df = summarize_schur_complement_approx(
        raw_df,
        relative_error_threshold=relative_error_threshold,
    )

    if show_summary:
        display(summary_df[SCHUR_COMPLEMENT_SUMMARY_COLS])

    if make_plots and plot_specs:
        plot_summary_series(
            summary_df,
            x_col="p_kernel",
            plots=plot_specs,
            thresholds={"relative_error": relative_error_threshold},
        )

    return raw_df, summary_df


def _tilde_u_trial_row(
    *,
    d: int,
    ntr: int,
    N: int,
    noise: str,
    blocks: dict,
    diag: TildeUDiagnostics,
    test: ResolvedTest,
    target: ResolvedTarget,
    exact: dict,
    identity: dict,
    actual: dict,
) -> dict:
    eps = 1e-15
    row = {
        "d": d,
        "r": blocks["r"],
        "nout": blocks["U1"].shape[0],
        "ntr": ntr,
        "q": blocks["q"],
        "p_kernel": blocks["p_kernel"],
        "N": N,
        "noise": noise,
        "test_state": test.mode,
        "test_average": test.average,
        "num_test_points": test.num_points,
        "target_kind": target.kind,
        "target_average": target.average,
        "target_normalization": target.normalization,
        "target_scale": target.scale,
        "C22_inv_C21_op": diag.C22_inv_C21_op,
        "correction_op": diag.correction_op,
        "correction_op_relative_difference": diag.correction_op_relative_difference,
        "C22_lambda_min": diag.C22_lambda_min,
        "C22_lambda_max": diag.C22_lambda_max,
        "C22_cond": diag.C22_cond,
        "C22_kept_rank": diag.C22_kept_rank,
        "leading_bias_sq_exact": exact["bias_sq"],
        "leading_var_exact": exact["variance"],
        "leading_mse_exact": exact["mse"],
        "leading_bias_sq_identity": identity["bias_sq"],
        "leading_var_identity": identity["variance"],
        "leading_mse_identity": identity["mse"],
        "bias_sq_identity_over_exact": identity["bias_sq"] / max(exact["bias_sq"], eps),
        "var_identity_over_exact": identity["variance"] / max(exact["variance"], eps),
        "mse_identity_over_exact": identity["mse"] / max(exact["mse"], eps),
        "actual_over_leading_exact": actual["actual_mse"] / max(exact["mse"], eps),
        "actual_over_leading_identity": actual["actual_mse"] / max(identity["mse"], eps),
        "leading_exact_relative_error": abs(exact["mse"] - actual["actual_mse"]) / max(actual["actual_mse"], eps),
        "leading_identity_relative_error": abs(identity["mse"] - actual["actual_mse"]) / max(actual["actual_mse"], eps),
    }
    row.update(actual)
    return row


def one_tilde_u_training_approx_trial_from_spec(
    *,
    spec: QELMTrainingSpec,
    rng: np.random.Generator | int | None = None,
    **_,
) -> dict:
    """
    One tilde-U approximation trial from a structured QELMTrainingSpec.
    """
    rng = get_rng(rng)
    r = spec.data.d * spec.data.d if spec.numerics.rank is None else int(spec.numerics.rank)
    if spec.data.nout <= r:
        raise ValueError(f"Need nout > r. Got nout={spec.data.nout}, r={r}.")

    run = QELMRun(spec, rng=rng)
    actual = run.train_model()
    leading_corrected = run.leading_error(corrected=True)
    leading_identity = run.leading_error(corrected=False)

    ntr = int(run.context.P_train.shape[1])
    if ntr < spec.data.nout:
        raise ValueError(f"Need ntr >= nout. Got ntr={ntr}, nout={spec.data.nout}.")

    return _tilde_u_trial_row(
        d=spec.data.d,
        ntr=ntr,
        N=_required_noise_N(spec.noise),
        noise=spec.noise.noise,
        blocks=run.diagnostics.blocks,
        diag=run.diagnostics,
        test=run.test,
        target=run.target,
        exact=leading_corrected.to_metrics_dict(),
        identity=leading_identity.to_metrics_dict(),
        actual=actual.to_metrics_dict(),
    )


TILDE_U_TRAINING_APPROX_METRIC_COLS = [
    "C22_inv_C21_op",
    "correction_op",
    "C22_lambda_min",
    "C22_cond",
    "leading_bias_sq_exact",
    "leading_bias_sq_identity",
    "leading_var_exact",
    "leading_var_identity",
    "leading_mse_exact",
    "leading_mse_identity",
    "actual_mse",
    "actual_bias_sq",
    "actual_variance",
    "bias_sq_identity_over_exact",
    "var_identity_over_exact",
    "mse_identity_over_exact",
    "actual_over_leading_exact",
    "actual_over_leading_identity",
    "leading_exact_relative_error",
    "leading_identity_relative_error",
]

TILDE_U_TRAINING_APPROX_PLOT_SPECS = {
    "correction": (
        [
            ("C22_inv_C21_op", r"$C_{22}^{-1}C_{21}$"),
            ("correction_op", r"$U_2 C_{22}^{-1} C_{21} U_1^T$"),
        ],
        "Size of the C22 correction",
        "operator norm",
    ),
    "bias": (
        [
            ("leading_bias_sq_exact", "with correction"),
            ("leading_bias_sq_identity", r"$\tilde U U_1^T = I$"),
        ],
        "Leading training squared bias",
        "mean test squared bias, leading formula",
    ),
    "variance": (
        [
            ("leading_var_exact", "with correction"),
            ("leading_var_identity", r"$\tilde U U_1^T = I$"),
        ],
        "Leading training variance",
        "mean test variance, leading formula",
    ),
    "leading_mse": (
        [
            ("leading_mse_exact", "with correction"),
            ("leading_mse_identity", r"$\tilde U U_1^T = I$"),
        ],
        "Leading bias-plus-variance prediction",
        "leading squared bias + leading variance",
    ),
    "mse": (
        [
            ("leading_mse_exact", r"leading, full $\tilde U U_1^T$"),
            ("leading_mse_identity", r"leading, $\tilde U U_1^T = I$"),
            ("actual_mse", "true MSE"),
        ],
        "MSE: true vs leading approximation",
        "MSE",
    ),
    "mse_ratio": (
        [("mse_identity_over_exact", r"identity / corrected")],
        "Effect of dropping the C22 correction in the leading MSE",
        "identity leading MSE / corrected leading MSE",
    ),
    "actual_ratio": (
        [
            ("actual_over_leading_exact", "actual / corrected leading"),
            ("actual_over_leading_identity", "actual / identity leading"),
        ],
        "Actual MSE divided by leading prediction",
        "actual MSE / leading MSE",
    ),
    "relative_error": (
        [
            ("leading_exact_relative_error", "corrected leading"),
            ("leading_identity_relative_error", "identity leading"),
        ],
        "Relative error of leading prediction",
        "|leading MSE - numerical MSE| / numerical MSE",
    ),
}
TILDE_U_TRAINING_APPROX_PLOTS = list(TILDE_U_TRAINING_APPROX_PLOT_SPECS.values())


def _tilde_u_training_approx_plots_from_keys(
    plots: Sequence[str] | str | None,
) -> list[tuple]:
    """
    Resolve short plot keys from TildeUTrainingApproxStudySpec into plot specs.
    """
    if plots is None:
        return []
    if isinstance(plots, str):
        keys = (plots,)
    else:
        keys = tuple(plots)

    if not keys or "all" in keys:
        return TILDE_U_TRAINING_APPROX_PLOTS

    unknown = tuple(key for key in keys if key not in TILDE_U_TRAINING_APPROX_PLOT_SPECS)
    if unknown:
        available = ", ".join(("all", *TILDE_U_TRAINING_APPROX_PLOT_SPECS))
        raise ValueError(f"Unknown tilde-U plot key(s): {unknown}. Available keys: {available}.")

    return [TILDE_U_TRAINING_APPROX_PLOT_SPECS[key] for key in keys]


def summarize_tilde_u_training_approx(
    raw_df: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.25, 0.75),
    group_cols: Sequence[str] = (
        "d",
        "r",
        "nout",
        "ntr",
        "q",
        "p_kernel",
        "N",
        "noise",
        "test_state",
        "target_kind",
        "target_average",
        "target_normalization",
    ),
) -> pd.DataFrame:
    """
    Summarize tilde-U approximation trials with configurable quantiles.
    """
    rows = []
    ok = raw_df[~raw_df.get("failed", False).astype(bool)].copy() if "failed" in raw_df else raw_df
    active_group_cols = tuple(col for col in group_cols if col in ok.columns)

    for key, group in ok.groupby(list(active_group_cols), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(active_group_cols, key))
        row["trials"] = len(group)
        if "actual_noise_trials" in group.columns:
            row["actual_noise_trials"] = int(group["actual_noise_trials"].iloc[0])
        if "num_test_points" in group.columns:
            row["num_test_points"] = int(group["num_test_points"].iloc[0])
        if "target_scale" in group.columns:
            row["target_scale_mean"] = float(group["target_scale"].mean())
        if "C22_kept_rank" in group.columns:
            row["C22_kept_rank_min"] = int(group["C22_kept_rank"].min())

        for col in TILDE_U_TRAINING_APPROX_METRIC_COLS:
            if col not in group.columns:
                continue
            stats = distribution_summary(group[col].to_numpy(dtype=float), quantiles=quantiles)
            for name, value in stats.items():
                row[f"{col}_{name}"] = value

        rows.append(row)

    return pd.DataFrame(rows).sort_values(list(active_group_cols)).reset_index(drop=True)


def fit_tilde_u_training_approx_slopes(
    summary_df: pd.DataFrame,
    *,
    x_col: str = "p_kernel",
    ycols: Sequence[str] = (
        "C22_inv_C21_op_median",
        "correction_op_median",
        "leading_mse_exact_median",
        "leading_mse_identity_median",
        "actual_mse_median",
    ),
    group_cols: Sequence[str] = ("d", "nout", "N", "noise"),
) -> pd.DataFrame:
    """
    Fit log-log slopes for selected summarized tilde-U approximation quantities.
    """
    rows = []
    grouped = summary_df.groupby(list(group_cols), dropna=False) if group_cols else [((), summary_df)]

    for key, group in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_tuple))

        for ycol in ycols:
            if ycol not in group.columns:
                continue
            slope, intercept = loglog_fit(group[x_col].to_numpy(dtype=float), group[ycol].to_numpy(dtype=float))
            rows.append(
                {
                    **base,
                    "x": x_col,
                    "y": ycol,
                    "slope": slope,
                    "intercept": intercept,
                    "law": f"{ycol} ~ {x_col}^{slope:.3f}" if np.isfinite(slope) else "insufficient data",
                    "num_points": int(np.isfinite(group[ycol]).sum()),
                }
            )

    return pd.DataFrame(rows)


def _tilde_u_slope_group_cols(summary_df: pd.DataFrame, x_col: str) -> tuple[str, ...]:
    excluded = {x_col}
    if x_col in {"q", "nout"}:
        excluded.update({"q", "nout"})
    if x_col in {"p_kernel", "ntr"}:
        excluded.update({"p_kernel", "ntr"})

    candidates = (
        "d",
        "nout",
        "N",
        "noise",
        "test_state",
        "target_kind",
        "target_average",
        "target_normalization",
    )
    return tuple(
        col for col in candidates
        if col in summary_df.columns and col not in excluded
    )


def _tilde_u_study_x_col(study: TildeUTrainingApproxStudySpec) -> str:
    if study.x_col is not None:
        return study.x_col
    elif study.sweep_col == "nout":
        return "q"
    elif study.sweep_col == "N":
        return "N"
    elif study.sweep_col == "ntr":
        return "ntr"
    else:
        raise ValueError("sweep_col must be 'ntr', 'nout', or 'N'.")


def _tilde_u_study_sweep_values(study: TildeUTrainingApproxStudySpec) -> tuple[int, ...]:
    if study.sweep_values is not None:
        return tuple(int(value) for value in study.sweep_values)
    if study.sweep_col == "ntr":
        count = _training_state_count_from_spec(study.base.data.train_states)
        if count is None:
            raise ValueError("ntr sweep_values are required when train_states has no fixed count.")
        return (count,)
    if study.sweep_col == "nout":
        return (int(study.base.data.nout),)
    if study.sweep_col == "N":
        return (_required_noise_N(study.base.noise),)
    raise ValueError("sweep_col must be 'ntr', 'nout', or 'N'.")


def _tilde_u_configs_from_study(study: TildeUTrainingApproxStudySpec) -> list[dict]:
    configs = []
    for value in _tilde_u_study_sweep_values(study):
        spec = with_training_sweep_value(study.base, study.sweep_col, value)
        ntr = _training_state_count_from_spec(spec.data.train_states)
        N = _required_noise_N(spec.noise)
        configs.append(
            {
                "spec": spec,
                "d": spec.data.d,
                "nout": spec.data.nout,
                "N": N,
                **({} if ntr is None else {"ntr": ntr}),
            }
        )
    return configs

def run_tilde_u_training_approx_experiment(
    study: TildeUTrainingApproxStudySpec,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run raw and summarized trials for a tilde-U approximation study.

    This calls one_tilde_u_training_approx_trial_from_spec repeatedly,
    passing it a QELMTrainingSpec with the sweep value set for each trial.
    We use _tilde_u_configs_from_study to convert the TildeUTrainingApproxStudySpec
    into a list of trial configurations and in particular the QELMTrainingSpec for each trial.
    """
    if study.repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    raw_df = run_repeated_trials(
        trial_fn=one_tilde_u_training_approx_trial_from_spec,
        configs=_tilde_u_configs_from_study(study),
        trials=study.repetitions,
        seed=study.seed,
        rng_arg="rng",
        verbose=study.verbose,
        progress_fields=("d", "nout", "ntr", "N"),
        fail_soft=study.fail_soft,
    )
    summary_df = summarize_tilde_u_training_approx(
        raw_df,
        quantiles=study.quantiles,
    )
    return raw_df, summary_df


def _save_tilde_u_training_approx_report_data(
    output_file: str | Path,
    *,
    raw_df: pd.DataFrame,
    metadata: dict,
    overwrite: bool = False,
) -> Path:
    path = _tilde_u_report_output_path(output_file)
    if not overwrite:
        path = _noncolliding_output_path(path)
    return _save_tilde_u_training_approx_report_zip(
        path,
        raw_df=raw_df,
        metadata=metadata,
    )


def _tilde_u_report_output_path(output_file: str | Path) -> Path:
    path = Path(output_file).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".zip")
    if path.suffix.lower() != ".zip":
        raise ValueError("tilde-U report output_file must end in .zip.")
    return path


def _noncolliding_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output path for {path}.")


def _portable_dataframe_bytes(df: pd.DataFrame) -> bytes:
    portable = df.copy()
    for col in portable.columns:
        if isinstance(portable[col].dtype, pd.StringDtype):
            portable[col] = portable[col].astype(object)
    buffer = BytesIO()
    portable.to_parquet(buffer, index=False, compression="zstd")
    return buffer.getvalue()


def _read_portable_dataframe_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(BytesIO(data))


def _array_payload(value) -> dict:
    array = np.asarray(value)
    payload = {
        "kind": "ndarray",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }
    if np.iscomplexobj(array):
        payload["real"] = np.real(array).tolist()
        payload["imag"] = np.imag(array).tolist()
    else:
        payload["data"] = array.tolist()
    return payload


def _selector_descriptor(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return _array_payload(value)
    if isinstance(value, dict):
        return {str(key): _selector_descriptor(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_selector_descriptor(item) for item in value]
    return repr(value)


def _povm_descriptor(povm) -> dict:
    kind = _povm_kind_from_spec(povm)
    if kind == "random_rank1":
        descriptor = {"kind": "random_rank1"}
        if isinstance(povm, dict):
            for key in ("nout", "dim"):
                if key in povm:
                    descriptor[key] = int(povm[key])
        return descriptor

    if hasattr(povm, "effects"):
        effects = np.asarray(povm.effects)
        return {
            "kind": "explicit",
            "effects": _array_payload(effects),
            **({"label": povm.label} if getattr(povm, "label", None) is not None else {}),
        }

    effects = povm.get("effects") if isinstance(povm, dict) else povm
    return {"kind": "explicit", "effects": _array_payload(effects)}


def _state_batch_descriptor(batch: QuantumStateBatch) -> dict:
    descriptor = {
        "kind": "explicit",
        "states": _array_payload(batch.states),
    }
    if getattr(batch, "label", None) is not None:
        descriptor["label"] = batch.label
    return descriptor


def _training_states_descriptor(train_states, *, dim: int) -> dict:
    if isinstance(train_states, str):
        if train_states.lower() == "haar_pure":
            return {"kind": "haar_pure"}
        return {"kind": train_states}
    if isinstance(train_states, QuantumStateBatch):
        return _state_batch_descriptor(train_states)
    if isinstance(train_states, dict):
        if "kind" in train_states:
            kind = str(train_states["kind"]).lower()
        elif "vectors" in train_states:
            kind = "state_vectors"
        else:
            kind = "states"
        if kind == "haar_pure":
            descriptor = {"kind": "haar_pure"}
            if "num_states" in train_states:
                descriptor["num_states"] = int(train_states["num_states"])
            return descriptor
        if kind == "state_vectors":
            batch = QuantumStateBatch.from_state_vectors(
                train_states["vectors"],
                dim=dim,
                axis=str(train_states.get("axis", "auto")),
                name="train_states",
            )
            return _state_batch_descriptor(batch)
        if kind == "states":
            batch = QuantumStateBatch.from_state_like(
                train_states["states"],
                dim=dim,
                name="train_states",
            )
            return _state_batch_descriptor(batch)
    batch = QuantumStateBatch.from_state_like(train_states, dim=dim, name="train_states")
    return _state_batch_descriptor(batch)


def _test_descriptor(test_state, *, dim: int) -> dict:
    selector, fixed_state, num_points = _resolve_test_state_request(test_state)
    if selector == "fixed_state":
        batch = QuantumStateBatch.from_state_like(
            fixed_state,
            dim=dim,
            name="test_state",
        )
        return {"kind": "fixed_state", "state": _array_payload(batch.states[0])}
    descriptor = {"kind": selector}
    if num_points is not None:
        descriptor["num_points"] = int(num_points)
    return descriptor


def _target_descriptor(observable, *, dim: int, nout: int | None) -> dict:
    if isinstance(observable, str):
        return {"kind": observable.lower()}
    target = np.asarray(observable)
    if target.ndim == 1 and target.shape[0] == dim:
        batch = QuantumStateBatch.from_state_like(
            target,
            dim=dim,
            name="target_observable",
        )
        return {"kind": "pure_state", "operator": _array_payload(batch.states[0])}
    if target.ndim == 1:
        descriptor = {"kind": "outcome_weights", "weights": _array_payload(target)}
        if nout is not None:
            descriptor["nout"] = int(nout)
        return descriptor
    if target.ndim == 2:
        return {"kind": "operator", "operator": _array_payload(target)}
    return {"kind": type(observable).__name__, "value": _selector_descriptor(observable)}


def _tilde_u_saved_raw_df(raw_df: pd.DataFrame, *, sweep_col: str) -> pd.DataFrame:
    metric_cols = {
        "leading_bias_sq_exact": "leading_bias_sq",
        "leading_var_exact": "leading_variance",
        "leading_bias_sq_identity": "identity_leading_bias_sq",
        "leading_var_identity": "identity_leading_variance",
        "actual_mse": "mse",
        "actual_bias_sq": "bias_sq",
        "actual_variance": "variance",
    }
    cols = ["trial"]
    if sweep_col in raw_df.columns:
        cols.append(sweep_col)
    if "failed" in raw_df.columns and raw_df["failed"].fillna(False).any():
        cols.append("failed")
    cols.extend(col for col in metric_cols if col in raw_df.columns)
    return raw_df.loc[:, cols].rename(columns=metric_cols)


def _save_tilde_u_training_approx_report_zip(
    path: Path,
    *,
    raw_df: pd.DataFrame,
    metadata: dict,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tables": {
            "raw": "raw.parquet",
        },
        "metadata": "metadata.json",
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        archive.writestr("raw.parquet", _portable_dataframe_bytes(raw_df))
        archive.writestr(
            "metadata.json",
            json.dumps(metadata, indent=2, sort_keys=True),
        )
    return path


def load_tilde_u_training_approx_report_data(path: str | Path) -> dict:
    """
    Load data saved by run_tilde_u_training_approx_report.

    Portable .zip reports store tables as Parquet and metadata as plain JSON,
    so they are intended for moving between machines and pandas versions.
    """
    report_path = Path(path).expanduser()
    if report_path.suffix.lower() == ".zip":
        with ZipFile(report_path, "r") as archive:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            tables = manifest["tables"]
            return {
                "raw": _read_portable_dataframe_bytes(archive.read(tables["raw"])),
                "metadata": json.loads(
                    archive.read(manifest["metadata"]).decode("utf-8")
                ),
            }
    raise ValueError("tilde-U report path must end in .zip.")


def _tilde_u_training_approx_report_metadata(
    study: TildeUTrainingApproxStudySpec,
    *,
    started_at: datetime,
    completed_at: datetime,
    elapsed_seconds: float,
) -> dict:
    base = study.base
    sweep_values = _tilde_u_study_sweep_values(study)
    data = {
        "d": int(base.data.d),
        "nout": None if base.data.nout is None else int(base.data.nout),
        "povm": _povm_descriptor(base.data.povm),
        "train_states": _training_states_descriptor(
            base.data.train_states,
            dim=int(base.data.d),
        ),
        "train_state_count": _training_state_count_from_spec(base.data.train_states),
    }
    target = {
        "observable": _target_descriptor(
            base.target.observable,
            dim=int(base.data.d),
            nout=base.data.nout,
        ),
    }
    if base.target.normalization != "none":
        target["normalization"] = base.target.normalization
    noise = {
        "noise": base.noise.noise,
        "N": None if base.noise.N is None else int(base.noise.N),
        "actual_noise_trials": int(base.noise.actual_noise_trials),
    }
    if base.noise.lstsq_rcond is not None:
        noise["lstsq_rcond"] = float(base.noise.lstsq_rcond)
    numerics = {
        "rank": None if base.numerics.rank is None else int(base.numerics.rank),
        "rcond": float(base.numerics.rcond),
        "ridge": float(base.numerics.ridge),
    }
    return {
        "created_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
        "sweep_col": study.sweep_col,
        "sweep_values": list(sweep_values),
        "repetitions": study.repetitions,
        "seed": study.seed,
        "data": data,
        "target": target,
        "test": {"state": _test_descriptor(base.test.state, dim=int(base.data.d))},
        "noise": noise,
        "numerics": numerics,
    }


def run_tilde_u_training_approx_report(
    study: TildeUTrainingApproxStudySpec,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Notebook-friendly report for a tilde-U approximation study.
    """
    if study.output_file is not None:
        _tilde_u_report_output_path(study.output_file)

    started_at = datetime.now().astimezone()
    start_time = perf_counter()
    raw_df, summary_df = run_tilde_u_training_approx_experiment(study)
    resolved_x_col = _tilde_u_study_x_col(study)
    slopes_df = fit_tilde_u_training_approx_slopes(
        summary_df,
        x_col=resolved_x_col,
        ycols=study.slope_ycols,
        group_cols=_tilde_u_slope_group_cols(summary_df, resolved_x_col),
    )
    elapsed_seconds = perf_counter() - start_time
    completed_at = datetime.now().astimezone()

    if study.output_file is not None:
        saved_path = _save_tilde_u_training_approx_report_data(
            study.output_file,
            raw_df=_tilde_u_saved_raw_df(raw_df, sweep_col=study.sweep_col),
            metadata=_tilde_u_training_approx_report_metadata(
                study,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=elapsed_seconds,
            ),
            overwrite=study.overwrite,
        )
        if study.verbose:
            print(f"Saved tilde-U report data to {saved_path}")

    if study.show_summary:
        display(summary_df)

    if study.show_slopes:
        display(slopes_df)

    if study.make_plots:
        plots = _tilde_u_training_approx_plots_from_keys(study.plots)
        plot_grouped_mean_median_quantile_summary(
            summary_df,
            x_col=resolved_x_col,
            plots=plots,
            quantile_band=study.quantile_band,
            logx=True,
            logy=True,
        )

    return raw_df, summary_df, slopes_df


def one_schur_correction_trial(
    d: int,
    nout: int,
    ntr: int,
    *,
    Nshots: int = 10_000,
    noise: str = "gaussian",
    seed: int | None = None,
    rcond: float = 1e-12,
    ridge: float = 0.0,
    rank: int | None = None,
) -> dict:
    """
    Run one realization and compute ||C12 C22^{-1} Xi21||_op diagnostics.
    """
    rng = get_rng(seed)

    r = d * d if rank is None else int(rank)
    if nout <= r:
        raise ValueError(f"Need nout > r. Got nout={nout}, r={r}.")
    if ntr <= r:
        raise ValueError(f"Need ntr > r. Got ntr={ntr}, r={r}.")

    povm = POVMEffects.random_isometry(nout=nout, dim=d, rng=rng)
    states = QuantumStateBatch.haar_pure_from_columns(num_states=ntr, dim=d, rng=rng)
    P = povm.probability_matrix(states)

    blocks = svd_probability_blocks(P, rank=r)
    U1 = blocks["U1"]
    U2 = blocks["U2"]
    V1 = blocks["V1"]
    pi2_diag = blocks["Pi2_diag"]

    C12, C22 = schur_covariance_blocks(P, U1, U2, pi2_diag)

    Xi = shot_noise_matrix(P, rng, Nshots=Nshots, noise=noise)
    Xi21 = U2.T @ Xi @ V1

    C22_inv_Xi21, eigvals_C22, kept = psd_solve(
        C22, Xi21, rcond=rcond, ridge=ridge
    )
    term = C12 @ C22_inv_Xi21

    C22_inv_C21, _, _ = psd_solve(C22, C12.T, rcond=rcond, ridge=ridge)
    Gamma = symmetrize(C12 @ C22_inv_C21)

    positive_eigs = eigvals_C22[eigvals_C22 > 0]
    lam_min = np.min(positive_eigs) if positive_eigs.size else np.nan
    lam_max = np.max(eigvals_C22) if eigvals_C22.size else np.nan
    cond_C22 = lam_max / lam_min if lam_min > 0 else np.inf

    return {
        "d": d,
        "r": blocks["r"],
        "nout": nout,
        "ntr": ntr,
        "q": blocks["q"],
        "p_kernel": blocks["p_kernel"],
        "Nshots": Nshots,
        "noise": noise,
        "term_op": opnorm(term),
        "term_fro": frobnorm(term),
        "C12_op": opnorm(C12),
        "Xi21_op": opnorm(Xi21),
        "Gamma_op": opnorm(Gamma),
        "Gamma_trace": np.trace(Gamma),
        "C22_lambda_min": lam_min,
        "C22_lambda_max": lam_max,
        "C22_cond": cond_C22,
        "C22_kept_rank": int(np.sum(kept)),
        "P_numerical_rank": blocks["numerical_rank"],
        "Pi2_diag_min": float(np.min(pi2_diag)),
        "Pi2_diag_max": float(np.max(pi2_diag)),
        "Pi2_diag_mean": float(np.mean(pi2_diag)),
    }


def summarize_schur_correction_trials(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize repeated Schur-correction trials by parameter setting.
    """
    group_cols = ["d", "r", "nout", "ntr", "q", "p_kernel", "Nshots", "noise"]

    summary = (
        raw_df
        .groupby(group_cols, dropna=False)
        .agg(
            trials=("term_op", "size"),
            term_op_mean=("term_op", "mean"),
            term_op_median=("term_op", "median"),
            term_op_q25=("term_op", lambda x: np.quantile(x, 0.25)),
            term_op_q75=("term_op", lambda x: np.quantile(x, 0.75)),
            term_fro_median=("term_fro", "median"),
            C12_op_median=("C12_op", "median"),
            Xi21_op_median=("Xi21_op", "median"),
            Gamma_op_median=("Gamma_op", "median"),
            Gamma_trace_median=("Gamma_trace", "median"),
            C22_lambda_min_median=("C22_lambda_min", "median"),
            C22_cond_median=("C22_cond", "median"),
            C22_kept_rank_min=("C22_kept_rank", "min"),
            P_numerical_rank_min=("P_numerical_rank", "min"),
        )
        .reset_index()
        .sort_values(["nout", "ntr"])
    )

    summary["q_over_p"] = summary["q"] / summary["p_kernel"]
    return summary


def fit_schur_correction_summary_slopes(
    summary_df: pd.DataFrame,
    xcol: str = "q",
    ycols: Sequence[str] = (
        "term_op_median",
        "Gamma_trace_median",
        "Gamma_op_median",
        "C12_op_median",
        "C22_lambda_min_median",
    ),
) -> pd.DataFrame:
    """
    Fit log-log slopes for selected median Schur-correction diagnostics.
    """
    rows = []
    x = summary_df[xcol].to_numpy(dtype=float)

    for ycol in ycols:
        y = summary_df[ycol].to_numpy(dtype=float)
        slope, intercept = loglog_fit(x, y)
        rows.append(
            {
                "x": xcol,
                "y": ycol,
                "slope": slope,
                "intercept": intercept,
                "law": f"{ycol} ~ {xcol}^{slope:.3f}" if np.isfinite(slope) else "insufficient data",
            }
        )

    return pd.DataFrame(rows)


def run_schur_correction_scaling_experiment(
    *,
    d: int = 2,
    nout_values: Sequence[int] = (8, 12, 16, 24, 32, 48, 64),
    ntr_values: Sequence[int] | Callable[[int], int] | None = None,
    ntr_multiplier: float = 50,
    trials: int = 20,
    Nshots: int = 10_000,
    noise: str = "gaussian",
    seed: int | None = None,
    rcond: float = 1e-12,
    ridge: float = 0.0,
    rank: int | None = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sweep over nout and compute Schur-correction scaling diagnostics.
    """
    nout_values = list(nout_values)

    if ntr_values is None:
        ntr_list = [int(np.ceil(ntr_multiplier * nout)) for nout in nout_values]
    elif callable(ntr_values):
        ntr_list = [int(ntr_values(nout)) for nout in nout_values]
    else:
        ntr_list = list(ntr_values)
        if len(ntr_list) != len(nout_values):
            raise ValueError("ntr_values must have the same length as nout_values.")

    raw_df = run_repeated_trials(
        trial_fn=one_schur_correction_trial,
        configs=[
            {"nout": nout, "ntr": ntr}
            for nout, ntr in zip(nout_values, ntr_list)
        ],
        trials=trials,
        seed=seed,
        shared_kwargs={
            "d": d,
            "Nshots": Nshots,
            "noise": noise,
            "rcond": rcond,
            "ridge": ridge,
            "rank": rank,
        },
        seed_arg="seed",
        verbose=verbose,
        progress_fields=("nout", "ntr"),
        fail_soft=True,
    )
    ok = raw_df[~raw_df["failed"].fillna(False)].copy()

    summary_df = summarize_schur_correction_trials(ok)
    slopes_df = fit_schur_correction_summary_slopes(summary_df, xcol="q")

    return raw_df, summary_df, slopes_df


def run_schur_correction_report(
    *,
    sweep_col: str = "nout",
    d: int = 2,
    nout: int = 32,
    nout_values: Sequence[int] = (8, 12, 16, 24, 32, 48, 64),
    ntr_values: Sequence[int] | Callable[[int], int] | None = None,
    ntr_multiplier: float = 50,
    trials: int = 20,
    Nshots: int = 10_000,
    noise: str = "gaussian",
    seed: int | None = None,
    rcond: float = 1e-12,
    ridge: float = 0.0,
    rank: int | None = None,
    verbose: bool = True,
    xcol: str = "q",
    ycols: Tuple[str, ...] = (
        "term_op_median",
        "Gamma_trace_median",
        "Gamma_op_median",
        "C12_op_median",
        "C22_lambda_min_median",
    ),
    show_summary: bool = True,
    show_slopes: bool = True,
    make_plots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a Schur-correction sweep and produce the notebook report.
    """
    if sweep_col == "ntr":
        if ntr_values is None:
            ntr_values = (128, 256, 512, 1024, 2048, 4096)
        nout_values = [nout] * len(ntr_values)
        if xcol == "q":
            xcol = "p_kernel"
    elif sweep_col != "nout":
        raise ValueError("sweep_col must be 'nout' or 'ntr'.")

    raw_df, summary_df, slopes_df = run_schur_correction_scaling_experiment(
        d=d,
        nout_values=nout_values,
        ntr_values=ntr_values,
        ntr_multiplier=ntr_multiplier,
        trials=trials,
        Nshots=Nshots,
        noise=noise,
        seed=seed,
        rcond=rcond,
        ridge=ridge,
        rank=rank,
        verbose=verbose,
    )

    if show_summary:
        display(summary_df)

    if show_slopes:
        display(slopes_df)

    if make_plots:
        plot_summary_series(
            summary_df,
            x_col=xcol,
            plots=[
                {
                    "title": ycol,
                    "ylabel": ycol,
                    "series": [ycol],
                    "annotate_slope": True,
                }
                for ycol in ycols
            ],
            figsize=(5.5, 4.0),
        )

    return raw_df, summary_df, slopes_df
