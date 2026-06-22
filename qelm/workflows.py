"""Notebook-oriented workflows that compose the lower-level QELM modules.

This module is intentionally orchestration-heavy.  It builds parameter grids,
constructs toy or quantum probability matrices, calls ``qelm.trials`` for
fixed-``P`` shot-noise diagnostics, and calls ``qelm.training`` for actual
QELM training and tilde-U approximation studies.  The numerical primitives live
in ``blocks.py``, ``noise.py``, ``quantum.py``, and ``training.py``; functions
here mainly package those pieces into repeatable experiments and summary
tables for notebooks.
"""

from datetime import datetime
from time import perf_counter
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from .blocks import (
    PBlocks,
    block_report,
    deterministic_blocks_from_P,
    schur_covariance_blocks,
    svd_probability_blocks,
)
from .linalg import (
    frobnorm,
    loglog_fit,
    opnorm,
    psd_solve,
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
    POVM,
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
    _required_noise_N,
    estimate_actual_training_mse,
    estimate_actual_training_mse_target_average,
    leading_training_bias_variance_terms,
    leading_training_bias_variance_terms_target_average,
    tilde_u_correction_operator_diagnostics,
)
from .training_reports import (
    TILDE_U_TRAINING_APPROX_METRIC_COLS,
    TILDE_U_TRAINING_APPROX_PLOT_SPECS,
    TILDE_U_TRAINING_APPROX_PLOTS,
    _array_payload,
    _array_from_payload,
    _as_tilde_u_training_approx_report_data,
    _noncolliding_output_path,
    _normalize_tilde_u_report_metadata,
    _portable_dataframe_bytes,
    _povm_descriptor,
    _read_portable_dataframe_bytes,
    _save_tilde_u_training_approx_report_data,
    _save_tilde_u_training_approx_report_zip,
    _selector_descriptor,
    _state_batch_descriptor,
    _target_descriptor,
    _test_descriptor,
    _tilde_u_configs_from_study,
    _tilde_u_context_average_label,
    _tilde_u_context_povm_label,
    _tilde_u_context_quantile_label,
    _tilde_u_povm_label_from_descriptor,
    _tilde_u_report_output_path,
    _tilde_u_report_rank_from_metadata,
    _tilde_u_report_summary_quantiles,
    _tilde_u_report_x_col,
    _tilde_u_saved_raw_df,
    _tilde_u_saved_report_plot_raw_df,
    _tilde_u_slope_group_cols,
    _tilde_u_study_sweep_values,
    _tilde_u_study_x_col,
    _tilde_u_training_approx_context_metadata,
    _tilde_u_training_approx_plots_from_keys,
    _tilde_u_training_approx_report_metadata,
    _tilde_u_training_context_title_suffix,
    _tilde_u_training_contextualized_plots,
    _training_states_descriptor,
    fit_tilde_u_training_approx_slopes,
    load_tilde_u_training_approx_report_data,
    plot_saved_training_data,
    render_tilde_u_training_approx_report,
    summarize_saved_training_data,
    summarize_tilde_u_training_approx,
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

                    povm = POVM.random_rank1(nout=nout, dim=d, rng=rng)
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
    progress_kwargs: dict | None = None,
    print_elapsed: bool = True,
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
    total_steps = len(configs) * trials
    started = perf_counter()

    pbar = None
    if verbose and tqdm is not None:
        tqdm_kwargs = {
            "total": total_steps,
            "desc": "Running trials",
            "unit": "trial",
            "leave": False,
            "dynamic_ncols": True,
        }
        tqdm_kwargs.update(progress_kwargs or {})
        pbar = tqdm(**tqdm_kwargs)
    elif verbose:
        print(f"Running {total_steps} trial(s)...")

    try:
        for config in configs:
            parts = []
            for field in progress_fields:
                if field in config:
                    value = config[field]
                elif field in shared_kwargs:
                    value = shared_kwargs[field]
                else:
                    continue
                parts.append(f"{field}={value}")

            if pbar is not None and parts:
                pbar.set_postfix_str(", ".join(parts), refresh=True)

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

                if pbar is not None:
                    pbar.update(1)
                elif verbose:
                    completed = len(rows)
                    print(f"\rProgress: {completed}/{total_steps} trial(s)", end="", flush=True)
    finally:
        elapsed = perf_counter() - started
        if pbar is not None:
            pbar.close()
        elif verbose and total_steps > 0:
            print("\r" + " " * 60 + "\r", end="", flush=True)
        if verbose and print_elapsed:
            print(f"Elapsed time: {elapsed:.2f}s")

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
    povm = POVM.random_isometry(nout=nout, dim=d, rng=rng)
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
        "nout": nout,
        "ntr": ntr,
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
    group_cols = ["d", "nout", "ntr", "noise"]
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

    summary = pd.DataFrame(rows).sort_values("ntr").reset_index(drop=True)
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
            x_col="ntr",
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
        "nout": blocks["U1"].shape[0],
        "ntr": ntr,
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


def run_tilde_u_training_approx_experiment(
    study: TildeUTrainingApproxStudySpec,
    *,
    progress_kwargs: dict | None = None,
    print_elapsed: bool = True,
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
        progress_kwargs=progress_kwargs,
        print_elapsed=print_elapsed,
        fail_soft=study.fail_soft,
    )
    summary_df = summarize_tilde_u_training_approx(
        raw_df,
        quantiles=study.quantiles,
    )
    return raw_df, summary_df


def run_tilde_u_training_approx_report(
    study: TildeUTrainingApproxStudySpec,
    *,
    progress_kwargs: dict | None = None,
    quiet: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Notebook-friendly report for a tilde-U approximation study.
    """
    if study.output_file is not None:
        _tilde_u_report_output_path(study.output_file)

    started_at = datetime.now().astimezone()
    start_time = perf_counter()
    raw_df, summary_df = run_tilde_u_training_approx_experiment(
        study,
        progress_kwargs=progress_kwargs,
        print_elapsed=not quiet,
    )
    resolved_x_col = _tilde_u_study_x_col(study)
    slopes_df = fit_tilde_u_training_approx_slopes(
        summary_df,
        x_col=resolved_x_col,
        ycols=study.slope_ycols,
        group_cols=_tilde_u_slope_group_cols(summary_df, resolved_x_col),
    )
    elapsed_seconds = perf_counter() - start_time
    completed_at = datetime.now().astimezone()
    metadata = _tilde_u_training_approx_report_metadata(
        study,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed_seconds,
    )

    if study.output_file is not None:
        saved_path = _save_tilde_u_training_approx_report_data(
            study.output_file,
            raw_df=_tilde_u_saved_raw_df(raw_df, sweep_col=study.sweep_col),
            metadata=metadata,
            overwrite=study.overwrite,
        )
        if study.verbose and not quiet:
            print(f"Saved tilde-U report data to {saved_path}")

    render_tilde_u_training_approx_report(
        summary_df,
        slopes_df,
        metadata=metadata,
        x_col=resolved_x_col,
        plots=study.plots,
        quantile_band=study.quantile_band,
        show_summary=study.show_summary,
        show_slopes=study.show_slopes,
        make_plots=study.make_plots,
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

    povm = POVM.random_isometry(nout=nout, dim=d, rng=rng)
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
        "nout": nout,
        "ntr": ntr,
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
    group_cols = ["d", "nout", "ntr", "Nshots", "noise"]

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

    return summary


def fit_schur_correction_summary_slopes(
    summary_df: pd.DataFrame,
    xcol: str = "nout",
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
    slopes_df = fit_schur_correction_summary_slopes(summary_df, xcol="nout")

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
    xcol: str = "nout",
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
        if xcol == "nout":
            xcol = "ntr"
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
