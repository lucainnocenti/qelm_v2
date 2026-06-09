
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .linalg import loglog_slope


def scatter_loglog(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    annotate_slope: bool = True,
) -> None:
    """
    Log-log scatter plot with optional fitted slope.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    if label_col is None:
        ax.scatter(df[x_col], df[y_col], alpha=0.8)
    else:
        for label, g in df.groupby(label_col):
            ax.scatter(g[x_col], g[y_col], alpha=0.8, label=f"{label_col}={label}")
        ax.legend()

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)

    if title is not None:
        ax.set_title(title)

    if annotate_slope:
        slope, _ = loglog_slope(df, x_col, y_col)
        ax.text(
            0.05,
            0.95,
            f"log-log slope: {slope:.2f}",
            transform=ax.transAxes,
            va="top",
        )

    ax.grid(True, which="both", alpha=0.3)
    plt.show()

def plot_metric_vs_predictors(summary: pd.DataFrame, quantile: str = "p90") -> None:
    """
    Core collapse plots for the three main quantities.
    """
    plots = [
        (
            "E_Y_shape",
            f"D_Y_{quantile}",
            "Y diagnostic",
            "r / (p lambda_min(C22))",
            f"{quantile} ||Y-I||",
        ),
        (
            "E_Q_shape",
            f"D_Q_{quantile}",
            "Q-term residual diagnostic",
            "delta * (1+c/lambda+c^2/lambda^2)",
            f"{quantile} Q residual",
        ),
        (
            "E_R_schur_shape",
            f"D_R_schur_{quantile}",
            "Schur-remainder diagnostic",
            "sqrt(r) * delta * (1/lambda+c/lambda^2)",
            f"{quantile} R_Schur",
        ),
        (
            "E_full_schur_lead_shape",
            f"D_lead_schur_{quantile}",
            "Schur leading-term diagnostic",
            "sqrt(r) * c/lambda",
            f"{quantile} leading Schur term",
        ),
    ]

    for x_col, y_col, title, xlabel, ylabel in plots:
        if x_col in summary.columns and y_col in summary.columns:
            scatter_loglog(
                summary,
                x_col=x_col,
                y_col=y_col,
                label_col="q" if "q" in summary.columns else None,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
            )

def plot_deterministic_block_scalings(det_df: pd.DataFrame) -> None:
    """
    Plot q*lambda, q*c_p, and q*||pbar||_inf across a sweep.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if "q" not in det_df.columns:
        raise ValueError("det_df must contain column 'q'.")

    for col in ["q*lambda_min_C22", "q*c_p", "q*||pbar||_inf", "q*||pbar-uniform||_inf"]:
        if col in det_df.columns:
            ax.plot(det_df["q"], det_df[col], marker="o", label=col)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("scaled deterministic quantity")
    ax.set_title("Deterministic block scalings")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.show()

def plot_metric_vs_kappa(
    summary: pd.DataFrame,
    kappa_col: str,
    metric_base: str,
    quantiles: Tuple[str, ...] = ("median", "p90"),
    title: Optional[str] = None,
) -> None:
    """
    Plot a metric against kappa_q3 or kappa_q5.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for q, g in summary.groupby("q"):
        g = g.sort_values(kappa_col)

        for quantile in quantiles:
            col = f"{metric_base}_{quantile}"
            if col in g.columns:
                ax.plot(
                    g[kappa_col],
                    g[col],
                    marker="o",
                    label=f"q={q}, {quantile}",
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(kappa_col)
    ax.set_ylabel(metric_base)
    ax.set_title(title or f"{metric_base} vs {kappa_col}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.show()

def plot_failure_rates(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    group_cols: Tuple[str, ...] = ("r", "q", "p"),
) -> pd.DataFrame:
    """
    Compute and plot empirical failure rates for chosen thresholds.

    thresholds example:
        {"D_Y": 0.1, "D_Q": 0.1, "D_R_schur": 0.1}
    """
    rows = []

    for key, g in df.groupby(list(group_cols)):
        row = dict(zip(group_cols, key))
        row["num_trials"] = len(g)

        for metric, threshold in thresholds.items():
            row[f"{metric}_fail_rate"] = float(np.mean(g[metric] > threshold))
            row[f"{metric}_threshold"] = threshold

        rows.append(row)

    fail_df = pd.DataFrame(rows)

    for metric in thresholds:
        col = f"{metric}_fail_rate"

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for q, g in fail_df.groupby("q"):
            if "kappa_q3" in df.columns:
                pass
            g = g.sort_values("p")
            ax.plot(g["p"], g[col], marker="o", label=f"q={q}")

        ax.set_xscale("log")
        ax.set_xlabel("p")
        ax.set_ylabel("empirical failure rate")
        ax.set_title(f"Failure rate for {metric}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.show()

    return fail_df

def plot_sweep_diagnostics(
    summary: pd.DataFrame,
    det_df: pd.DataFrame,
    scaling: str = "q3",
    quantile: str = "p90",
) -> None:
    """
    Make the main sweep plots.
    """
    print("Deterministic scaling plots:")
    plot_deterministic_block_scalings(det_df)

    print("Metric-vs-predictor collapse plots:")
    plot_metric_vs_predictors(summary, quantile=quantile)

    if scaling == "q3":
        kappa_col = "kappa_q3"
    elif scaling == "q5":
        kappa_col = "kappa_q5"
    else:
        kappa_col = "p"

    if kappa_col in summary.columns:
        for metric in ["D_Y", "D_Q", "D_R_schur", "D_lead_schur", "D_full_schur"]:
            plot_metric_vs_kappa(
                summary=summary,
                kappa_col=kappa_col,
                metric_base=metric,
                quantiles=("median", quantile),
                title=f"{metric} vs {kappa_col}",
            )

def compare_q3_q5_collapse(summary: pd.DataFrame, quantile: str = "p90") -> None:
    """
    Compare Schur remainder collapse against q^3 and q^5 kappa variables.
    """
    y_col = f"D_R_schur_{quantile}"

    for x_col, title in [
        ("kappa_q3", "Schur remainder vs p/(r q^3)"),
        ("kappa_q5", "Schur remainder vs p/(r q^5)"),
        ("B_cp_qminus1", "Schur remainder vs r q^2 M / p"),
        ("B_general_worst_cp1", "Schur remainder vs r q^4 M / p"),
        ("E_R_schur_shape", "Schur remainder vs exact predictor"),
    ]:
        if x_col in summary.columns and y_col in summary.columns:
            scatter_loglog(
                summary,
                x_col=x_col,
                y_col=y_col,
                label_col="q",
                title=title,
                xlabel=x_col,
                ylabel=y_col,
            )

def deterministic_cp_scaling_fit(det_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate empirical power laws:
        lambda_min(C22) ~ q^{-alpha_lambda}
        c_p ~ q^{-alpha_c}
        ||pbar||_inf ~ q^{-alpha_pbar}
    """
    rows = []

    for key_cols, g in det_df.groupby(["d", "scaling"], dropna=False):
        d, scaling = key_cols

        for quantity in ["lambda_min_C22", "c_p=||C12||", "||pbar||_inf", "||pbar-uniform||_inf"]:
            if quantity not in g.columns:
                continue

            temp = g[["q", quantity]].dropna()
            temp = temp[(temp["q"] > 0) & (temp[quantity] > 0)]

            if len(temp) < 2:
                alpha = np.nan
                intercept = np.nan
            else:
                slope, intercept = loglog_slope(temp, "q", quantity)
                alpha = -slope

            rows.append({
                "d": d,
                "scaling": scaling,
                "quantity": quantity,
                "estimated_alpha_in_quantity~q^-alpha": alpha,
                "log_intercept": intercept,
            })

    return pd.DataFrame(rows)

def plot_Xi21_markov_check(trials_df: pd.DataFrame) -> None:
    """
    Plot empirical Xi21 scale, ||Xi21||/sqrt(r).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    grouped = trials_df.groupby(["r", "q", "p"], dropna=False)

    xs = []
    med = []
    p90 = []
    labels = []

    for key, g in grouped:
        r, q, p = key
        xs.append(p)
        med.append(np.median(g["norm_Xi21_over_sqrt_r"]))
        p90.append(np.quantile(g["norm_Xi21_over_sqrt_r"], 0.90))
        labels.append(f"r={r}, q={q}")

    xs = np.asarray(xs)
    med = np.asarray(med)
    p90 = np.asarray(p90)

    order = np.argsort(xs)

    ax.scatter(xs[order], med[order], label="median")
    ax.scatter(xs[order], p90[order], label="p90")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("||Xi21|| / sqrt(r)")
    ax.set_title("Markov-scale check for Xi21")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.show()


def plot_random_quantum_scaling(
    df: pd.DataFrame,
    quantities: Tuple[str, ...] = (
        "lambda_min_C22",
        "lambda_max_C22",
        "trace_C22",
        "c_p",
    ),
    x_cols: Tuple[str, ...] = ("nout", "ntr"),
    label_col: str = "d",
) -> None:
    """
    Make quick log-log scatter plots for deterministic random-quantum sweeps.
    """
    for quantity in quantities:
        if quantity not in df.columns:
            continue

        for x_col in x_cols:
            if x_col not in df.columns:
                continue

            scatter_loglog(
                df,
                x_col=x_col,
                y_col=quantity,
                label_col=label_col if label_col in df.columns else None,
                title=f"{quantity} vs {x_col}",
                xlabel=x_col,
                ylabel=quantity,
            )
