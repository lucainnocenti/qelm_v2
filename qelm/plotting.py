
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .linalg import loglog_slope, quantile_suffix


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
        ax.legend(loc="lower left")

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


def plot_median_iqr(
    ax,
    df: pd.DataFrame,
    x_col: str,
    base_col: str,
    label: str,
) -> None:
    """
    Plot median with q25-q75 shading from summarized trial columns.
    """
    x = df[x_col].to_numpy(dtype=float)
    y_q25 = df[f"{base_col}_q25"].to_numpy(dtype=float)
    y_med = df[f"{base_col}_median"].to_numpy(dtype=float)
    y_q75 = df[f"{base_col}_q75"].to_numpy(dtype=float)

    line = ax.plot(x, y_med, marker="o", label=f"median {label}")[0]
    ax.fill_between(
        x,
        y_q25,
        y_q75,
        color=line.get_color(),
        alpha=0.18,
        linewidth=0,
        label=f"q25-q75 {label}",
    )


def plot_summary_series(
    summary_df: pd.DataFrame,
    *,
    x_col: str = "ntr",
    plots,
    thresholds: Optional[Dict[str, float]] = None,
    figsize: Tuple[float, float] = (7, 4.5),
    logx: bool = True,
    logy: bool = True,
) -> None:
    """
    Plot summarized series from a DataFrame using declarative plot specs.

    Each plot spec is a dict with title, ylabel, and series. A series entry is
    either "metric_base" or ("metric_base", "label"). If metric_base_q25,
    metric_base_median, and metric_base_q75 exist, the median and IQR are
    plotted. Otherwise metric_base itself is plotted directly.
    """
    thresholds = thresholds or {}

    for spec in plots:
        fig, ax = plt.subplots(figsize=figsize)
        plotted_y_cols = []

        for series in spec["series"]:
            if isinstance(series, str):
                base_col = label = series
            else:
                base_col, label = series

            if f"{base_col}_median" in summary_df.columns:
                plot_median_iqr(ax, summary_df, x_col, base_col, label)
                plotted_y_cols.append(f"{base_col}_median")
            else:
                ax.plot(
                    summary_df[x_col],
                    summary_df[base_col],
                    marker="o",
                    label=label,
                )
                plotted_y_cols.append(base_col)

        threshold_key = spec.get("threshold")
        if threshold_key:
            ax.axhline(
                thresholds[threshold_key],
                color="black",
                linestyle="--",
                linewidth=1,
                label="threshold",
            )

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(x_col)
        ax.set_ylabel(spec["ylabel"])

        title = spec["title"]
        if spec.get("annotate_slope") and len(plotted_y_cols) == 1:
            slope, _ = loglog_slope(summary_df, x_col, plotted_y_cols[0])
            if np.isfinite(slope):
                title += f"    fitted slope: {slope:.3f}"

        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.show()


def plot_mean_median_quantile_summary(
    summary_df: pd.DataFrame,
    *,
    x_col: str,
    metrics,
    quantile_band: Tuple[float, float] = (0.25, 0.75),
    label_col: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4.5),
    logx: bool = True,
    logy: bool = True,
    show_mean: bool = True,
    show_median: bool = True,
    annotate_slope: bool = True,
) -> None:
    """
    Plot summarized metrics with mean, median, and a configurable quantile band.

    metrics entries can be:
        "metric_base"
        ("metric_base", "title and ylabel")
        ("metric_base", "title", "ylabel")

    The summary is expected to contain columns like metric_base_mean,
    metric_base_median, metric_base_q25, metric_base_q75.
    """
    q_low, q_high = quantile_band
    low_suffix = quantile_suffix(q_low)
    high_suffix = quantile_suffix(q_high)

    groups = [(None, summary_df)]
    if label_col is not None and label_col in summary_df.columns:
        groups = list(summary_df.groupby(label_col, dropna=False))

    for metric in metrics:
        if isinstance(metric, str):
            base_col = ylabel = title = metric
        else:
            if len(metric) == 2:
                base_col, ylabel = metric
                title = ylabel
            elif len(metric) == 3:
                base_col, title, ylabel = metric
            else:
                raise ValueError(
                    "Metric specs must be strings, (metric, label), or "
                    "(metric, title, ylabel)."
                )

        fig, ax = plt.subplots(figsize=figsize)
        slope_col = None

        for label, group in groups:
            group = group.sort_values(x_col)
            prefix = "" if label is None else f"{label_col}={label}, "
            x = group[x_col].to_numpy(dtype=float)

            qlo_col = f"{base_col}_{low_suffix}"
            qhi_col = f"{base_col}_{high_suffix}"
            if qlo_col in group.columns and qhi_col in group.columns:
                y_lo = group[qlo_col].to_numpy(dtype=float)
                y_hi = group[qhi_col].to_numpy(dtype=float)
                band = ax.fill_between(
                    x,
                    y_lo,
                    y_hi,
                    alpha=0.14,
                    linewidth=0,
                    label=f"{prefix}{low_suffix}-{high_suffix} across realizations",
                )
                band_color = band.get_facecolor()[0]
            else:
                band_color = None

            if show_median and f"{base_col}_median" in group.columns:
                line = ax.plot(
                    x,
                    group[f"{base_col}_median"],
                    marker="o",
                    label=f"{prefix}median across realizations",
                )[0]
                slope_col = f"{base_col}_median"
                if band_color is not None:
                    line.set_color(band_color)

            if show_mean and f"{base_col}_mean" in group.columns:
                ax.plot(
                    x,
                    group[f"{base_col}_mean"],
                    marker="s",
                    linestyle="--",
                    label=f"{prefix}mean across realizations",
                )

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        plot_title = title
        if annotate_slope and slope_col is not None and label_col is None:
            slope, _ = loglog_slope(summary_df, x_col, slope_col)
            if np.isfinite(slope):
                plot_title += f"    fitted median slope: {slope:.3f}"

        ax.set_title(plot_title)
        x_labels = {
            "N": "training shots per state N",
            "ntr": "number of training states n_tr",
            "nout": "number of POVM outcomes n_out",
        }
        ax.set_xlabel(x_labels.get(x_col, x_col))
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.show()


def _visible_y_limits(x, y_arrays, xlim, *, logy):
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if xlim is not None:
        lo, hi = xlim
        if lo is not None and hi is not None:
            lo, hi = min(lo, hi), max(lo, hi)
        if lo is not None:
            mask &= x >= lo
        if hi is not None:
            mask &= x <= hi

    values = []
    for y in y_arrays:
        y = np.asarray(y, dtype=float)
        visible = y[mask & np.isfinite(y)]
        if logy:
            visible = visible[visible > 0]
        if visible.size:
            values.append(visible)
    if not values:
        return None

    values = np.concatenate(values)
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return None

    if logy:
        if ymin <= 0:
            return None
        if ymin == ymax:
            factor = 1.2
            return ymin / factor, ymax * factor
        log_min = np.log10(ymin)
        log_max = np.log10(ymax)
        pad = 0.05 * (log_max - log_min)
        return 10 ** (log_min - pad), 10 ** (log_max + pad)

    if ymin == ymax:
        pad = 0.05 * abs(ymin) if ymin != 0 else 1.0
    else:
        pad = 0.05 * (ymax - ymin)
    return ymin - pad, ymax + pad


def plot_grouped_mean_median_quantile_summary(
    summary_df,
    *,
    x_col,
    plots,
    quantile_band=(0.25, 0.75),
    logx=True,
    logy=True,
    figsize=(5.5, 4.0),
    show_mean=True,
    show_median=True,
    show_band=True,
    xlim=None,
    ylim=None,
    legend_outside=False,
    ax=None,
):
    """
    Plot several summarized quantities on the same axes.

    Each entry of plots is:
        (
            [(base_col, label), ...],
            title,
            ylabel,
        )

    For each base_col, summary_df should contain:
        base_col_median
        base_col_mean
        base_col_q25, base_col_q75, etc.

    Color distinguishes the physical/model quantity.
    Line style distinguishes statistic:
        solid  = median
        dashed = mean
    """
    qlo, qhi = quantile_band
    qlo_suffix = f"q{int(round(100 * qlo))}"
    qhi_suffix = f"q{int(round(100 * qhi))}"

    x = summary_df[x_col].to_numpy(dtype=float)
    x_labels = {
        "N": r"$N$",
        "ntr": r"$n_{\mathrm{tr}}$",
        "nout": r"$n_{\mathrm{out}}$",
    }
    plots = list(plots)

    if ax is not None and len(plots) != 1:
        raise ValueError("ax can only be used when rendering exactly one plot")

    for series_specs, title, ylabel in plots:
        # fig, ax = plt.subplots(figsize=figsize)
        if ax is None:
            legend_width = 1.8 if legend_outside else 0.0
            fig, plot_ax = plt.subplots(
                figsize=(figsize[0] + legend_width, figsize[1])
            )
        else:
            plot_ax = ax
            fig = plot_ax.figure
        y_for_autoscale = []

        for base, label in series_specs:
            median_col = f"{base}_median"
            mean_col = f"{base}_mean"
            qlo_col = f"{base}_{qlo_suffix}"
            qhi_col = f"{base}_{qhi_suffix}"

            if show_median and median_col not in summary_df.columns:
                raise ValueError(f"Missing column {median_col}")

            color = None
            if show_median:
                y_med = summary_df[median_col].to_numpy(dtype=float)
                y_for_autoscale.append(y_med)
                median_line, = plot_ax.plot(
                    x,
                    y_med,
                    marker="o",
                    linestyle="-",
                    label=f"{label}, median",
                )
                color = median_line.get_color()

            if show_mean and mean_col in summary_df.columns:
                y_mean = summary_df[mean_col].to_numpy(dtype=float)
                y_for_autoscale.append(y_mean)
                mean_line, = plot_ax.plot(
                    x,
                    y_mean,
                    marker="x",
                    linestyle="--",
                    **({} if color is None else {"color": color}),
                    label=f"{label}, mean",
                )
                color = mean_line.get_color()

            if show_band and qlo_col in summary_df.columns and qhi_col in summary_df.columns:
                y_lo = summary_df[qlo_col].to_numpy(dtype=float)
                y_hi = summary_df[qhi_col].to_numpy(dtype=float)
                y_for_autoscale.extend((y_lo, y_hi))
                fill_kwargs = {"alpha": 0.15}
                if color is not None:
                    fill_kwargs["color"] = color
                plot_ax.fill_between(x, y_lo, y_hi, **fill_kwargs)

        plot_ax.set_title(title)
        plot_ax.set_xlabel(x_labels.get(x_col, x_col))
        plot_ax.set_ylabel(ylabel)

        if logx:
            plot_ax.set_xscale("log")
        if logy:
            plot_ax.set_yscale("log")
        if xlim is not None:
            plot_ax.set_xlim(xlim)
        if ylim is not None:
            plot_ax.set_ylim(ylim)
        elif xlim is not None:
            visible_ylim = _visible_y_limits(x, y_for_autoscale, xlim, logy=logy)
            if visible_ylim is not None:
                plot_ax.set_ylim(visible_ylim)

        plot_ax.grid(True, which="both", alpha=0.3)
        if legend_outside:
            plot_ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                framealpha=0.4,
                fontsize="small"
            )
        else:
            plot_ax.legend(framealpha=0.4, fontsize="small")

        if ax is None:
            fig.tight_layout()
            plt.show()
        # ax.legend(framealpha=0.4)
        # fig.tight_layout()
        # plt.show()

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


from pathlib import Path
import math
import matplotlib.pyplot as plt


def _default_log_yticks_and_labels(ymin, ymax):
    """
    Return base-10 ticks between ymin and ymax, with the bottom/top labels hidden.
    Assumes positive log-scale limits.
    """
    kmin = math.ceil(math.log10(ymin))
    kmax = math.floor(math.log10(ymax))

    yticks = [10.0**k for k in range(kmin, kmax + 1)]
    yticklabels = [rf"$10^{{{k}}}$" for k in range(kmin, kmax + 1)]

    if len(yticklabels) >= 1:
        yticklabels[0] = ""
    if len(yticklabels) >= 2:
        yticklabels[-1] = ""

    return yticks, yticklabels


def plot_mse_grid_over_N(
    folder,
    *,
    d=2,
    nout=16,
    n_min=2,
    n_max=13,
    ncols=3,
    xlim=(15, None),
    ylim=(1e-7, 10),
    plots="mse",
    quantile_band=(0.05, 0.95),
    show_mean=False,
    n_realizations=1000,
    figsize_per_panel=(5.0, 3.5),
    title=None,
    filename_template="_d{d}_nout{nout}_N{N}_vsntr.zip",
    plot_func=None,
    title_fontsize=20,
    panel_title_fontsize=14,
    tick_labelsize=12,
    axis_labelsize=18,
    legend_fontsize=14,
    legend_loc="lower left",
    title_y=0.995,
    margins=None,
):
    """
    Plot saved training curves for files indexed by N = 2**n.

    Parameters
    ----------
    folder : str or Path
        Folder containing files of the form specified by filename_template.

    d : int
        Hilbert-space dimension used in filename/title.

    nout : int
        Number of POVM outcomes used in filename/title.

    n_min, n_max : int
        Plot N = 2**n for n_min <= n <= n_max.

    ncols : int
        Number of subplot columns.

    xlim, ylim : tuple
        Axis limits passed to plot_func.

    filename_template : str
        Template for filenames. Available fields: d, nout, N.

    plot_func : callable or None
        Plotting function. If None, uses the global plot_saved_traindata.

    Returns
    -------
    fig, axes
        Matplotlib figure and flattened axes array.
    """
    if plot_func is None:
        from .training_reports import plot_saved_traindata

        plot_func = plot_saved_traindata

    folder = Path(folder)

    Ns = [2**n for n in range(n_min, n_max + 1)]
    paths = [
        folder / filename_template.format(d=d, nout=nout, N=N)
        for N in Ns
    ]

    nrows = math.ceil(len(paths) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True,
        sharey=True,
        gridspec_kw={
            "wspace": 0.0,
            "hspace": 0.0,
        },
    )

    axes = np.asarray(axes).ravel()

    ymin, ymax = ylim
    yticks, yticklabels = _default_log_yticks_and_labels(ymin, ymax)

    for i, (ax, N, path) in enumerate(zip(axes, Ns, paths)):
        row = i // ncols
        col = i % ncols

        plot_func(
            path,
            plots=plots,
            quantile_band=quantile_band,
            xlim=xlim,
            ylim=ylim,
            show_mean=show_mean,
            ax=ax,
        )

        ax.set_title("")

        # Panel title inside plot
        ax.text(
            0.5,
            0.95,
            rf"$N={N}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=panel_title_fontsize,
            bbox=dict(
                facecolor="white",
                alpha=0.75,
                edgecolor="none",
                pad=1.5,
            ),
        )

        ax.tick_params(
            axis="both",
            which="both",
            labelsize=tick_labelsize,
            direction="in",
            top=True,
            right=True,
        )

        # Labels only on outer axes
        if row == nrows - 1:
            ax.xaxis.label.set_size(axis_labelsize)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if col == 0:
            ax.yaxis.label.set_size(axis_labelsize)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        # Keep legend only in the first subplot
        leg = ax.get_legend()

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

            if leg is not None:
                leg.remove()

            if handles and labels:
                ax.legend(
                    handles,
                    labels,
                    fontsize=legend_fontsize,
                    loc=legend_loc,
                    framealpha=0.75,
                )
        else:
            if leg is not None:
                leg.remove()

        # Fixed y-limits and suppressed boundary tick labels
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(yticks)

        if col == 0:
            ax.set_yticklabels(yticklabels)

    # Hide unused axes
    for ax in axes[len(paths):]:
        ax.set_visible(False)

    if title is None:
        quantile_label = ""
        if quantile_band is not None:
            qlo, qhi = quantile_band
            quantile_label = f", q{100 * qlo:g}-q{100 * qhi:g}"

        title = (
            rf"MSE for random rank-1 POVMs, Haar states, "
            rf"$d={d}$, $n_{{\mathrm{{out}}}}={nout}$, "
            rf"${n_realizations}$ realizations{quantile_label}"
        )

    fig.suptitle(
        title,
        fontsize=title_fontsize,
        y=title_y,
    )

    if margins is None:
        margins = dict(
            left=0.065,
            right=0.995,
            bottom=0.055,
            top=0.96,
            wspace=0.0,
            hspace=0.0,
        )

    fig.subplots_adjust(**margins)

    return fig, axes


def plot_leading_mse_difference_grid_over_N(
    folder,
    *,
    d=2,
    nout=16,
    n_min=2,
    n_max=13,
    ncols=3,
    xlim=(15, None),
    ylim=(1e-4, 2e2),
    rescale_power=1,
    quantile_band=(0.05, 0.95),
    show_mean=False,
    n_realizations=1000,
    figsize_per_panel=(5.0, 3.5),
    title=None,
    filename_template="_d{d}_nout{nout}_N{N}_vsntr.zip",
    plot_func=None,
    title_fontsize=20,
    panel_title_fontsize=14,
    tick_labelsize=12,
    axis_labelsize=18,
    legend_fontsize=14,
    legend_loc="lower left",
    title_y=0.995,
    margins=None,
):
    """
    Plot N- or N^2-scaled leading-MSE differences for files indexed by N = 2**n.

    The plotted quantity is
    N**rescale_power * (leading_mse_identity - leading_mse_exact), where
    ``leading_mse_identity`` is the term computed with
    ``tilde U U_1^T = I`` and ``leading_mse_exact`` keeps the full correction.
    """
    plot_keys = {
        1: "leading_mse_delta_N",
        2: "leading_mse_delta_N2",
    }
    if rescale_power not in plot_keys:
        raise ValueError("rescale_power must be 1 or 2.")

    if title is None:
        quantile_label = ""
        if quantile_band is not None:
            qlo, qhi = quantile_band
            quantile_label = f", q{100 * qlo:g}-q{100 * qhi:g}"

        scale_label = "$N$" if rescale_power == 1 else "$N^2$"
        title = (
            rf"{scale_label}-rescaled leading-MSE change from "
            rf"$\tilde U U_1^T=I$, "
            rf"$d={d}$, $n_{{\mathrm{{out}}}}={nout}$, "
            rf"${n_realizations}$ realizations{quantile_label}"
        )

    return plot_mse_grid_over_N(
        folder,
        d=d,
        nout=nout,
        n_min=n_min,
        n_max=n_max,
        ncols=ncols,
        xlim=xlim,
        ylim=ylim,
        plots=plot_keys[rescale_power],
        quantile_band=quantile_band,
        show_mean=show_mean,
        n_realizations=n_realizations,
        figsize_per_panel=figsize_per_panel,
        title=title,
        filename_template=filename_template,
        plot_func=plot_func,
        title_fontsize=title_fontsize,
        panel_title_fontsize=panel_title_fontsize,
        tick_labelsize=tick_labelsize,
        axis_labelsize=axis_labelsize,
        legend_fontsize=legend_fontsize,
        legend_loc=legend_loc,
        title_y=title_y,
        margins=margins,
    )
