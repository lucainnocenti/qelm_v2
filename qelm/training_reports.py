"""Training-result loading, summaries, and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Callable, Sequence, Tuple
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .linalg import distribution_summary, loglog_fit, quantile_suffix
from .plotting import plot_grouped_mean_median_quantile_summary
from .quantum import qubit_mub_povm


@dataclass(frozen=True)
class MetricExpr:
    """A derived metric that can be requested by summaries or plot specs.

    ``deps`` names the raw or derived columns required by ``compute``. Use this
    for ad hoc quantities without permanently adding them to the built-in metric
    registry.
    """

    name: str
    deps: tuple[str, ...]
    compute: Callable[[pd.DataFrame], object]


# Ordered public metric schema used by summarize_dataraw(), fit_summary_slopes(),
# and the saved-report plotting helpers. Each string is a DataFrame column name:
# raw columns may come directly from a simulation report, while derived columns
# such as leading_mse or mse_identity_over_exact are reconstructed from the
# MetricExpr registry below when TrainingReport.expanded_df() loads compact saved
# data. The order controls display/table ordering; it is not a dependency order.
TRAINING_METRIC_COLS = [
    "C22_inv_C21_op",
    "correction_op",
    "C22_lambda_min",
    "C22_cond",
    "leading_bias2",
    "leading_bias2_identity",
    "bias2_delta",
    "leading_var",
    "leading_var_identity",
    "variance_delta",
    "bias2_delta_times_N2",
    "variance_delta_times_N_ntr",
    "leading_mse",
    "leading_mse_identity",
    "mse",
    "actual_bias_sq",
    "actual_variance",
    "bias2_identity_over_exact",
    "var_identity_over_exact",
    "mse_identity_over_exact",
    "actual_over_leading_exact",
    "actual_over_leading_identity",
    "leading_exact_relative_error",
    "leading_identity_relative_error",
    "leading_mse_identity_minus_exact_times_N",
    "leading_mse_identity_minus_exact_times_N2",
]

# Registry of named training plots. The dict key is the user-facing plot selector
# accepted by plot_saved_traindata(plots=...). Each value is a 3-tuple:
#   (series_specs, title, ylabel)
# where series_specs is a list of (metric, legend_label) pairs. A metric can be a
# plain DataFrame column name or a MetricExpr for an ad hoc computed quantity.
# MetricExpr objects are materialized into raw data before summarization, then
# replaced by their .name before calling plot_grouped_mean_median_quantile_summary().
TRAINING_PLOT_SPECS = {
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
            ("leading_bias2", "with correction"),
            ("leading_bias2_identity", r"$\tilde U U_1^T = I$"),
        ],
        "Leading training squared bias",
        r"$\text{bias}^2_{\text{train}}$",
    ),
    "bias2_delta": (
        [
            ("bias2_delta", "with correction - identity"),
        ],
        "Full minus approx leading training squared bias",
        r"$\Delta\text{bias}^2_{\text{train}}$",
    ),
    "variance": (
        [
            ("leading_var", "with correction"),
            ("leading_var_identity", r"$\tilde U U_1^T = I$"),
        ],
        "Leading training variance",
        r"$\text{variance}_{\text{train}}$",
    ),
    "variance_delta": (
        [
            ("variance_delta", "with correction - identity"),
        ],
        "Full minus approx leading training variance",
        r"$\Delta\text{variance}_{\text{train}}$",
    ),
    "leading_deltas": (
        [
            ("bias2_delta", r"$\Delta\mathrm{bias}^2$"),
            ("variance_delta", r"$\Delta\mathrm{variance}$"),
        ],
        "Full minus approx leading training terms",
        r"$\Delta\text{leading term}_{\text{train}}$",
    ),
    "leading_deltas_rescaled": (
        [
            ("bias2_delta_times_N2", r"$N^2\Delta\mathrm{bias}^2$"),
            ("variance_delta_times_N_ntr", r"$Nn_{\mathrm{tr}}\Delta\mathrm{variance}$"),
        ],
        "Rescaled |full - approx| leading terms",
        r"rescaled $|\Delta\text{leading term}_{\text{train}}|$",
    ),
    "leading_mse": (
        [
            ("leading_mse", "with correction"),
            ("leading_mse_identity", r"$\tilde U U_1^T = I$"),
        ],
        "Leading bias-plus-variance prediction",
        r"$\text{MSE}_{\text{leading}}$",
    ),
    "mse": (
        [
            ("leading_mse", r"leading, full $\tilde U U_1^T$"),
            ("leading_mse_identity", r"leading, $\tilde U U_1^T = I$"),
            ("mse", r"$\mathrm{MSE}(N,\infty)$"),
        ],
        "MSE vs leading terms",
        r"$\mathrm{MSE}(N,\infty)$",
    ),
    "mse_ratio": (
        [("mse_identity_over_exact", r"identity / corrected")],
        "Effect of dropping the C22 correction in the leading MSE",
        "identity leading MSE / corrected leading MSE",
    ),
    "leading_mse_delta_N": (
        [
            (
                "leading_mse_identity_minus_exact_times_N",
                r"$N(\mathrm{identity}-\mathrm{corrected})$",
            )
        ],
        r"Leading-MSE change from $\tilde U U_1^T = I$",
        (
            r"$N\left(\mathrm{MSE}_{\tilde U U_1^T=I}^{\mathrm{lead}}"
            r"-\mathrm{MSE}_{\mathrm{full}}^{\mathrm{lead}}\right)$"
        ),
    ),
    "leading_mse_delta_N2": (
        [
            (
                "leading_mse_identity_minus_exact_times_N2",
                r"$N^2(\mathrm{identity}-\mathrm{corrected})$",
            )
        ],
        r"$N^2$-rescaled leading-MSE change from $\tilde U U_1^T = I$",
        (
            r"$N^2\left(\mathrm{MSE}_{\tilde U U_1^T=I}^{\mathrm{lead}}"
            r"-\mathrm{MSE}_{\mathrm{full}}^{\mathrm{lead}}\right)$"
        ),
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
# Backward-compatible list form of TRAINING_PLOT_SPECS values. Passing plots=None
# resolves to this list, and older callers can still pass these already-expanded
# (series_specs, title, ylabel) tuples instead of named plot keys.
TRAINING_PLOTS = list(TRAINING_PLOT_SPECS.values())

NOISE_MSE_COLS = (
    "mse_multinomial",
    "mse_gaussian",
    "mse_centered_gaussian",
)
NOISE_MSE_LABELS = {
    "mse_multinomial": "multinomial",
    "mse_gaussian": "gaussian",
    "mse_centered_gaussian": "centered gaussian",
}

# Internal-to-saved metric-name mapping for report serialization. Keys are the
# canonical names used everywhere in this module; values are the shorter column
# names found in saved JSON/ZIP reports. TrainingReport.expanded_df() applies the
# reverse map (_COMPACT_TO_INTERNAL_METRIC_COLS) so downstream code only sees the
# canonical names, regardless of how the report was stored on disk.
TRAINING_SAVED_METRIC_COLS = {
    "leading_bias2": "leading_bias2",
    "leading_var": "leading_variance",
    "leading_bias2_identity": "identity_leading_bias2",
    "leading_var_identity": "identity_leading_variance",
    "mse": "mse",
    "actual_bias_sq": "bias_sq",
    "actual_variance": "variance",
}

# Private aliases for loader internals. _SAVED_METRIC_COLS names the map used by
# the report loader, while _COMPACT_TO_INTERNAL_METRIC_COLS reverses it from
# saved-column -> canonical-column for fast column renaming during expansion.
_SAVED_METRIC_COLS = TRAINING_SAVED_METRIC_COLS
_COMPACT_TO_INTERNAL_METRIC_COLS = {
    compact: internal for internal, compact in _SAVED_METRIC_COLS.items()
}
_METRIC_ALIASES = {
    "leading_var_exact": "leading_var",
}

# Small positive floor used by _safe_denominator(). Ratios and relative errors can
# be plotted on log scales, so denominators at or below machine-zero are clamped
# to this value instead of producing infinities or division-by-zero warnings.
_EPS = 1e-15

# Built-in derived metric registry. Each key is the output column name, and each
# MetricExpr stores the input column dependencies plus the computation used by
# _ensure_metric_column(). Dependencies may themselves be derived metrics, so this
# single registry replaces the older operation-specific maps and dependency map.
# New named derived metrics can be added here, while one-off plots can pass a
# MetricExpr directly in their series_specs without changing any module constant.
DERIVED_TRAINING_METRICS = {
    "leading_mse": MetricExpr(
        "leading_mse",
        ("leading_bias2", "leading_var"),
        lambda df: df["leading_bias2"] + df["leading_var"],
    ),
    "leading_mse_identity": MetricExpr(
        "leading_mse_identity",
        ("leading_bias2_identity", "leading_var_identity"),
        lambda df: df["leading_bias2_identity"] + df["leading_var_identity"],
    ),
    "bias2_identity_over_exact": MetricExpr(
        "bias2_identity_over_exact",
        ("leading_bias2_identity", "leading_bias2"),
        lambda df: (
            df["leading_bias2_identity"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_bias2"])
        ),
    ),
    "bias2_delta": MetricExpr(
        "bias2_delta",
        ("leading_bias2", "leading_bias2_identity"),
        lambda df: (
            np.abs(df["leading_bias2"].to_numpy(dtype=float)
            - df["leading_bias2_identity"].to_numpy(dtype=float))
        ),
    ),
    "variance_delta": MetricExpr(
        "variance_delta",
        ("leading_var", "leading_var_identity"),
        lambda df: (
            np.abs(df["leading_var"].to_numpy(dtype=float)
            - df["leading_var_identity"].to_numpy(dtype=float))
        ),
    ),
    "bias2_delta_times_N2": MetricExpr(
        "bias2_delta_times_N2",
        ("bias2_delta", "N"),
        lambda df: (
            df["N"].to_numpy(dtype=float) ** 2
            * df["bias2_delta"].to_numpy(dtype=float)
        ),
    ),
    "variance_delta_times_N_ntr": MetricExpr(
        "variance_delta_times_N_ntr",
        ("variance_delta", "N", "ntr"),
        lambda df: (
            df["N"].to_numpy(dtype=float)
            * df["ntr"].to_numpy(dtype=float)
            * df["variance_delta"].to_numpy(dtype=float)
        ),
    ),
    "var_identity_over_exact": MetricExpr(
        "var_identity_over_exact",
        ("leading_var_identity", "leading_var"),
        lambda df: (
            df["leading_var_identity"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_var"])
        ),
    ),
    "mse_identity_over_exact": MetricExpr(
        "mse_identity_over_exact",
        ("leading_mse_identity", "leading_mse"),
        lambda df: (
            df["leading_mse_identity"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_mse"])
        ),
    ),
    "actual_over_leading_exact": MetricExpr(
        "actual_over_leading_exact",
        ("mse", "leading_mse"),
        lambda df: (
            df["mse"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_mse"])
        ),
    ),
    "actual_over_leading_identity": MetricExpr(
        "actual_over_leading_identity",
        ("mse", "leading_mse_identity"),
        lambda df: (
            df["mse"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_mse_identity"])
        ),
    ),
    "leading_exact_relative_error": MetricExpr(
        "leading_exact_relative_error",
        ("leading_mse", "mse"),
        lambda df: (
            np.abs(
                df["leading_mse"].to_numpy(dtype=float)
                - df["mse"].to_numpy(dtype=float)
            )
            / _safe_denominator(df["mse"])
        ),
    ),
    "leading_identity_relative_error": MetricExpr(
        "leading_identity_relative_error",
        ("leading_mse_identity", "mse"),
        lambda df: (
            np.abs(
                df["leading_mse_identity"].to_numpy(dtype=float)
                - df["mse"].to_numpy(dtype=float)
            )
            / _safe_denominator(df["mse"])
        ),
    ),
    "leading_mse_identity_minus_exact_times_N": MetricExpr(
        "leading_mse_identity_minus_exact_times_N",
        ("leading_mse_identity", "leading_mse", "N"),
        lambda df: (
            df["N"].to_numpy(dtype=float)
            * (
                df["leading_mse_identity"].to_numpy(dtype=float)
                - df["leading_mse"].to_numpy(dtype=float)
            )
        ),
    ),
    "leading_mse_identity_minus_exact_times_N2": MetricExpr(
        "leading_mse_identity_minus_exact_times_N2",
        ("leading_mse_identity", "leading_mse", "N"),
        lambda df: (
            df["N"].to_numpy(dtype=float) ** 2
            * (
                df["leading_mse_identity"].to_numpy(dtype=float)
                - df["leading_mse"].to_numpy(dtype=float)
            )
        ),
    ),
}

# Report-dimension columns written by older raw-data formats. Current reports store
# these values in metadata and TrainingReport.expanded_df() re-adds them as needed;
# _drop_legacy_report_dimension_cols() removes stale in-table copies first to keep
# grouping keys unambiguous.
_LEGACY_REPORT_DIMENSION_COLS = ("r", "q", "p_kernel")


@dataclass(frozen=True)
class TrainingReport:
    """Raw training result data plus metadata loaded from or saved to a report."""

    data: pd.DataFrame
    metadata: dict

    @property
    def datadict(self) -> dict:
        """Return a plain dict representation with data and metadata."""
        return {"data": self.data, "metadata": self.metadata}

    def expanded_df(
        self,
        *,
        metric_cols: Sequence[str | MetricExpr] | None = None,
    ) -> pd.DataFrame:
        """Return raw rows with metadata constants and requested derived metrics.

        Saved reports keep repeated dimensions such as ``N`` or ``ntr`` in
        metadata when possible. This method restores those columns so the rows
        can be summarized or inspected as a normal DataFrame.
        """
        raw_df = _drop_legacy_report_dimension_cols(self.data).copy()

        for compact_col, internal_col in _COMPACT_TO_INTERNAL_METRIC_COLS.items():
            if compact_col in raw_df.columns and internal_col not in raw_df.columns:
                raw_df[internal_col] = raw_df[compact_col]
        for old_col, new_col in _METRIC_ALIASES.items():
            if old_col in raw_df.columns and new_col not in raw_df.columns:
                raw_df[new_col] = raw_df[old_col]

        data = self.metadata.get("data", {})
        noise = self.metadata.get("noise", {})
        target = self.metadata.get("target", {})
        test = self.metadata.get("test", {})
        constant_cols = {
            "d": data.get("d"),
            "nout": data.get("nout"),
            "ntr": data.get("train_state_count"),
            "N": noise.get("N"),
            "noise": noise.get("noise"),
            "test_state": test.get("state", {}).get("kind"),
            "target_kind": target.get("observable", {}).get("kind"),
            "target_normalization": target.get("normalization", "none"),
        }

        for col, value in constant_cols.items():
            if col not in raw_df.columns and value is not None:
                raw_df[col] = value
        _add_derived_saved_report_metrics(raw_df, metric_cols=metric_cols)

        return raw_df

    def summarize(
        self,
        *,
        quantiles: Sequence[float] | None = None,
        quantile_band: tuple[float, float] | None = (0.25, 0.75),
        x_col: str | None = None,
        metric_cols: Sequence[str | MetricExpr] | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return expanded raw rows, grouped summaries, and fitted slopes."""
        return summarize_traindata(
            self,
            quantiles=quantiles,
            quantile_band=quantile_band,
            x_col=x_col,
            metric_cols=metric_cols,
        )

    def plot(
        self,
        **kwargs,
    ) -> None:
        """Plot this report. Use ``summarize()`` when data tables are needed."""
        plot_saved_traindata(self, **kwargs)


def _drop_legacy_report_dimension_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove obsolete saved traindata dimension columns before summarizing data."""
    return df.drop(columns=list(_LEGACY_REPORT_DIMENSION_COLS), errors="ignore")


def _training_plots_from_keys(
    plots: Sequence | str | None,
) -> list[tuple]:
    """Resolve short plot keys or already-expanded plot specs."""
    if plots is None:
        return []
    if isinstance(plots, str):
        items = (plots,)
    elif _is_training_plot_spec(plots):
        return [plots]
    else:
        items = tuple(plots)

    if not items:
        return TRAINING_PLOTS

    if any(item == "all" for item in items if isinstance(item, str)):
        return TRAINING_PLOTS

    resolved = []
    unknown = []
    for item in items:
        if _is_training_plot_spec(item):
            resolved.append(item)
        elif isinstance(item, str) and item in TRAINING_PLOT_SPECS:
            resolved.append(TRAINING_PLOT_SPECS[item])
        else:
            unknown.append(item)

    if unknown:
        available = ", ".join(("all", *TRAINING_PLOT_SPECS))
        raise ValueError(f"Unknown training plot key(s): {unknown}. Available keys: {available}.")

    return resolved


def _is_training_plot_spec(value: object) -> bool:
    """Return whether value has the resolved training plot tuple shape."""
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], (list, tuple))
    )


def _metric_name(metric: str | MetricExpr) -> str:
    """Return the concrete DataFrame column name for a metric spec."""
    name = metric.name if isinstance(metric, MetricExpr) else metric
    return _METRIC_ALIASES.get(name, name)


def _metric_specs_from_plot_specs(plots: Sequence[tuple]) -> tuple[str | MetricExpr, ...]:
    """Extract unique metric specs used by resolved training plot specs."""
    metric_specs = []
    seen = set()
    for series_specs, _title, _ylabel in plots:
        for metric, _label in series_specs:
            name = _metric_name(metric)
            if name not in seen:
                metric_specs.append(metric)
                seen.add(name)
    return tuple(metric_specs)


def _plot_specs_with_metric_names(plots: Sequence[tuple]) -> list[tuple]:
    """Replace MetricExpr entries in plot specs with concrete column names."""
    resolved = []
    for series_specs, title, ylabel in plots:
        resolved.append(
            (
                [(_metric_name(metric), label) for metric, label in series_specs],
                title,
                ylabel,
            )
        )
    return resolved


def _context_povm_label(data: dict, sweep_col: str | None) -> str:
    """Build the POVM label shown in contextualized training plot titles."""
    povm = data.get("povm", {})
    kind = str(povm.get("kind", "unknown"))
    label = _povm_label_from_descriptor(povm)
    parts = [f"POVM={label}"]

    if kind == "random_rank1":
        nout = "sweep" if sweep_col == "nout" else data.get("nout")
        if nout is not None:
            parts.append(f"nout={nout}")

    return ", ".join(parts)


def _array_from_payload(payload: dict) -> np.ndarray:
    """Decode an array payload stored in saved report metadata."""
    if payload.get("kind") != "ndarray":
        raise ValueError("Array payload kind must be 'ndarray'.")
    if "real" in payload and "imag" in payload:
        array = np.asarray(payload["real"]) + 1j * np.asarray(payload["imag"])
    else:
        array = np.asarray(payload["data"])
    return array.astype(payload.get("dtype", array.dtype), copy=False)


def _povm_label_from_descriptor(povm: dict) -> str:
    """Infer a human-readable POVM label from saved report metadata."""
    label = povm.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()

    kind = str(povm.get("kind", "unknown"))
    if kind == "qubit_mub":
        return "qubit_mub"
    if kind == "explicit" and "effects" in povm:
        try:
            effects = _array_from_payload(povm["effects"])
        except (KeyError, TypeError, ValueError):
            effects = None
        if effects is not None and effects.shape == (6, 2, 2):
            if np.allclose(effects, qubit_mub_povm(), atol=1e-12):
                return "qubit_mub"

    return kind


def _context_average_label(kind: object) -> str:
    """Convert metadata selector kinds into compact plot-title labels."""
    kind = "unknown" if kind is None else str(kind)
    if kind in {"fixed_state", "operator", "pure_state", "outcome_weights", "explicit"}:
        return "fixed"
    if kind == "haar_pure_average":
        return "haar"
    if kind == "haar_sample":
        return "haar sample"
    if kind == "random_haar_pure_state":
        return "fixed haar state"
    return kind


def _context_quantile_label(
    quantile_band: tuple[float, float] | None,
) -> str | None:
    """Format a quantile band for use in a training plot title."""
    if quantile_band is None:
        return None
    qlo, qhi = quantile_band
    return f"{quantile_suffix(qlo)}-{quantile_suffix(qhi)}"


def _training_context_title_suffix(
    metadata: dict,
    *,
    quantile_band: tuple[float, float] | None,
) -> str:
    """Build the report-context suffix appended to saved and live MSE plots."""
    sweep_col = metadata.get("sweep_col")
    data = metadata.get("data", {})
    noise = metadata.get("noise", {})
    test = metadata.get("test", {}).get("state", {})
    target = metadata.get("target", {}).get("observable", {})

    first_line = [_context_povm_label(data, sweep_col)]

    ntr = data.get("train_state_count")
    if sweep_col != "ntr" and ntr is not None:
        first_line.append(f"ntr={ntr}")

    N = noise.get("N")
    if sweep_col != "N" and N is not None:
        first_line.append(f"N={N}")

    second_line = [
        f"test={_context_average_label(test.get('kind'))}",
        f"target={_context_average_label(target.get('kind'))}",
    ]

    quantiles = _context_quantile_label(quantile_band)
    if quantiles is not None:
        second_line.append(quantiles)

    noise_kind = noise.get("noise")
    if noise_kind is not None:
        second_line.append(f"noise={noise_kind}")

    return "\n".join(
        "; ".join(str(part) for part in line if part)
        for line in (first_line, second_line)
        if line
    )


def _contextualized_training_plots(
    plots: Sequence[tuple],
    *,
    title_suffix: str | None,
) -> list[tuple]:
    """Attach report context to the MSE plot while preserving other plot specs."""
    if not title_suffix:
        return list(plots)

    mse_title = TRAINING_PLOT_SPECS["mse"][1]
    contextualized = []
    for series_specs, title, ylabel in plots:
        if title == mse_title:
            title = f"{title}\n{title_suffix}"
        contextualized.append((series_specs, title, ylabel))
    return contextualized


def _training_plots_with_title(
    plots: Sequence[tuple],
    *,
    title: str | None,
) -> list[tuple]:
    """Override resolved training plot titles when an explicit title is given."""
    if title is None:
        return list(plots)
    return [(series_specs, title, ylabel) for series_specs, _title, ylabel in plots]


def summarize_dataraw(
    raw_df: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.25, 0.75),
    metric_cols: Sequence[str | MetricExpr] | None = None,
    group_cols: Sequence[str] = (
        "d",
        "nout",
        "ntr",
        "N",
        "noise",
        "test_state",
        "target_kind",
        "target_average",
        "target_normalization",
    ),
) -> pd.DataFrame:
    """Summarize already-expanded training trial rows by report dimensions."""
    rows = []
    ok = raw_df[~raw_df.get("failed", False).astype(bool)].copy() if "failed" in raw_df else raw_df
    for old_col, new_col in _METRIC_ALIASES.items():
        if old_col in ok.columns and new_col not in ok.columns:
            ok = ok.copy()
            ok[new_col] = ok[old_col]
    active_group_cols = tuple(col for col in group_cols if col in ok.columns)
    active_metric_cols = _summary_metric_cols(metric_cols)

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

        for col in active_metric_cols:
            if col not in group.columns:
                continue
            stats = distribution_summary(group[col].to_numpy(dtype=float), quantiles=quantiles)
            for name, value in stats.items():
                row[f"{col}_{name}"] = value

        rows.append(row)

    return pd.DataFrame(rows).sort_values(list(active_group_cols)).reset_index(drop=True)


def fit_summary_slopes(
    summary_df: pd.DataFrame,
    *,
    x_col: str = "ntr",
    ycols: Sequence[str] = (
        "C22_inv_C21_op_median",
        "correction_op_median",
        "leading_mse_median",
        "leading_mse_identity_median",
        "mse_median",
    ),
    group_cols: Sequence[str] = ("d", "nout", "N", "noise"),
) -> pd.DataFrame:
    """
    Fit log-log slopes for selected summarized training quantities.
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


def _slope_group_cols(summary_df: pd.DataFrame, x_col: str) -> tuple[str, ...]:
    """Choose grouping columns for fitting training-result log-log slopes."""
    excluded = {x_col}

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


def _read_portable_dataframe_bytes(data: bytes) -> pd.DataFrame:
    """Read a Parquet table from a portable report archive."""
    return _drop_legacy_report_dimension_cols(pd.read_parquet(BytesIO(data)))


def _normalize_metadata(metadata: dict) -> dict:
    """Normalize saved metadata and repair labels missing from older reports."""
    normalized = json.loads(json.dumps(metadata))
    povm = normalized.get("data", {}).get("povm")
    if isinstance(povm, dict):
        label = povm.get("label")
        if not isinstance(label, str) or not label.strip():
            inferred = _povm_label_from_descriptor(povm)
            if inferred not in {"unknown", "explicit"}:
                povm["label"] = inferred
    return normalized


def load_traindata(path: str | Path) -> TrainingReport:
    """
    Load a portable training report without expanding or summarizing it.

    Portable .zip reports store tables as Parquet and metadata as plain JSON,
    so they are intended for moving between machines and pandas versions. Call
    ``TrainingReport.expanded_df()`` or ``summarize_traindata()`` when you need
    analysis-ready DataFrames.
    """
    report_path = Path(path).expanduser()
    if report_path.suffix.lower() == ".zip":
        with ZipFile(report_path, "r") as archive:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            tables = manifest["tables"]
            return TrainingReport(
                data=_read_portable_dataframe_bytes(archive.read(tables["raw"])),
                metadata=_normalize_metadata(
                    json.loads(archive.read(manifest["metadata"]).decode("utf-8"))
                ),
        )
    raise ValueError("training report path must end in .zip.")


def _scan_file_ntr(path: Path) -> int:
    parts = path.stem.split("_")
    for part in parts:
        try:
            return int(part)
        except ValueError:
            continue
    return 0


def load_noise_mse_scan_files(
    scan_path: str | Path,
    *,
    pattern: str = "ntr_*_vsN.zip",
    skip_unreadable: bool = True,
) -> pd.DataFrame:
    """Load noise-model MSE scan ZIPs into one raw DataFrame."""
    scan_path = Path(scan_path).expanduser()
    if scan_path.is_file():
        paths = [scan_path]
    else:
        paths = sorted(scan_path.glob(pattern), key=_scan_file_ntr)

    if not paths:
        raise FileNotFoundError(
            f"No scan ZIPs matching {pattern!r} found directly in {scan_path}."
        )

    frames = []
    for path in paths:
        try:
            frame = load_traindata(path).data.copy()
        except (BadZipFile, EOFError, OSError, ValueError):
            if skip_unreadable:
                continue
            raise
        frame["scan_file"] = str(path)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No readable scan ZIPs found in {scan_path}.")

    return pd.concat(frames, ignore_index=True)


def _noise_mse_scan_summary_data(
    scan_path: str | Path,
    *,
    scale_by_N: bool = False,
    quantiles: Sequence[float] | None = None,
    quantile_band: tuple[float, float] = (0.10, 0.90),
    skip_unreadable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], str]:
    raw_df = load_noise_mse_scan_files(
        scan_path,
        skip_unreadable=skip_unreadable,
    )
    if "N" not in raw_df.columns or "ntr" not in raw_df.columns:
        raise ValueError("Scan data must contain N and ntr columns.")

    metric_cols = [col for col in NOISE_MSE_COLS if col in raw_df.columns]
    if not metric_cols:
        raise ValueError(f"Scan data has none of the expected columns: {NOISE_MSE_COLS}.")

    plot_df = raw_df.copy()
    if scale_by_N:
        scaled_cols = []
        for col in metric_cols:
            scaled_col = f"{col}_times_N"
            plot_df[scaled_col] = (
                plot_df[col].to_numpy(dtype=float)
                * plot_df["N"].to_numpy(dtype=float)
            )
            scaled_cols.append(scaled_col)
        labels = [
            NOISE_MSE_LABELS[col.removesuffix("_times_N")]
            for col in scaled_cols
        ]
        metric_cols = scaled_cols
        ylabel = r"$N\,\mathrm{MSE}$"
    else:
        labels = [NOISE_MSE_LABELS[col] for col in metric_cols]
        ylabel = "MSE"

    summary_df = summarize_dataraw(
        plot_df,
        quantiles=_summary_quantiles(quantiles, quantile_band),
        metric_cols=metric_cols,
        group_cols=("ntr", "N"),
    )
    return raw_df, summary_df, metric_cols, labels, ylabel


def plot_noise_mse_scan_files(
    scan_path: str | Path,
    *,
    scale_by_N: bool = False,
    panel_by: str = "ntr",
    quantiles: Sequence[float] | None = None,
    quantile_band: tuple[float, float] = (0.10, 0.90),
    show_mean: bool = False,
    show_median: bool = True,
    show_band: bool = True,
    logx: bool = True,
    logy: bool = True,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (5.5, 4.0),
    legend_outside: bool = False,
    title: str | None = None,
    skip_unreadable: bool = True,
) -> None:
    """
    Plot the three actual-MSE noise-model curves from scan ZIPs.

    ``scan_path`` must be either one scan ZIP or a directory containing scan
    ZIPs matching ``ntr_*_vsN.zip``.

    ``panel_by="ntr"`` plots MSE vs N in one panel per ntr value.
    ``panel_by="N"`` plots MSE vs ntr in one panel per N value.
    """
    import matplotlib.pyplot as plt

    if panel_by not in {"ntr", "N"}:
        raise ValueError("panel_by must be 'ntr' or 'N'.")

    _raw_df, summary_df, metric_cols, labels, ylabel = _noise_mse_scan_summary_data(
        scan_path,
        scale_by_N=scale_by_N,
        quantiles=quantiles,
        quantile_band=quantile_band,
        skip_unreadable=skip_unreadable,
    )

    x_col = "N" if panel_by == "ntr" else "ntr"
    panel_values = sorted(int(value) for value in summary_df[panel_by].dropna().unique())
    if not panel_values:
        raise ValueError(f"No {panel_by} values found in scan data.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(panel_values) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    plot_specs = [
        (
            list(zip(metric_cols, labels)),
            "Actual MSE by noise model",
            ylabel,
        )
    ]
    for ax, panel_value in zip(axes.ravel(), panel_values):
        panel = summary_df[summary_df[panel_by] == panel_value].sort_values(x_col)
        plot_grouped_mean_median_quantile_summary(
            panel,
            x_col=x_col,
            plots=plot_specs,
            quantile_band=quantile_band,
            logx=logx,
            logy=logy,
            show_mean=show_mean,
            show_median=show_median,
            show_band=show_band,
            xlim=xlim,
            ylim=ylim,
            legend_outside=legend_outside,
            ax=ax,
        )
        ax.set_title(f"{panel_by}={panel_value}")

    for ax in axes.ravel()[len(panel_values):]:
        ax.set_visible(False)

    if title is None:
        title = f"Scan actual MSEs vs {x_col}"
        if scale_by_N:
            title += r" ($N\cdot\,\mathrm{MSE}$)"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    # fig.tight_layout()
    plt.show()


def _as_traindata(report: str | Path | dict | TrainingReport) -> TrainingReport:
    """Normalize a report path, dict, or TrainingReport into TrainingReport."""
    if isinstance(report, (str, Path)):
        return load_traindata(report)
    if isinstance(report, TrainingReport):
        return TrainingReport(
            data=_drop_legacy_report_dimension_cols(report.data),
            metadata=_normalize_metadata(report.metadata),
        )
    if not isinstance(report, dict) or "metadata" not in report:
        raise ValueError(
            "report must be a .zip path, TrainingReport, or dict with "
            "data/dataraw/raw and metadata."
        )
    # otherwise, assume a dict with "data" and "metadata" keys
    data = report.get("data", report.get("dataraw", report.get("raw")))
    if data is None:
        raise ValueError("report dict must contain data, dataraw, or raw.")
    return TrainingReport(
        data=_drop_legacy_report_dimension_cols(data),
        metadata=_normalize_metadata(report["metadata"]),
    )


def _xcol_from_metadata(
    metadata: dict,
    x_col: str | None,
) -> str:
    """Compute the x-axis data column for the MSE etc plots."""
    if x_col is not None:
        return x_col

    sweep_col = metadata.get("sweep_col")
    if sweep_col == "nout":
        return "nout"
    elif sweep_col == "N":
        return "N"
    elif sweep_col == "ntr":
        return "ntr"
    else:
        raise ValueError("Could not infer x_col from saved traindata metadata.")


def _summary_quantiles(
    quantiles: Sequence[float] | None,
    quantile_band: tuple[float, float] | None,
) -> tuple[float, ...]:
    """Combine summary quantiles with the plotted quantile band."""
    values = [] if quantiles is None else [float(q) for q in quantiles]
    if quantiles is None:
        values.extend((0.25, 0.75))
    if quantile_band is not None:
        values.extend(float(q) for q in quantile_band)
    return tuple(sorted(set(values)))


def _has_columns(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    """Return whether a DataFrame has all requested columns."""
    return set(cols) <= set(df.columns)


def _summary_metric_cols(
    metric_cols: Sequence[str | MetricExpr] | None,
) -> tuple[str, ...]:
    """Return built-in summary metrics plus any requested ad hoc metrics."""
    if metric_cols is None:
        return tuple(TRAINING_METRIC_COLS)
    return tuple(
        dict.fromkeys(
            (*TRAINING_METRIC_COLS, *(_metric_name(metric) for metric in metric_cols))
        )
    )


def _metric_expr(metric: str | MetricExpr) -> MetricExpr | None:
    """Resolve a metric spec to a MetricExpr if it is computed by this module."""
    if isinstance(metric, MetricExpr):
        return metric
    return DERIVED_TRAINING_METRICS.get(_metric_name(metric))


def _ensure_metric_column(
    raw_df: pd.DataFrame,
    metric: str | MetricExpr,
    *,
    stack: tuple[str, ...] = (),
) -> None:
    """Materialize a requested metric column and any derived dependencies."""
    name = _metric_name(metric)
    if name in raw_df.columns:
        return
    for old_col, new_col in _METRIC_ALIASES.items():
        if new_col == name and old_col in raw_df.columns:
            raw_df[name] = raw_df[old_col]
            return

    expr = _metric_expr(metric)
    if expr is None:
        return
    if name in stack:
        cycle = " -> ".join((*stack, name))
        raise ValueError(f"Cyclic derived metric dependency: {cycle}")

    for dependency in expr.deps:
        _ensure_metric_column(raw_df, dependency, stack=(*stack, name))

    if name not in raw_df.columns and _has_columns(raw_df, expr.deps):
        raw_df[name] = expr.compute(raw_df)


def _safe_denominator(values) -> np.ndarray:
    """Clamp denominators away from zero for derived ratio metrics."""
    return np.maximum(np.asarray(values, dtype=float), _EPS)


def _add_derived_saved_report_metrics(
    raw_df: pd.DataFrame,
    *,
    metric_cols: Sequence[str | MetricExpr] | None = None,
) -> None:
    """Add requested derived metric columns to saved raw data."""
    requested = (
        tuple(DERIVED_TRAINING_METRICS)
        if metric_cols is None
        else tuple(metric_cols)
    )
    for metric in requested:
        _ensure_metric_column(raw_df, metric)


def summarize_traindata(
    report: str | Path | dict | TrainingReport,
    *,
    quantiles: Sequence[float] | None = None,
    quantile_band: tuple[float, float] | None = (0.25, 0.75),
    x_col: str | None = None,
    metric_cols: Sequence[str | MetricExpr] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return expanded raw rows, grouped summaries, and fitted slopes.

    Use this for analysis, table inspection, custom plotting, or tests that
    need the data products. ``plot_saved_traindata()`` is the convenience
    wrapper for plotting only.
    """
    traindata = _as_traindata(report)
    expanded_df = traindata.expanded_df(metric_cols=metric_cols)
    resolved_x_col = _xcol_from_metadata(metadata=traindata.metadata, x_col=x_col)
    summary_df = summarize_dataraw(
        expanded_df,
        quantiles=_summary_quantiles(quantiles, quantile_band),
        metric_cols=metric_cols,
    )
    slopes_df = fit_summary_slopes(
        summary_df,
        x_col=resolved_x_col,
        group_cols=_slope_group_cols(summary_df, resolved_x_col),
    )
    return expanded_df, summary_df, slopes_df


def plot_training_summary(
    summary_df: pd.DataFrame,
    *,
    metadata: dict,
    x_col: str,
    plots: Sequence[str] | str | None,
    quantile_band: tuple[float, float],
    logx: bool = True,
    logy: bool = True,
    figsize: tuple[float, float] = (5.5, 4.0),
    show_mean: bool = True,
    show_median: bool = True,
    show_band: bool = True,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    title: str | None = None,
    legend_outside: bool = False,
    ax=None,
) -> None:
    """Plot selected metrics from an existing training summary table."""
    resolved_plots = _contextualized_training_plots(
        _training_plots_from_keys(plots),
        title_suffix=_training_context_title_suffix(
            metadata,
            quantile_band=quantile_band,
        ),
    )
    resolved_plots = _training_plots_with_title(resolved_plots, title=title)
    resolved_plots = _plot_specs_with_metric_names(resolved_plots)
    plot_grouped_mean_median_quantile_summary(
        summary_df,
        x_col=x_col,
        plots=resolved_plots,
        quantile_band=quantile_band,
        logx=logx,
        logy=logy,
        figsize=figsize,
        show_mean=show_mean,
        show_median=show_median,
        show_band=show_band,
        xlim=xlim,
        ylim=ylim,
        legend_outside=legend_outside,
        ax=ax,
    )


def plot_saved_traindata(
    report: str | Path | dict | TrainingReport,
    *,
    plots: Sequence[str] | str | None = "mse",
    quantile_band: tuple[float, float] = (0.25, 0.75),
    x_col: str | None = None,
    logx: bool = True,
    logy: bool = True,
    figsize: tuple[float, float] = (5.5, 4.0),
    show_mean: bool = True,
    show_median: bool = True,
    show_band: bool = True,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    title: str | None = None,
    legend_outside=False,
    ax=None,
) -> None:
    """
    Load a saved training report and plot the selected summary curves.

    This function is intentionally plot-only. Use ``summarize_traindata()`` if
    you need raw rows, summary statistics, or fitted slopes.

    Example:
        plot_saved_traindata(
            "data/mub_haarstates_N1000_vsntr_extended.zip",
            plots="mse",
            title="Custom MSE title",
            quantile_band=(0.10, 0.90),
        )
    """
    traindata = _as_traindata(report)
    resolved_plots = _training_plots_from_keys(plots)
    metric_specs = _metric_specs_from_plot_specs(resolved_plots)
    expanded_df = traindata.expanded_df(metric_cols=metric_specs)
    summary_df = summarize_dataraw(
        expanded_df,
        quantiles=_summary_quantiles(None, quantile_band),
        metric_cols=metric_specs,
    )
    resolved_x_col = _xcol_from_metadata(metadata=traindata.metadata, x_col=x_col)

    plot_training_summary(
        summary_df,
        metadata=traindata.metadata,
        x_col=resolved_x_col,
        plots=plots,
        quantile_band=quantile_band,
        logx=logx,
        logy=logy,
        figsize=figsize,
        show_mean=show_mean,
        show_median=show_median,
        show_band=show_band,
        xlim=xlim,
        ylim=ylim,
        title=title,
        legend_outside=legend_outside,
        ax=ax,
    )
