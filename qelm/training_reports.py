"""Training-result loading, summaries, and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Callable, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .linalg import distribution_summary, loglog_fit, quantile_suffix
from .plotting import plot_grouped_mean_median_quantile_summary
from .quantum import qubit_mub_povm

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - plain Python fallback
    def display(obj):
        """Print objects when IPython display is unavailable."""
        print(obj)


@dataclass(frozen=True)
class MetricExpr:
    """A lazily materialized training metric column."""

    name: str
    deps: tuple[str, ...]
    compute: Callable[[pd.DataFrame], object]


# Ordered public metric schema used by summarize_dataraw(), fit_summary_slopes(),
# and the saved-report plotting helpers. Each string is a DataFrame column name:
# raw columns may come directly from a simulation report, while derived columns
# such as leading_mse_exact or mse_identity_over_exact are reconstructed from the
# MetricExpr registry below when TrainingReport.expanded_df() loads compact saved
# data. The order controls display/table ordering; it is not a dependency order.
TRAINING_METRIC_COLS = [
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
            ("actual_mse", r"$\mathrm{MSE}(N,\infty)$"),
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

# Internal-to-saved metric-name mapping for report serialization. Keys are the
# canonical names used everywhere in this module; values are the shorter column
# names found in saved JSON/ZIP reports. TrainingReport.expanded_df() applies the
# reverse map (_COMPACT_TO_INTERNAL_METRIC_COLS) so downstream code only sees the
# canonical names, regardless of how the report was stored on disk.
TRAINING_SAVED_METRIC_COLS = {
    "leading_bias_sq_exact": "leading_bias_sq",
    "leading_var_exact": "leading_variance",
    "leading_bias_sq_identity": "identity_leading_bias_sq",
    "leading_var_identity": "identity_leading_variance",
    "actual_mse": "mse",
    "actual_bias_sq": "bias_sq",
    "actual_variance": "variance",
}

# Historical aliases for the earlier tilde-U approximation API. They point to the
# same list/dict objects rather than copies, so adding a training metric or plot
# keeps old notebooks and scripts in sync with the generalized training helpers.
TILDE_U_TRAINING_APPROX_METRIC_COLS = TRAINING_METRIC_COLS
TILDE_U_TRAINING_APPROX_PLOT_SPECS = TRAINING_PLOT_SPECS
TILDE_U_TRAINING_APPROX_PLOTS = TRAINING_PLOTS
TILDE_U_TRAINING_APPROX_SAVED_METRIC_COLS = TRAINING_SAVED_METRIC_COLS

# Private aliases for loader internals. _SAVED_METRIC_COLS names the map used by
# the report loader, while _COMPACT_TO_INTERNAL_METRIC_COLS reverses it from
# saved-column -> canonical-column for fast column renaming during expansion.
_SAVED_METRIC_COLS = TRAINING_SAVED_METRIC_COLS
_COMPACT_TO_INTERNAL_METRIC_COLS = {
    compact: internal for internal, compact in _SAVED_METRIC_COLS.items()
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
    "leading_mse_exact": MetricExpr(
        "leading_mse_exact",
        ("leading_bias_sq_exact", "leading_var_exact"),
        lambda df: df["leading_bias_sq_exact"] + df["leading_var_exact"],
    ),
    "leading_mse_identity": MetricExpr(
        "leading_mse_identity",
        ("leading_bias_sq_identity", "leading_var_identity"),
        lambda df: df["leading_bias_sq_identity"] + df["leading_var_identity"],
    ),
    "bias_sq_identity_over_exact": MetricExpr(
        "bias_sq_identity_over_exact",
        ("leading_bias_sq_identity", "leading_bias_sq_exact"),
        lambda df: (
            df["leading_bias_sq_identity"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_bias_sq_exact"])
        ),
    ),
    "var_identity_over_exact": MetricExpr(
        "var_identity_over_exact",
        ("leading_var_identity", "leading_var_exact"),
        lambda df: (
            df["leading_var_identity"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_var_exact"])
        ),
    ),
    "mse_identity_over_exact": MetricExpr(
        "mse_identity_over_exact",
        ("leading_mse_identity", "leading_mse_exact"),
        lambda df: (
            df["leading_mse_identity"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_mse_exact"])
        ),
    ),
    "actual_over_leading_exact": MetricExpr(
        "actual_over_leading_exact",
        ("actual_mse", "leading_mse_exact"),
        lambda df: (
            df["actual_mse"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_mse_exact"])
        ),
    ),
    "actual_over_leading_identity": MetricExpr(
        "actual_over_leading_identity",
        ("actual_mse", "leading_mse_identity"),
        lambda df: (
            df["actual_mse"].to_numpy(dtype=float)
            / _safe_denominator(df["leading_mse_identity"])
        ),
    ),
    "leading_exact_relative_error": MetricExpr(
        "leading_exact_relative_error",
        ("leading_mse_exact", "actual_mse"),
        lambda df: (
            np.abs(
                df["leading_mse_exact"].to_numpy(dtype=float)
                - df["actual_mse"].to_numpy(dtype=float)
            )
            / _safe_denominator(df["actual_mse"])
        ),
    ),
    "leading_identity_relative_error": MetricExpr(
        "leading_identity_relative_error",
        ("leading_mse_identity", "actual_mse"),
        lambda df: (
            np.abs(
                df["leading_mse_identity"].to_numpy(dtype=float)
                - df["actual_mse"].to_numpy(dtype=float)
            )
            / _safe_denominator(df["actual_mse"])
        ),
    ),
    "leading_mse_identity_minus_exact_times_N": MetricExpr(
        "leading_mse_identity_minus_exact_times_N",
        ("leading_mse_identity", "leading_mse_exact", "N"),
        lambda df: (
            df["N"].to_numpy(dtype=float)
            * (
                df["leading_mse_identity"].to_numpy(dtype=float)
                - df["leading_mse_exact"].to_numpy(dtype=float)
            )
        ),
    ),
    "leading_mse_identity_minus_exact_times_N2": MetricExpr(
        "leading_mse_identity_minus_exact_times_N2",
        ("leading_mse_identity", "leading_mse_exact", "N"),
        lambda df: (
            df["N"].to_numpy(dtype=float) ** 2
            * (
                df["leading_mse_identity"].to_numpy(dtype=float)
                - df["leading_mse_exact"].to_numpy(dtype=float)
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
        """Return raw data expanded with metadata constants and derived metrics."""
        raw_df = _drop_legacy_report_dimension_cols(self.data).copy()

        for compact_col, internal_col in _COMPACT_TO_INTERNAL_METRIC_COLS.items():
            if compact_col in raw_df.columns and internal_col not in raw_df.columns:
                raw_df[internal_col] = raw_df[compact_col]

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
        """Build plotting-ready raw, summary, and slope tables."""
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Render plots from this report and return raw, summary, and slopes."""
        return plot_saved_traindata(self, **kwargs)


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
    return metric.name if isinstance(metric, MetricExpr) else metric


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
    """
    Summarize training-result trials with configurable quantiles.
    """
    rows = []
    ok = raw_df[~raw_df.get("failed", False).astype(bool)].copy() if "failed" in raw_df else raw_df
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
        "leading_mse_exact_median",
        "leading_mse_identity_median",
        "actual_mse_median",
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
    Load a portable training report as a TrainingReport.

    Portable .zip reports store tables as Parquet and metadata as plain JSON,
    so they are intended for moving between machines and pandas versions.
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
    return DERIVED_TRAINING_METRICS.get(metric)


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
    """Add requested derived metric columns to saved raw data.
    
    Parameters
    ----------
    raw_df : pd.DataFrame
        The raw training data DataFrame to which derived metrics will be added.
    metric_cols : Sequence[str | MetricExpr] | None, optional
        A sequence of metric names or MetricExpr objects specifying which derived metrics to add.
        If None, all built-in derived metrics are added. Otherwise, only the requested metrics are
    """
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
    Build plotting-ready raw, summary, and slope tables from a saved report.

    `report` can be a .zip path or the dictionary returned by
    `load_traindata`. Compact saved reports store
    user-facing metric names; this helper restores the internal metric columns
    expected by the notebook plotting functions and derives MSE ratios/errors.
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


def render_training_results(
    summary_df: pd.DataFrame,
    slopes_df: pd.DataFrame,
    *,
    metadata: dict,
    x_col: str,
    plots: Sequence[str] | str | None,
    quantile_band: tuple[float, float],
    show_summary: bool = False,
    show_slopes: bool = False,
    make_plots: bool = True,
    logx: bool = True,
    logy: bool = True,
    figsize: tuple[float, float] = (5.5, 4.0),
    show_mean: bool = True,
    show_median: bool = True,
    show_band: bool = True,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    legend_outside: bool = False,
    ax=None,
) -> None:
    """Display optional tables and render the configured training summary plots."""
    if show_summary:
        display(summary_df)
    if show_slopes:
        display(slopes_df)
    if not make_plots:
        return

    resolved_plots = _contextualized_training_plots(
        _training_plots_from_keys(plots),
        title_suffix=_training_context_title_suffix(
            metadata,
            quantile_band=quantile_band,
        ),
    )
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
    quantiles: Sequence[float] | None = None,
    quantile_band: tuple[float, float] = (0.25, 0.75),
    x_col: str | None = None,
    show_summary: bool = False,
    show_slopes: bool = False,
    make_plots: bool = True,
    logx: bool = True,
    logy: bool = True,
    figsize: tuple[float, float] = (5.5, 4.0),
    show_mean: bool = True,
    show_median: bool = True,
    show_band: bool = True,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    legend_outside=False,
    ax=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load or reuse saved training data and plot it like live results.

    Example:
        plot_saved_traindata(
            "data/mub_haarstates_N1000_vsntr_extended.zip",
            plots="mse",
            quantile_band=(0.10, 0.90),
        )
    """
    traindata = _as_traindata(report)
    resolved_plots = _training_plots_from_keys(plots)
    raw_df, summary_df, slopes_df = summarize_traindata(
        traindata,
        quantiles=quantiles,
        quantile_band=quantile_band,
        x_col=x_col,
        metric_cols=_metric_specs_from_plot_specs(resolved_plots),
    )
    resolved_x_col = _xcol_from_metadata(metadata=traindata.metadata, x_col=x_col)

    render_training_results(
        summary_df,
        slopes_df,
        metadata=traindata.metadata,
        x_col=resolved_x_col,
        plots=plots,
        quantile_band=quantile_band,
        show_summary=show_summary,
        show_slopes=show_slopes,
        make_plots=make_plots,
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

    return raw_df, summary_df, slopes_df


summarize_tilde_u_training_approx = summarize_dataraw
fit_tilde_u_training_approx_slopes = fit_summary_slopes
load_tilde_u_training_approx_report_data = load_traindata
summarize_saved_training_data = summarize_traindata
render_tilde_u_training_approx_report = render_training_results
plot_saved_training_data = plot_saved_traindata
