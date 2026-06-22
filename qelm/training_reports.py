"""Tilde-U training report serialization, summaries, and plotting."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from typing import Sequence, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .linalg import distribution_summary, loglog_fit, quantile_suffix
from .plotting import plot_grouped_mean_median_quantile_summary
from .quantum import QuantumStateBatch, generate_qubit_mub_povm
from .training import (
    TildeUTrainingApproxStudySpec,
    _povm_kind_from_spec,
    _required_noise_N,
    _resolve_test_state_request,
    _training_state_count_from_spec,
    with_training_sweep_value,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - plain Python fallback
    def display(obj):
        print(obj)


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

_SAVED_METRIC_COLS = {
    "leading_bias_sq_exact": "leading_bias_sq",
    "leading_var_exact": "leading_variance",
    "leading_bias_sq_identity": "identity_leading_bias_sq",
    "leading_var_identity": "identity_leading_variance",
    "actual_mse": "mse",
    "actual_bias_sq": "bias_sq",
    "actual_variance": "variance",
}
_COMPACT_TO_INTERNAL_METRIC_COLS = {
    compact: internal for internal, compact in _SAVED_METRIC_COLS.items()
}
_ADDITIVE_METRIC_COLS = {
    "leading_mse_exact": ("leading_bias_sq_exact", "leading_var_exact"),
    "leading_mse_identity": ("leading_bias_sq_identity", "leading_var_identity"),
}
_RATIO_METRIC_COLS = {
    "bias_sq_identity_over_exact": (
        "leading_bias_sq_identity",
        "leading_bias_sq_exact",
    ),
    "var_identity_over_exact": ("leading_var_identity", "leading_var_exact"),
    "mse_identity_over_exact": ("leading_mse_identity", "leading_mse_exact"),
    "actual_over_leading_exact": ("actual_mse", "leading_mse_exact"),
    "actual_over_leading_identity": ("actual_mse", "leading_mse_identity"),
}
_RELATIVE_ERROR_METRIC_COLS = {
    "leading_exact_relative_error": ("leading_mse_exact", "actual_mse"),
    "leading_identity_relative_error": ("leading_mse_identity", "actual_mse"),
}
_EPS = 1e-15
_LEGACY_REPORT_DIMENSION_COLS = ("r", "q", "p_kernel")


def _drop_legacy_report_dimension_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(_LEGACY_REPORT_DIMENSION_COLS), errors="ignore")


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


def _tilde_u_training_approx_context_metadata(study: TildeUTrainingApproxStudySpec) -> dict:
    base = study.base
    return {
        "sweep_col": study.sweep_col,
        "data": {
            "d": int(base.data.d),
            "nout": None if base.data.nout is None else int(base.data.nout),
            "povm": _povm_descriptor(base.data.povm),
            "train_state_count": _training_state_count_from_spec(base.data.train_states),
        },
        "target": {
            "observable": _target_descriptor(
                base.target.observable,
                dim=int(base.data.d),
                nout=base.data.nout,
            ),
            "normalization": base.target.normalization,
        },
        "test": {"state": _test_descriptor(base.test.state, dim=int(base.data.d))},
        "noise": {
            "noise": base.noise.noise,
            "N": None if base.noise.N is None else int(base.noise.N),
        },
    }


def _tilde_u_context_povm_label(data: dict, sweep_col: str | None) -> str:
    povm = data.get("povm", {})
    kind = str(povm.get("kind", "unknown"))
    label = _tilde_u_povm_label_from_descriptor(povm)
    parts = [f"POVM={label}"]

    if kind == "random_rank1":
        nout = "sweep" if sweep_col == "nout" else data.get("nout")
        if nout is not None:
            parts.append(f"nout={nout}")

    return ", ".join(parts)


def _array_from_payload(payload: dict) -> np.ndarray:
    if payload.get("kind") != "ndarray":
        raise ValueError("Array payload kind must be 'ndarray'.")
    if "real" in payload and "imag" in payload:
        array = np.asarray(payload["real"]) + 1j * np.asarray(payload["imag"])
    else:
        array = np.asarray(payload["data"])
    return array.astype(payload.get("dtype", array.dtype), copy=False)


def _tilde_u_povm_label_from_descriptor(povm: dict) -> str:
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
            if np.allclose(effects, generate_qubit_mub_povm(), atol=1e-12):
                return "qubit_mub"

    return kind


def _tilde_u_context_average_label(kind: object) -> str:
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


def _tilde_u_context_quantile_label(
    quantile_band: tuple[float, float] | None,
) -> str | None:
    if quantile_band is None:
        return None
    qlo, qhi = quantile_band
    return f"{quantile_suffix(qlo)}-{quantile_suffix(qhi)}"


def _tilde_u_training_context_title_suffix(
    metadata: dict,
    *,
    quantile_band: tuple[float, float] | None,
) -> str:
    sweep_col = metadata.get("sweep_col")
    data = metadata.get("data", {})
    noise = metadata.get("noise", {})
    test = metadata.get("test", {}).get("state", {})
    target = metadata.get("target", {}).get("observable", {})

    first_line = [_tilde_u_context_povm_label(data, sweep_col)]

    ntr = data.get("train_state_count")
    if sweep_col != "ntr" and ntr is not None:
        first_line.append(f"ntr={ntr}")

    N = noise.get("N")
    if sweep_col != "N" and N is not None:
        first_line.append(f"N={N}")

    second_line = [
        f"test={_tilde_u_context_average_label(test.get('kind'))}",
        f"target={_tilde_u_context_average_label(target.get('kind'))}",
    ]

    quantiles = _tilde_u_context_quantile_label(quantile_band)
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


def _tilde_u_training_contextualized_plots(
    plots: Sequence[tuple],
    *,
    title_suffix: str | None,
) -> list[tuple]:
    if not title_suffix:
        return list(plots)

    mse_title = TILDE_U_TRAINING_APPROX_PLOT_SPECS["mse"][1]
    contextualized = []
    for series_specs, title, ylabel in plots:
        if title == mse_title:
            title = f"{title}\n{title_suffix}"
        contextualized.append((series_specs, title, ylabel))
    return contextualized


def summarize_tilde_u_training_approx(
    raw_df: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.25, 0.75),
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
        return "nout"
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
    return _drop_legacy_report_dimension_cols(pd.read_parquet(BytesIO(data)))


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
        descriptor = {"kind": "random_rank1", "label": "random_rank1"}
        if isinstance(povm, dict):
            if isinstance(povm.get("label"), str) and povm["label"].strip():
                descriptor["label"] = povm["label"].strip()
            for key in ("nout", "dim"):
                if key in povm:
                    descriptor[key] = int(povm[key])
        return descriptor
    if kind == "qubit_mub":
        return {"kind": "qubit_mub", "label": "qubit_mub", "nout": 6, "dim": 2}

    if hasattr(povm, "effects"):
        effects = np.asarray(povm.effects)
        label = getattr(povm, "label", None)
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Explicit POVM metadata requires a non-empty label.")
        return {
            "kind": "explicit",
            "label": label.strip(),
            "effects": _array_payload(effects),
        }

    label = povm.get("label") if isinstance(povm, dict) else None
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Explicit POVM metadata requires a non-empty label.")
    effects = povm.get("effects") if isinstance(povm, dict) else povm
    return {"kind": "explicit", "label": label.strip(), "effects": _array_payload(effects)}


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
    cols = ["trial"]
    if sweep_col in raw_df.columns:
        cols.append(sweep_col)
    if "failed" in raw_df.columns and raw_df["failed"].fillna(False).any():
        cols.append("failed")
    cols.extend(col for col in _SAVED_METRIC_COLS if col in raw_df.columns)
    return raw_df.loc[:, cols].rename(columns=_SAVED_METRIC_COLS)


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


def _normalize_tilde_u_report_metadata(metadata: dict) -> dict:
    normalized = json.loads(json.dumps(metadata))
    povm = normalized.get("data", {}).get("povm")
    if isinstance(povm, dict):
        label = povm.get("label")
        if not isinstance(label, str) or not label.strip():
            inferred = _tilde_u_povm_label_from_descriptor(povm)
            if inferred not in {"unknown", "explicit"}:
                povm["label"] = inferred
    return normalized


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
                "metadata": _normalize_tilde_u_report_metadata(
                    json.loads(archive.read(manifest["metadata"]).decode("utf-8"))
                ),
            }
    raise ValueError("tilde-U report path must end in .zip.")


def _as_tilde_u_training_approx_report_data(report: str | Path | dict) -> dict:
    if isinstance(report, (str, Path)):
        return load_tilde_u_training_approx_report_data(report)
    if not isinstance(report, dict) or "raw" not in report or "metadata" not in report:
        raise ValueError(
            "report must be a .zip path or a dict returned by "
            "load_tilde_u_training_approx_report_data."
        )
    normalized = dict(report)
    normalized["metadata"] = _normalize_tilde_u_report_metadata(report["metadata"])
    return normalized


def _tilde_u_report_rank_from_metadata(metadata: dict) -> int | None:
    numerics = metadata.get("numerics", {})
    rank = numerics.get("rank")
    if rank is not None:
        return int(rank)

    data = metadata.get("data", {})
    d = data.get("d")
    if d is None:
        return None
    return int(d) ** 2


def _tilde_u_report_x_col(
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
        raise ValueError("Could not infer x_col from saved report data.")


def _tilde_u_report_summary_quantiles(
    quantiles: Sequence[float] | None,
    quantile_band: tuple[float, float] | None,
) -> tuple[float, ...]:
    values = [] if quantiles is None else [float(q) for q in quantiles]
    if quantiles is None:
        values.extend((0.25, 0.75))
    if quantile_band is not None:
        values.extend(float(q) for q in quantile_band)
    return tuple(sorted(set(values)))


def _tilde_u_saved_report_plot_raw_df(report_data: dict) -> pd.DataFrame:
    raw_df = _drop_legacy_report_dimension_cols(report_data["raw"]).copy()
    metadata = report_data.get("metadata", {})

    for compact_col, internal_col in _COMPACT_TO_INTERNAL_METRIC_COLS.items():
        if compact_col in raw_df.columns and internal_col not in raw_df.columns:
            raw_df[internal_col] = raw_df[compact_col]

    data = metadata.get("data", {})
    noise = metadata.get("noise", {})
    target = metadata.get("target", {})
    test = metadata.get("test", {})
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
    _add_derived_saved_report_metrics(raw_df)

    return raw_df


def _has_columns(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return set(cols) <= set(df.columns)


def _safe_denominator(values) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), _EPS)


def _add_derived_saved_report_metrics(raw_df: pd.DataFrame) -> None:
    for col, parts in _ADDITIVE_METRIC_COLS.items():
        if col not in raw_df.columns and _has_columns(raw_df, parts):
            raw_df[col] = sum(raw_df[part] for part in parts)

    for col, (numerator, denominator) in _RATIO_METRIC_COLS.items():
        if col not in raw_df.columns and _has_columns(raw_df, (numerator, denominator)):
            raw_df[col] = (
                raw_df[numerator].to_numpy(dtype=float)
                / _safe_denominator(raw_df[denominator])
            )

    for col, (prediction, actual) in _RELATIVE_ERROR_METRIC_COLS.items():
        if col not in raw_df.columns and _has_columns(raw_df, (prediction, actual)):
            raw_df[col] = (
                np.abs(
                    raw_df[prediction].to_numpy(dtype=float)
                    - raw_df[actual].to_numpy(dtype=float)
                )
                / _safe_denominator(raw_df[actual])
            )


def summarize_saved_training_data(
    report: str | Path | dict,
    *,
    quantiles: Sequence[float] | None = None,
    quantile_band: tuple[float, float] | None = (0.25, 0.75),
    x_col: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build plotting-ready raw, summary, and slope tables from a saved report.

    `report` can be a .zip path or the dictionary returned by
    `load_tilde_u_training_approx_report_data`. Compact saved reports store
    user-facing metric names; this helper restores the internal metric columns
    expected by the notebook plotting functions and derives MSE ratios/errors.
    """
    report_data = _as_tilde_u_training_approx_report_data(report)
    plot_raw_df = _tilde_u_saved_report_plot_raw_df(report_data)
    resolved_x_col = _tilde_u_report_x_col(metadata=report_data["metadata"], x_col=x_col)
    summary_df = summarize_tilde_u_training_approx(
        plot_raw_df,
        quantiles=_tilde_u_report_summary_quantiles(quantiles, quantile_band),
    )
    slopes_df = fit_tilde_u_training_approx_slopes(
        summary_df,
        x_col=resolved_x_col,
        group_cols=_tilde_u_slope_group_cols(summary_df, resolved_x_col),
    )
    return plot_raw_df, summary_df, slopes_df


def render_tilde_u_training_approx_report(
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
    show_band: bool = True,
) -> None:
    """Display optional tables and render the configured tilde-U summary plots."""
    if show_summary:
        display(summary_df)
    if show_slopes:
        display(slopes_df)
    if not make_plots:
        return

    resolved_plots = _tilde_u_training_contextualized_plots(
        _tilde_u_training_approx_plots_from_keys(plots),
        title_suffix=_tilde_u_training_context_title_suffix(
            metadata,
            quantile_band=quantile_band,
        ),
    )
    plot_grouped_mean_median_quantile_summary(
        summary_df,
        x_col=x_col,
        plots=resolved_plots,
        quantile_band=quantile_band,
        logx=logx,
        logy=logy,
        figsize=figsize,
        show_mean=show_mean,
        show_band=show_band,
    )


def plot_saved_training_data(
    report: str | Path | dict,
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
    show_band: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load or reuse a saved tilde-U report and plot it like the live report.

    Example:
        plot_saved_training_data(
            "data/mub_haarstates_N1000_vsntr_extended.zip",
            plots="mse",
            quantile_band=(0.10, 0.90),
        )
    """
    report_data = _as_tilde_u_training_approx_report_data(report)
    raw_df, summary_df, slopes_df = summarize_saved_training_data(
        report_data,
        quantiles=quantiles,
        quantile_band=quantile_band,
        x_col=x_col,
    )
    resolved_x_col = _tilde_u_report_x_col(metadata=report_data["metadata"], x_col=x_col)

    render_tilde_u_training_approx_report(
        summary_df,
        slopes_df,
        metadata=report_data["metadata"],
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
        show_band=show_band,
    )

    return raw_df, summary_df, slopes_df


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
