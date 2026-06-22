from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd


CONFIG_COLS = ["d", "N", "nout", "ntr"]
LEGACY_DIMENSION_COLS = ["r", "q", "p_kernel"]


def unique_sorted(values) -> list:
    vals = pd.Series(values).dropna().unique().tolist()
    try:
        return sorted(vals, key=lambda x: float(x))
    except Exception:
        return sorted(vals, key=str)


def fmt_values(values, max_items: int = 14) -> str:
    vals = unique_sorted(values)
    if not vals:
        return "-"
    vals = [int(v) if isinstance(v, float) and v.is_integer() else v for v in vals]
    if len(vals) <= max_items:
        return ", ".join(map(str, vals))
    head = ", ".join(map(str, vals[:7]))
    tail = ", ".join(map(str, vals[-4:]))
    return f"{head}, ..., {tail} ({len(vals)} values)"


def array_shape(payload) -> list | None:
    if isinstance(payload, dict) and payload.get("kind") == "ndarray":
        return payload.get("shape")
    return None


def metadata_fixed_value(meta: dict, col: str):
    data = meta.get("data", {})
    noise = meta.get("noise", {})

    if col == "d":
        return data.get("d")

    if col == "N":
        value = noise.get("N")
        return value if isinstance(value, (int, float)) else None

    if col == "nout":
        if data.get("nout") is not None:
            return data.get("nout")
        povm = data.get("povm", {})
        if isinstance(povm, dict):
            if povm.get("nout") is not None:
                return povm.get("nout")
            shape = array_shape(povm.get("effects"))
            if shape:
                return shape[0]
        return None

    if col == "ntr":
        if data.get("train_state_count") is not None:
            return data.get("train_state_count")
        states = data.get("train_states", {})
        if isinstance(states, dict):
            n = states.get("num_states")
            if isinstance(n, (int, float)):
                return n
            shape = array_shape(states.get("states"))
            if shape:
                return shape[0]
        return None

    return None


def materialize_config_columns(raw: pd.DataFrame, meta: dict) -> pd.DataFrame:
    cfg = pd.DataFrame(index=raw.index)
    for col in CONFIG_COLS:
        cfg[col] = raw[col] if col in raw.columns else metadata_fixed_value(meta, col)
    return cfg


def label_from_meta(meta: dict, *keys: str, default: str = "-") -> str:
    value = meta
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    if isinstance(value, dict):
        return value.get("label") or value.get("kind") or default
    return default if value is None else str(value)


def read_report_zip(path: Path | str) -> tuple[pd.DataFrame, dict]:
    with ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        raw_name = manifest.get("tables", {}).get("raw", "raw.parquet")
        meta_name = manifest.get("metadata", "metadata.json")
        raw = pd.read_parquet(io.BytesIO(archive.read(raw_name)))
        meta = json.loads(archive.read(meta_name).decode("utf-8"))
    return raw.drop(columns=LEGACY_DIMENSION_COLS, errors="ignore"), meta


def summarize_data_folder(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    config_rows = []
    error_rows = []

    for path in sorted(Path(data_dir).rglob("*.zip")):
        rel = str(path)
        try:
            raw, meta = read_report_zip(path)
        except (BadZipFile, KeyError, json.JSONDecodeError, OSError, ValueError) as exc:
            error_rows.append({"file": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue

        cfg = materialize_config_columns(raw, meta)
        if "failed" in raw:
            failed = raw["failed"].fillna(False).astype(bool)
        else:
            failed = pd.Series(False, index=raw.index)

        per_config = (
            cfg.assign(rows=1, failures=failed)
            .groupby(CONFIG_COLS, dropna=False, as_index=False)
            .agg(rows=("rows", "sum"), failures=("failures", "sum"))
            .sort_values(CONFIG_COLS, kind="stable")
        )

        povm = label_from_meta(meta, "data", "povm")
        train_states = label_from_meta(meta, "data", "train_states")
        noise = label_from_meta(meta, "noise", "noise")
        target = label_from_meta(meta, "target", "observable")
        test = label_from_meta(meta, "test", "state")

        for _, row in per_config.iterrows():
            config_rows.append(
                {
                    "file": rel,
                    "povm": povm,
                    "train_states": train_states,
                    "noise": noise,
                    "target": target,
                    "test": test,
                    **{col: row[col] for col in CONFIG_COLS},
                    "rows": int(row["rows"]),
                    "failures": int(row["failures"]),
                }
            )

        repetitions = (
            str(int(per_config["rows"].iloc[0]))
            if per_config["rows"].nunique() == 1
            else f"{int(per_config['rows'].min())}-{int(per_config['rows'].max())}"
        )
        sweep_col = meta.get("sweep_col") or ",".join(meta.get("grid", {}).get("sweep_cols", []))
        summary_rows.append(
            {
                "file": rel,
                "rows": len(raw),
                "configs": len(per_config),
                "reps/config": repetitions,
                "failures": int(per_config["failures"].sum()),
                "d": fmt_values(per_config["d"]),
                "N": fmt_values(per_config["N"]),
                "nout": fmt_values(per_config["nout"]),
                "ntr": fmt_values(per_config["ntr"]),
                "povm": povm,
                "train_states": train_states,
                "target": target,
                "test": test,
                "sweep_col": sweep_col,
                "created_at": meta.get("created_at"),
                "completed_at": meta.get("completed_at"),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(config_rows), pd.DataFrame(error_rows)


def _tested_values_from_configs(
    configs: pd.DataFrame,
    *,
    include_files: bool = False,
    include_target_test: bool = False,
) -> pd.DataFrame:
    if configs.empty:
        return pd.DataFrame()

    group_cols = ["povm", "train_states", "noise"]
    if include_target_test:
        group_cols.extend(["target", "test"])
    group_cols.extend(["d", "nout"])

    aggregations = {
        "N": ("N", fmt_values),
        "ntr": ("ntr", fmt_values),
        "configs": ("rows", "size"),
        "raw_rows": ("rows", "sum"),
        "failures": ("failures", "sum"),
    }
    if include_files:
        aggregations["files"] = ("file", lambda x: ", ".join(sorted(set(map(str, x)))))

    table = (
        configs.groupby(group_cols, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["povm", "train_states", "d", "nout"], kind="stable")
        .reset_index(drop=True)
    )

    return table


def tested_values_table(
    data_dir: str | Path = "data",
    *,
    include_files: bool = False,
    include_target_test: bool = False,
    include_errors: bool = False,
) -> pd.DataFrame:
    """
    Return a compact notebook-friendly table of tested N, ntr, and nout values.

    The table is grouped by experiment case, using the raw Parquet rows as the
    source of truth for completed configurations. Unreadable zip files are
    skipped; their details are stored in ``table.attrs["errors"]``.
    """
    _, configs, errors = summarize_data_folder(data_dir)
    table = _tested_values_from_configs(
        configs,
        include_files=include_files,
        include_target_test=include_target_test,
    )
    table.attrs["errors"] = errors
    if include_errors and not errors.empty:
        table.attrs["error_message"] = errors.to_string(index=False)
    return table


def file_summary_table(data_dir: str | Path = "data") -> pd.DataFrame:
    """Return one summary row per readable zip report."""
    summary, _, errors = summarize_data_folder(data_dir)
    summary.attrs["errors"] = errors
    return summary


def print_summary(summary: pd.DataFrame, configs: pd.DataFrame, errors: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 240)

    print("\n=== Files ===")
    print(summary.to_string(index=False) if not summary.empty else "No readable zip reports found.")

    if not configs.empty:
        grouped = _tested_values_from_configs(configs, include_files=True)
        print("\n=== Tested values grouped by case ===")
        print(grouped.to_string(index=False))

    if not errors.empty:
        print("\n=== Files skipped because they could not be read ===")
        print(errors.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize QELM zip reports in a data folder.")
    parser.add_argument("data_dir", nargs="?", default="data")
    args = parser.parse_args()
    print_summary(*summarize_data_folder(args.data_dir))


if __name__ == "__main__":
    main()
