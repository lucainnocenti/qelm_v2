# scripts/run_fixedrndpovm_haarstates_avgtesttarget_grid.py

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qelm import (  # noqa: E402
    POVM,
    QELMDataSpec,
    QELMNoiseSpec,
    QELMRun,
    QELMTargetRequest,
    QELMTestRequest,
    QELMTrainingSpec,
    TrainingStudySpec,
    generate_random_rank1_povm,
)
from qelm.training import (  # noqa: E402
    _compute_qelm_leading_error_resolved,
    _run_qelm_training_resolved,
)
from qelm.workflows import (  # noqa: E402
    _save_traindata,
    _saved_dataraw,
    _training_metadata,
    _training_trial_row,
)


DIMS = (2,)
NOUT_VALUES = (16,)
NTR_VALUES = tuple(2**power for power in range(2, 14))
N_VALUES = tuple(2**power for power in range(0, 12))

REPETITIONS = 1000
SEED = 20260620
MAX_PROCESSES = 5
FAIL_SOFT = False
OVERWRITE = False

OUTPUT_DIR = PROJECT_ROOT / "data" / "fixedrndpovm_haarstates_avgtesttarget_grid"


def _nout_values(d: int) -> tuple[int, ...]:
    return tuple(value for value in NOUT_VALUES if value > int(d) ** 2)


def _ntr_values(nout: int) -> tuple[int, ...]:
    return tuple(value for value in NTR_VALUES if value >= int(nout))


def _fixed_povms(rng: np.random.Generator) -> dict[tuple[int, int], POVM]:
    return {
        (d, nout): POVM.from_effects(
            generate_random_rank1_povm(nout=nout, dim=d, rng=rng),
            dim=d,
            nout=nout,
            label=f"fixed_random_rank1_d{d}_nout{nout}",
        )
        for d in DIMS
        for nout in _nout_values(d)
    }


def _work_items() -> list[dict]:
    rng = np.random.default_rng(SEED)
    povms = _fixed_povms(rng)
    return [
        {
            "d": int(d),
            "nout": int(nout),
            "ntr": int(ntr),
            "N_values": tuple(int(N) for N in N_VALUES),
            "povm": povms[(d, nout)],
            "seed": int(rng.integers(0, 2**32 - 1)),
            "output_file": str(OUTPUT_DIR / f"_d{d}_nout{nout}_ntr{int(ntr)}_vsN.zip"),
        }
        for d in DIMS
        for nout in _nout_values(d)
        for ntr in _ntr_values(nout)
    ]


def _item_cost(item: dict) -> int:
    return int(item["nout"]) * int(item["ntr"]) * len(item["N_values"])


def _split_work_items(items: list[dict], parts: int) -> list[list[dict]]:
    chunks = [[] for _ in range(parts)]
    loads = [0 for _ in range(parts)]
    for item in sorted(items, key=_item_cost, reverse=True):
        index = min(range(parts), key=lambda idx: loads[idx])
        chunks[index].append(item)
        loads[index] += _item_cost(item)
    for chunk in chunks:
        chunk.sort(key=lambda item: (item["d"], item["nout"], item["ntr"]))
    return chunks


def _base_spec_from_item(item: dict, *, N: int | None) -> QELMTrainingSpec:
    return QELMTrainingSpec(
        data=QELMDataSpec(
            d=int(item["d"]),
            nout=int(item["nout"]),
            povm=item["povm"],
            train_states={
                "kind": "haar_pure",
                "num_states": int(item["ntr"]),
            },
        ),
        target=QELMTargetRequest(observable="haar_pure_average"),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(
            noise="multinomial",
            N=None if N is None else int(N),
            actual_noise_trials=1,
        ),
    )


def _study_from_item(item: dict) -> TrainingStudySpec:
    return TrainingStudySpec(
        base=_base_spec_from_item(item, N=None),
        sweep_col="N",
        sweep_values=tuple(int(value) for value in item["N_values"]),
        repetitions=REPETITIONS,
        seed=int(item["seed"]),
        quantiles=(0.10, 0.25, 0.75, 0.90),
        quantile_band=(0.1, 0.9),
        show_summary=False,
        show_slopes=False,
        make_plots=False,
        verbose=True,
        fail_soft=FAIL_SOFT,
        output_file=Path(item["output_file"]),
        overwrite=OVERWRITE,
    )


def _item_label(item: dict) -> str:
    N_values = tuple(int(value) for value in item["N_values"])
    N_label = (
        str(N_values[0])
        if len(N_values) == 1
        else f"{N_values[0]}..{N_values[-1]} ({len(N_values)})"
    )
    return f"d={item['d']} nout={item['nout']} ntr={item['ntr']} N={N_label}"


def _row_for_N(
    *,
    item: dict,
    rng: np.random.Generator,
    run: QELMRun,
    N: int,
    leading_corrected_at_N1: dict,
    leading_identity_at_N1: dict,
) -> dict:
    spec = replace(
        run.spec,
        noise=replace(run.spec.noise, N=int(N)),
    )
    actual = _run_qelm_training_resolved(
        spec,
        rng=rng,
        context=run.context,
        target=run.target,
        test=run.test,
    )

    return _training_trial_row(
        d=int(item["d"]),
        ntr=int(run.context.P_train.shape[1]),
        N=int(N),
        noise=spec.noise.noise,
        blocks=run.diagnostics.blocks,
        diag=run.diagnostics,
        test=run.test,
        target=run.target,
        exact=_scale_leading_metrics(leading_corrected_at_N1, N),
        identity=_scale_leading_metrics(leading_identity_at_N1, N),
        actual=actual.to_metrics_dict(),
    )


def _scale_leading_metrics(metrics_at_N1: dict, N: int) -> dict:
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive.")

    bias_sq = float(metrics_at_N1["bias_sq"]) / (N**2)
    variance = float(metrics_at_N1["variance"]) / N
    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "mse": bias_sq + variance,
        "bias_abs_mean": _scale_optional_metric(metrics_at_N1.get("bias_abs_mean", np.nan), N),
        "bias_sq_max": _scale_optional_metric(metrics_at_N1.get("bias_sq_max", np.nan), N**2),
        "variance_max": _scale_optional_metric(metrics_at_N1.get("variance_max", np.nan), N),
    }


def _scale_optional_metric(value: float, denominator: int) -> float:
    value = float(value)
    if not np.isfinite(value):
        return value
    return value / denominator


def _failed_row(item: dict, *, N: int, trial: int, trial_seed: int, error: Exception) -> dict:
    return {
        "d": int(item["d"]),
        "nout": int(item["nout"]),
        "ntr": int(item["ntr"]),
        "N": int(N),
        "noise": "multinomial",
        "trial": int(trial),
        "trial_seed": int(trial_seed),
        "failed": True,
        "error": repr(error),
    }


def _run_item_rows(item: dict, *, progress_kwargs: dict | None = None) -> pd.DataFrame:
    master_rng = np.random.default_rng(int(item["seed"]))
    rows = []
    N_values = tuple(int(value) for value in item["N_values"])

    from tqdm import tqdm

    pbar = tqdm(
        total=REPETITIONS * len(N_values),
        unit="trial",
        **(progress_kwargs or {}),
    )
    try:
        for trial in range(REPETITIONS):
            trial_seed = int(master_rng.integers(0, 2**32 - 1))
            rng = np.random.default_rng(trial_seed)

            try:
                run = QELMRun(_base_spec_from_item(item, N=N_values[0]), rng=rng)
                # Resolve the N-independent objects once, then reuse them across N.
                _ = run.context
                _ = run.test
                _ = run.target
                _ = run.diagnostics
                spec_at_N1 = replace(
                    run.spec,
                    noise=replace(run.spec.noise, N=1),
                )
                leading_corrected_at_N1 = _compute_qelm_leading_error_resolved(
                    spec_at_N1,
                    run.context,
                    target=run.target,
                    test=run.test,
                    diagnostics=run.diagnostics,
                    corrected=True,
                ).to_metrics_dict()
                leading_identity_at_N1 = _compute_qelm_leading_error_resolved(
                    spec_at_N1,
                    run.context,
                    target=run.target,
                    test=run.test,
                    diagnostics=run.diagnostics,
                    corrected=False,
                ).to_metrics_dict()
            except Exception as exc:
                if not FAIL_SOFT:
                    raise
                for N in N_values:
                    rows.append(
                        _failed_row(
                            item,
                            N=N,
                            trial=trial,
                            trial_seed=trial_seed,
                            error=exc,
                        )
                    )
                    pbar.update(1)
                continue

            for N in N_values:
                try:
                    row = _row_for_N(
                        item=item,
                        rng=rng,
                        run=run,
                        N=N,
                        leading_corrected_at_N1=leading_corrected_at_N1,
                        leading_identity_at_N1=leading_identity_at_N1,
                    )
                    row["failed"] = False
                    row["error"] = ""
                except Exception as exc:
                    if not FAIL_SOFT:
                        raise
                    row = _failed_row(
                        item,
                        N=N,
                        trial=trial,
                        trial_seed=trial_seed,
                        error=exc,
                    )
                row["trial"] = trial
                row["trial_seed"] = trial_seed
                rows.append(row)
                pbar.update(1)
    finally:
        pbar.close()

    return pd.DataFrame(rows)


def _run_item(item: dict, worker_index: int | None = None) -> dict:
    desc = _item_label(item)
    if worker_index is not None:
        desc = f"worker {worker_index}: {desc}"

    started_at = datetime.now().astimezone()
    start_time = perf_counter()
    raw_df = _run_item_rows(
        item,
        progress_kwargs={
            "desc": desc,
            "position": None if worker_index is None else worker_index - 1,
            "leave": False,
            "dynamic_ncols": False,
            "ncols": 160,
        },
    )
    elapsed_seconds = perf_counter() - start_time
    completed_at = datetime.now().astimezone()
    metadata = _training_metadata(
        _study_from_item(item),
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed_seconds,
    )
    saved_path = _save_traindata(
        item["output_file"],
        dataraw=_saved_dataraw(raw_df, sweep_col="N"),
        metadata=metadata,
        overwrite=OVERWRITE,
    )
    return {
        "d": int(item["d"]),
        "nout": int(item["nout"]),
        "ntr": int(item["ntr"]),
        "output_file": str(saved_path),
    }


def _run_worker(payload: dict) -> list[dict]:
    return [
        _run_item(item, worker_index=int(payload["worker_index"]))
        for item in payload["items"]
    ]


def _init_tqdm_lock(lock) -> None:
    from tqdm import tqdm

    tqdm.set_lock(lock)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = _work_items()
    worker_count = min(MAX_PROCESSES, os.cpu_count() or 1, len(items))
    worker_count = max(1, worker_count)
    payloads = [
        {"worker_index": index + 1, "items": chunk}
        for index, chunk in enumerate(_split_work_items(items, worker_count))
    ]

    print(
        f"Running {len(items)} ntr checkpoint file(s) with {REPETITIONS} repetitions "
        f"and {len(N_VALUES)} N value(s) per repetition across "
        f"{worker_count} worker process(es).",
        flush=True,
    )

    if worker_count == 1:
        result_batches = [_run_worker(payloads[0])]
    else:
        context = get_context("spawn")
        lock = context.RLock()
        from tqdm import tqdm

        tqdm.set_lock(lock)
        with context.Pool(
            processes=worker_count,
            initializer=_init_tqdm_lock,
            initargs=(lock,),
        ) as pool:
            result_batches = pool.map(_run_worker, payloads)

    results = [result for batch in result_batches for result in batch]
    print(f"Completed {len(results)} file(s) in {OUTPUT_DIR}.", flush=True)


if __name__ == "__main__":
    main()
