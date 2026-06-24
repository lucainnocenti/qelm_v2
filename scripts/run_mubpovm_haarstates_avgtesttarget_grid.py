# scripts/run_mubpovm_haarstates_avgtesttarget_grid.py

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path
import os
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qelm import (  # noqa: E402
    QELMDataSpec,
    QELMNoiseSpec,
    QELMTargetRequest,
    QELMTestRequest,
    QELMTrainingSpec,
    TrainingStudySpec,
    run_training_and_report_results,
)


GRID_VALUES = tuple(2**power for power in range(0, 12))
NTR_VALUES = tuple(value for value in GRID_VALUES if value >= 6)

REPETITIONS = 1000
SEED = 20260620
MAX_PROCESSES = 5
FAIL_SOFT = False
OVERWRITE = False

OUTPUT_DIR = PROJECT_ROOT / "data" / "mubpovm_haarstates_avgtesttarget_grid"


def _work_items() -> list[dict]:
    rng = np.random.default_rng(SEED)
    return [
        {
            "N": int(N),
            "seed": int(rng.integers(0, 2**32 - 1)),
            "output_file": str(OUTPUT_DIR / f"_N{int(N)}_vsntr.zip"),
        }
        for N in GRID_VALUES
    ]


def _study_from_item(item: dict) -> TrainingStudySpec:
    base = QELMTrainingSpec(
        data=QELMDataSpec(d=2, povm="qubit_mub", train_states="haar_pure"),
        target=QELMTargetRequest(observable="haar_pure_average"),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(
            noise="multinomial",
            N=int(item["N"]),
            actual_noise_trials=1,
        ),
    )

    return TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=NTR_VALUES,
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


def _run_item(item: dict, worker_index: int | None = None) -> dict:
    desc = f"N={item['N']} ntr={NTR_VALUES[0]}..{NTR_VALUES[-1]} ({len(NTR_VALUES)})"
    if worker_index is not None:
        desc = f"worker {worker_index}: {desc}"

    run_training_and_report_results(
        _study_from_item(item),
        progress_kwargs={
            "desc": desc,
            "position": None if worker_index is None else worker_index - 1,
            "leave": False,
            "dynamic_ncols": False,
            "ncols": 160,
        },
        quiet=True,
    )
    return {"N": int(item["N"]), "output_file": str(item["output_file"])}


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
    payloads = [
        {"worker_index": index + 1, "items": items[index::worker_count]}
        for index in range(worker_count)
    ]

    print(
        f"Running {len(items)} ntr sweep file(s) with {REPETITIONS} repetitions "
        f"per ntr value across {worker_count} worker process(es).",
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
