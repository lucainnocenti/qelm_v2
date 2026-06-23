# scripts/run_rndpovm_haarstates_avgtesttarget_grid.py

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


DIMS = (2, 4)
POWERS = tuple(range(2, 14))
GRID_VALUES = tuple(2**power for power in POWERS)

REPETITIONS = 1000
SEED = 20260620
MAX_PROCESSES = 5
FAIL_SOFT = False
OVERWRITE = False

OUTPUT_DIR = PROJECT_ROOT / "data" / "rndpovm_haarstates_avgtesttarget_grid"
OUTPUT_STEM = ""

def _nout_values(d: int) -> tuple[int, ...]:
    rank = int(d)**2
    return tuple(int(nout) for nout in GRID_VALUES if int(nout) > rank)


def _ntr_values(nout: int) -> tuple[int, ...]:
    # used by _work_items to generate the ntr values for a given nout value
    return tuple(int(ntr) for ntr in GRID_VALUES if int(ntr) >= int(nout))


def _output_file(*, d: int, nout: int, N: int) -> Path:
    # used by _work_items to generate the output file path for a given d, nout, and N value
    return OUTPUT_DIR / f"{OUTPUT_STEM}_d{int(d)}_nout{int(nout)}_N{int(N)}_vsntr.zip"


def _work_items() -> list[dict]:
    # returns a list of work items, where each item is a dict containing the parameters for a single run
    rng = np.random.default_rng(SEED)
    items = []

    for d in DIMS:
        # _nout_values ensures nout > d^2
        for nout in _nout_values(d):
            # _ntr_values ensures ntr >= nout
            sweep_values = _ntr_values(nout)
            if not sweep_values:
                continue

            for N in GRID_VALUES:
                items.append(
                    {
                        "d": int(d),
                        "nout": int(nout),
                        "N": int(N),
                        "sweep_values": sweep_values,
                        "seed": int(rng.integers(0, 2**32 - 1)),
                        "output_file": str(_output_file(d=d, nout=nout, N=N)),
                    }
                )

    return items


def _item_cost(item: dict) -> int:
    return int(item["nout"]) * sum(int(value) for value in item["sweep_values"])


def _split_work_items(items: list[dict], parts: int) -> list[list[dict]]:
    # this attempts to balance the work across the workers by sorting the items
    # by cost and assigning them to the worker with the least total cost so far
    # The cost is defined as the product of nout and the sum of the sweep values
    chunks = [[] for _ in range(parts)]
    loads = [0 for _ in range(parts)]

    for item in sorted(items, key=_item_cost, reverse=True):
        index = min(range(parts), key=lambda idx: loads[idx])
        chunks[index].append(item)
        loads[index] += _item_cost(item)

    for chunk in chunks:
        chunk.sort(key=lambda item: (item["d"], item["nout"], item["N"]))
    return chunks


def _study_from_item(item: dict) -> TrainingStudySpec:
    # prepare the study spec for a given work item, ie takes as input parameters
    # d, nout, N, and sweep_values, and returns a TrainingStudySpec object
    base = QELMTrainingSpec(
        data=QELMDataSpec(
            d=int(item["d"]),
            nout=int(item["nout"]),
            povm="random_rank1",
            train_states="haar_pure",
        ),
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
        sweep_values=tuple(int(value) for value in item["sweep_values"]),
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


def _run_item(item: dict, *, worker_index: int | None = None) -> dict:
    study = _study_from_item(item)
    bar_desc = (
        _item_label(item)
        if worker_index is None
        else f"worker {worker_index}: {_item_label(item)}"
    )
    run_training_and_report_results(
        study,
        progress_kwargs={
            "desc": bar_desc,
            "position": None if worker_index is None else worker_index - 1,
            "leave": False,
            "dynamic_ncols": False,
            "ncols": 160,
        },
        quiet=True,
    )
    return {
        "d": int(item["d"]),
        "nout": int(item["nout"]),
        "N": int(item["N"]),
        "ntr_values": len(item["sweep_values"]),
        "output_file": str(item["output_file"]),
    }


def _item_label(item: dict) -> str:
    sweep_values = tuple(int(value) for value in item["sweep_values"])
    ntr_label = (
        str(sweep_values[0])
        if len(sweep_values) == 1
        else f"{sweep_values[0]}..{sweep_values[-1]} ({len(sweep_values)})"
    )
    return f"d={item['d']} nout={item['nout']} N={item['N']} ntr={ntr_label}"


def _run_worker(payload: dict) -> list[dict]:
    worker_index = int(payload["worker_index"])
    items = list(payload["items"])
    results = []

    for item in items:
        results.append(_run_item(item, worker_index=worker_index))

    return results


def _init_tqdm_lock(lock) -> None:
    from tqdm import tqdm

    tqdm.set_lock(lock)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # each item is a dict containing the parameters for a single run
    items = _work_items()
    # ensure meaningful worker count
    worker_count = min(MAX_PROCESSES, os.cpu_count() or 1, len(items))
    worker_count = max(1, worker_count)

    print(
        f"Running {len(items)} ntr sweep file(s) with {REPETITIONS} repetitions "
        f"per ntr value across {worker_count} worker process(es).",
        flush=True,
    )

    # partitions the work items into chunks, to be processed by each worker
    chunks = _split_work_items(items, worker_count)
    payloads = [
        {"worker_index": index + 1, "items": chunk}
        for index, chunk in enumerate(chunks)
    ]

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
