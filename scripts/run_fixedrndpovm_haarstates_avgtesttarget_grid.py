# scripts/run_fixedrndpovm_haarstates_avgtesttarget_grid.py

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
    generate_random_rank1_povm,
    run_training_and_report_results,
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


def _fixed_povms(rng: np.random.Generator) -> dict[tuple[int, int], np.ndarray]:
    return {
        (d, nout): generate_random_rank1_povm(nout=nout, dim=d, rng=rng)
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
            "N": int(N),
            "sweep_values": _ntr_values(nout),
            "povm": {"effects": povms[(d, nout)], "label": f"fixed_random_rank1_d{d}_nout{nout}"},
            "seed": int(rng.integers(0, 2**32 - 1)),
            "output_file": str(OUTPUT_DIR / f"_d{d}_nout{nout}_N{int(N)}_vsntr.zip"),
        }
        for d in DIMS
        for nout in _nout_values(d)
        for N in N_VALUES
    ]


def _item_cost(item: dict) -> int:
    return int(item["nout"]) * sum(int(value) for value in item["sweep_values"])


def _split_work_items(items: list[dict], parts: int) -> list[list[dict]]:
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
    base = QELMTrainingSpec(
        data=QELMDataSpec(
            d=int(item["d"]),
            nout=int(item["nout"]),
            povm=item["povm"],
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


def _item_label(item: dict) -> str:
    sweep_values = tuple(int(value) for value in item["sweep_values"])
    ntr_label = (
        str(sweep_values[0])
        if len(sweep_values) == 1
        else f"{sweep_values[0]}..{sweep_values[-1]} ({len(sweep_values)})"
    )
    return f"d={item['d']} nout={item['nout']} N={item['N']} ntr={ntr_label}"


def _run_item(item: dict, worker_index: int | None = None) -> dict:
    desc = _item_label(item)
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
    return {
        "d": int(item["d"]),
        "nout": int(item["nout"]),
        "N": int(item["N"]),
        "output_file": str(item["output_file"]),
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
