# scripts/run_rndnout16_haarstates_comparenoises_grid.py

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from multiprocessing import TimeoutError as MultiprocessingTimeoutError
from multiprocessing import current_process, get_context
from pathlib import Path
from time import perf_counter
import os
import signal
import sys

# On Windows, Ctrl-C is delivered to every process attached to the console.
# Let the parent process handle interruption and terminate the pool so worker
# imports do not each emit a traceback while starting up.
if current_process().name != "MainProcess":
    signal.signal(signal.SIGINT, signal.SIG_IGN)

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qelm import (  # noqa: E402
    QELMDataSpec,
    QELMNoiseSpec,
    QELMRun,
    QELMTargetRequest,
    QELMTestRequest,
    QELMTrainingSpec,
    TrainingStudySpec,
)
from qelm.training import _run_qelm_training_resolved  # noqa: E402
from qelm.workflows import _save_traindata, _training_metadata  # noqa: E402


D = 2
NOUT = 16
NTR_VALUES = tuple(2**power for power in range(4, 14))
N_VALUES = tuple(2**power for power in range(0, 12))
NOISE_MODELS = ("multinomial", "gaussian", "centered_gaussian")

REPETITIONS = 1000
SEED = 20260625
MAX_PROCESSES = 5
FAIL_SOFT = False
OVERWRITE = False

OUTPUT_DIR = PROJECT_ROOT / "data" / "rndnout16_haarstates_comparenoises_grid"
OUTPUT_FILE = OUTPUT_DIR / "_d2_nout16_noise_models_vsntr_vsN.zip"
SCAN_DIR = OUTPUT_DIR


def _base_spec(*, ntr: int, N: int | None, noise: str = "multinomial") -> QELMTrainingSpec:
    return QELMTrainingSpec(
        data=QELMDataSpec(
            d=D,
            nout=NOUT,
            povm="random_rank1",
            train_states={
                "kind": "haar_pure",
                "num_states": int(ntr),
            },
        ),
        target=QELMTargetRequest(observable="haar_pure_average"),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(
            noise=noise,
            N=None if N is None else int(N),
            actual_noise_trials=1,
        ),
    )


def _metadata_study() -> TrainingStudySpec:
    return TrainingStudySpec(
        base=QELMTrainingSpec(
            data=QELMDataSpec(
                d=D,
                nout=NOUT,
                povm="random_rank1",
                train_states="haar_pure",
            ),
            target=QELMTargetRequest(observable="haar_pure_average"),
            test=QELMTestRequest(state="haar_pure_average"),
            noise=QELMNoiseSpec(
                noise="multiple",
                N=None,
                actual_noise_trials=1,
            ),
        ),
        sweep_col="N",
        sweep_values=N_VALUES,
        repetitions=REPETITIONS,
        seed=SEED,
        quantiles=(0.10, 0.25, 0.75, 0.90),
        quantile_band=(0.1, 0.9),
        show_summary=False,
        show_slopes=False,
        make_plots=False,
        verbose=True,
        fail_soft=FAIL_SOFT,
        output_file=OUTPUT_FILE,
        overwrite=OVERWRITE,
    )


def _work_items() -> list[dict]:
    rng = np.random.default_rng(SEED)
    return [
        {
            "ntr": int(ntr),
            "seed": int(rng.integers(0, 2**32 - 1)),
        }
        for ntr in NTR_VALUES
    ]


def _failed_rows(
    *,
    ntr: int,
    trial: int,
    trial_seed: int,
    error: Exception,
) -> list[dict]:
    rows = []
    for N in N_VALUES:
        row = {
            "trial": int(trial),
            "trial_seed": int(trial_seed),
            "d": D,
            "nout": NOUT,
            "ntr": int(ntr),
            "N": int(N),
            "failed": True,
            "error": repr(error),
        }
        for noise in NOISE_MODELS:
            row[f"mse_{noise}"] = np.nan
            row[f"failed_{noise}"] = True
            row[f"error_{noise}"] = repr(error)
        rows.append(row)
    return rows


def _run_ntr_rows(item: dict, *, progress_kwargs: dict | None = None) -> pd.DataFrame:
    ntr = int(item["ntr"])
    master_rng = np.random.default_rng(int(item["seed"]))
    rows = []

    from tqdm import tqdm

    pbar = tqdm(
        total=REPETITIONS * len(N_VALUES) * len(NOISE_MODELS),
        unit="fit",
        **(progress_kwargs or {}),
    )
    try:
        for trial in range(REPETITIONS):
            trial_seed = int(master_rng.integers(0, 2**32 - 1))
            rng = np.random.default_rng(trial_seed)

            try:
                run = QELMRun(_base_spec(ntr=ntr, N=N_VALUES[0]), rng=rng)
                # Resolve only objects needed by actual least-squares MSE.
                context = run.context
                target = run.target
                test = run.test
            except Exception as exc:
                if not FAIL_SOFT:
                    raise
                rows.extend(
                    _failed_rows(
                        ntr=ntr,
                        trial=trial,
                        trial_seed=trial_seed,
                        error=exc,
                    )
                )
                pbar.update(len(N_VALUES) * len(NOISE_MODELS))
                continue

            for N in N_VALUES:
                row = {
                    "trial": int(trial),
                    "trial_seed": int(trial_seed),
                    "d": D,
                    "nout": NOUT,
                    "ntr": ntr,
                    "N": int(N),
                    "failed": False,
                    "error": "",
                }
                for noise in NOISE_MODELS:
                    try:
                        spec = replace(
                            run.spec,
                            noise=replace(run.spec.noise, N=int(N), noise=noise),
                        )
                        actual = _run_qelm_training_resolved(
                            spec,
                            rng=rng,
                            context=context,
                            target=target,
                            test=test,
                        ).to_metrics_dict()
                        row[f"mse_{noise}"] = actual["mse"]
                        row[f"failed_{noise}"] = False
                        row[f"error_{noise}"] = ""
                    except Exception as exc:
                        if not FAIL_SOFT:
                            raise
                        row["failed"] = True
                        row["error"] = repr(exc)
                        row[f"mse_{noise}"] = np.nan
                        row[f"failed_{noise}"] = True
                        row[f"error_{noise}"] = repr(exc)
                    pbar.update(1)
                rows.append(row)
    finally:
        pbar.close()

    return pd.DataFrame(rows)


def _item_label(item: dict) -> str:
    return f"d={D} nout={NOUT} ntr={int(item['ntr'])}"


def _run_item(item: dict, worker_index: int | None = None) -> pd.DataFrame:
    desc = _item_label(item)
    if worker_index is not None:
        desc = f"worker {worker_index}: {desc}"
    return _run_ntr_rows(
        item,
        progress_kwargs={
            "desc": desc,
            "position": None if worker_index is None else worker_index - 1,
            "leave": False,
            "dynamic_ncols": False,
            "ncols": 160,
        },
    )


def _worker_index() -> int | None:
    identity = current_process()._identity
    if not identity:
        return None
    return ((int(identity[0]) - 1) % max(1, MAX_PROCESSES)) + 1


def _run_item_payload(payload: dict) -> pd.DataFrame:
    return _run_item(payload["item"], worker_index=_worker_index())


def _init_tqdm_lock(lock) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from tqdm import tqdm

    tqdm.set_lock(lock)


def _metadata(
    *,
    started_at: datetime,
    completed_at: datetime,
    elapsed_seconds: float,
    ntr_values,
    partial: bool,
) -> dict:
    ntr_values = [int(value) for value in ntr_values]
    metadata = _training_metadata(
        _metadata_study(),
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed_seconds,
    )
    metadata["scan"] = {
        "ntr_values": ntr_values,
        "N_values": [int(value) for value in N_VALUES],
        "noise_models": list(NOISE_MODELS),
    }
    metadata["data"]["train_state_counts"] = ntr_values
    if len(ntr_values) == 1:
        metadata["data"]["train_state_count"] = ntr_values[0]
    metadata["noise"]["noise_models"] = list(NOISE_MODELS)
    metadata["partial"] = bool(partial)
    metadata["notes"] = (
        "Actual least-squares MSE only. Leading terms and block diagnostics "
        "are intentionally not computed."
    )
    return metadata


def _save_ntr_scan_file(
    frame: pd.DataFrame,
    *,
    scan_dir: Path,
    started_at: datetime,
    start_time: float,
) -> Path | None:
    if frame.empty:
        return None
    ntr_values = sorted(int(value) for value in frame["ntr"].dropna().unique())
    if len(ntr_values) != 1:
        raise ValueError(f"Expected one ntr value per scan file, got {ntr_values}.")

    ntr = ntr_values[0]
    completed_at = datetime.now().astimezone()
    metadata = _metadata(
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=perf_counter() - start_time,
        ntr_values=ntr_values,
        partial=True,
    )
    saved_path = _save_traindata(
        scan_dir / f"ntr_{ntr:05d}_vsN.zip",
        dataraw=_dataraw(frame),
        metadata=metadata,
        overwrite=True,
    )
    print(f"Saved scan file for ntr={ntr} to {saved_path}", flush=True)
    return saved_path


def _collect_pool_results(
    pool,
    items: list[dict],
    *,
    scan_dir: Path,
    started_at: datetime,
    start_time: float,
) -> list[pd.DataFrame]:
    iterator = pool.imap_unordered(
        _run_item_payload,
        [{"item": item} for item in items],
        chunksize=1,
    )
    frames = []
    while len(frames) < len(items):
        try:
            frame = iterator.next(timeout=0.5)
        except MultiprocessingTimeoutError:
            continue
        frames.append(frame)
        _save_ntr_scan_file(
            frame,
            scan_dir=scan_dir,
            started_at=started_at,
            start_time=start_time,
        )
    return frames


def _dataraw(raw_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trial",
        "trial_seed",
        "d",
        "nout",
        "ntr",
        "N",
    ]
    for noise in NOISE_MODELS:
        columns.append(f"mse_{noise}")
    if "failed" in raw_df.columns and raw_df["failed"].fillna(False).any():
        columns.extend(["failed", "error"])
        for noise in NOISE_MODELS:
            columns.extend([f"failed_{noise}", f"error_{noise}"])
    return raw_df.loc[:, [col for col in columns if col in raw_df.columns]]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = _work_items()
    worker_count = min(MAX_PROCESSES, os.cpu_count() or 1, len(items))
    worker_count = max(1, worker_count)

    print(
        f"Running actual-MSE scans for {len(NTR_VALUES)} ntr value(s), "
        f"{len(N_VALUES)} N value(s), {len(NOISE_MODELS)} noise model(s), "
        f"and {REPETITIONS} repetitions across {worker_count} worker process(es).",
        flush=True,
    )

    started_at = datetime.now().astimezone()
    start_time = perf_counter()
    scan_dir = SCAN_DIR
    scan_dir.mkdir(parents=True, exist_ok=True)

    try:
        if worker_count == 1:
            frames = []
            for item in items:
                frame = _run_item(item)
                frames.append(frame)
                _save_ntr_scan_file(
                    frame,
                    scan_dir=scan_dir,
                    started_at=started_at,
                    start_time=start_time,
                )
        else:
            context = get_context("spawn")
            lock = context.RLock()
            from tqdm import tqdm

            tqdm.set_lock(lock)
            pool = None
            try:
                pool = context.Pool(
                    processes=worker_count,
                    initializer=_init_tqdm_lock,
                    initargs=(lock,),
                )
                frames = _collect_pool_results(
                    pool,
                    items,
                    scan_dir=scan_dir,
                    started_at=started_at,
                    start_time=start_time,
                )
            except BaseException:
                if pool is not None:
                    pool.terminate()
                raise
            else:
                pool.close()
            finally:
                if pool is not None:
                    pool.join()
    except KeyboardInterrupt:
        print(
            f"\nInterrupted. Completed ntr scan files remain in {scan_dir}.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = raw_df.sort_values(["ntr", "trial", "N"], kind="stable")

    elapsed_seconds = perf_counter() - start_time
    completed_at = datetime.now().astimezone()
    metadata = _metadata(
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed_seconds,
        ntr_values=NTR_VALUES,
        partial=False,
    )

    saved_path = _save_traindata(
        OUTPUT_FILE,
        dataraw=_dataraw(raw_df),
        metadata=metadata,
        overwrite=OVERWRITE,
    )
    print(f"Saved combined actual-MSE noise-model data to {saved_path}", flush=True)


if __name__ == "__main__":
    main()
