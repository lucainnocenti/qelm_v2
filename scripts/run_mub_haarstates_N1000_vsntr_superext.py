# scripts/run_mub_haarstates_N1000_vsntr_ext.py

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qelm import (
    QELMDataSpec,
    QELMNoiseSpec,
    QELMTargetRequest,
    QELMTestRequest,
    QELMTrainingSpec,
    TrainingStudySpec,
    run_training_and_report_results,
)


def main():
    output_file = PROJECT_ROOT / "data" / "mub_haarstates_N1000_vsntr_extended.zip"
    output_file.parent.mkdir(exist_ok=True)

    training_spec = QELMTrainingSpec(
        data=QELMDataSpec(d=2, povm="qubit_mub", train_states="haar_pure"),
        target=QELMTargetRequest(observable="haar_pure_average"),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(noise="multinomial", N=1000, actual_noise_trials=1),
    )

    training_study = TrainingStudySpec(
        base=training_spec,
        sweep_col="ntr",
        sweep_values=(6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10**4),
        repetitions=1000,
        quantiles=(0.10, 0.25, 0.75, 0.90), quantile_band=(0.1, 0.9),
        show_summary=False, show_slopes=False, make_plots=False,
        output_file=output_file,
    )

    run_training_and_report_results(training_study)


if __name__ == "__main__":
    main()
