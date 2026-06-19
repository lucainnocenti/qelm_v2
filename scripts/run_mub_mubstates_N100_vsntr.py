# scripts/run_mub_mubstates_N100_vsntr.py

from pathlib import Path

from qelm import (
    QELMDataSpec,
    QELMNoiseSpec,
    QELMTargetRequest,
    QELMTestRequest,
    QELMTrainingSpec,
    TildeUTrainingApproxStudySpec,
    run_tilde_u_training_approx_report,
)


def main():
    Path("data").mkdir(exist_ok=True)

    tilde_u_training_spec = QELMTrainingSpec(
        data=QELMDataSpec(d=2, povm="qubit_mub", train_states="haar_pure"),
        target=QELMTargetRequest(observable="haar_pure_average"),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(noise="multinomial", N=100, actual_noise_trials=1),
    )

    tilde_u_study = TildeUTrainingApproxStudySpec(
        base=tilde_u_training_spec,
        sweep_col="ntr",
        sweep_values=(64, 128, 256, 512, 1024, 2048, 4096),
        repetitions=1000,
        quantiles=(0.10, 0.25, 0.75, 0.90),
        quantile_band=(0.1, 0.9),
        show_summary=False,
        show_slopes=False,
        plots="mse",
        output_file="data/mub_mubstates_N100_vsntr.zip",
    )

    raw, summary, slopes = run_tilde_u_training_approx_report(tilde_u_study)
    print(summary)
    if slopes is not None:
        print(slopes)


if __name__ == "__main__":
    main()
