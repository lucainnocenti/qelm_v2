# scripts/run_rndnout16_haarstates_N100_vsntr.py

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
    TildeUTrainingApproxStudySpec,
    run_tilde_u_training_approx_report,
)


def main():
    output_file = PROJECT_ROOT / "data" / "rndnout16_haarstates_N100_vsntr.zip"
    output_file.parent.mkdir(exist_ok=True)

    tilde_u_training_spec = QELMTrainingSpec(
        data=QELMDataSpec(d=2, nout=16, povm='random_rank1', train_states='haar_pure'),
        target=QELMTargetRequest(observable="haar_pure_average"),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(noise="multinomial", N=100, actual_noise_trials=1),
    )

    tilde_u_study = TildeUTrainingApproxStudySpec(
        base=tilde_u_training_spec,
        sweep_col="ntr",
        sweep_values=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10**4),
        repetitions=1000,
        quantiles=(0.10, 0.25, 0.75, 0.90), quantile_band=(0.1, 0.9),
        show_summary=False, show_slopes=False, make_plots=False,
        output_file=output_file,
    )

    run_tilde_u_training_approx_report(tilde_u_study)


if __name__ == "__main__":
    main()
