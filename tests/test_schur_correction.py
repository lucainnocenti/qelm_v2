import numpy as np
import pytest

from qelm import (
    POVM,
    QELMDataSpec,
    QELMNoiseSpec,
    QELMQuantumDataset,
    QELMTargetRequest,
    QELMTestRequest,
    QELMTrainingSpec,
    QuantumStateBatch,
    TrainingStudySpec,
    generate_random_rank1_povm,
    leading_training_bias_variance_terms,
    leading_training_bias_variance_terms_target_average,
    run_training_experiment,
    one_schur_correction_trial,
    run_schur_complement_approx_experiment,
    run_schur_correction_report,
    run_schur_correction_scaling_experiment,
    tilde_u_correction_operator_diagnostics,
    validate_probability_matrix,
)


EXPLICIT_TARGET = np.array([[1.0, 0.0], [0.0, 0.0]])


def _training_study(
    *,
    d=2,
    nout=8,
    ntr=12,
    N=20,
    povm=None,
    train_states=None,
    target_observable=EXPLICIT_TARGET,
    target_normalization="none",
    test_state=None,
    repetitions=1,
    actual_noise_trials=1,
    noise="gaussian",
    seed=123,
    verbose=False,
    sweep_values=None,
) -> TrainingStudySpec:
    if train_states is None:
        train_states = {"kind": "haar_pure", "num_states": ntr}
    if isinstance(povm, np.ndarray):
        povm = {"effects": povm, "label": "test_povm"}
    base = QELMTrainingSpec(
        data=QELMDataSpec(
            d=d,
            nout=nout,
            povm=povm,
            train_states=train_states,
        ),
        target=QELMTargetRequest(
            observable=target_observable,
            normalization=target_normalization,
        ),
        test=QELMTestRequest(state=test_state),
        noise=QELMNoiseSpec(
            N=N,
            noise=noise,
            actual_noise_trials=actual_noise_trials,
        ),
    )
    return TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(ntr,) if sweep_values is None else sweep_values,
        repetitions=repetitions,
        seed=seed,
        verbose=verbose,
        show_summary=False,
        show_slopes=False,
        make_plots=False,
    )


def _run_training(**kwargs):
    return run_training_experiment(_training_study(**kwargs))


def test_random_isometry_povm_probability_matrix_is_column_stochastic():
    rng = np.random.default_rng(123)

    povm = POVM.random_isometry(nout=8, dim=2, rng=rng)
    states = QuantumStateBatch.haar_pure_from_columns(num_states=20, dim=2, rng=rng)
    P = povm.probability_matrix(states)

    assert P.shape == (8, 20)
    validate_probability_matrix(P, atol=1e-10)


def test_random_isometry_train_test_probability_matrices_share_outcomes():
    rng = np.random.default_rng(123)

    povm = POVM.random_isometry(nout=8, dim=2, rng=rng)
    train_states = QuantumStateBatch.haar_pure_from_columns(num_states=20, dim=2, rng=rng)
    test_states = QuantumStateBatch.haar_pure_from_columns(num_states=5, dim=2, rng=rng)
    P_train = povm.probability_matrix(train_states)
    P_test = povm.probability_matrix(test_states)

    assert P_train.shape == (8, 20)
    assert P_test.shape == (8, 5)
    validate_probability_matrix(P_train, atol=1e-10)
    validate_probability_matrix(P_test, atol=1e-10)


def test_run_training_experiment_default_haar_test_state_shapes():
    raw, summary = _run_training(
        repetitions=2,
        actual_noise_trials=2,
    )

    assert len(raw) == 2
    assert set(raw["test_state"]) == {"haar_pure_average"}
    assert set(raw["test_average"]) == {"exact_haar_second_moment"}
    assert set(raw["num_test_points"]) == {0}
    assert len(summary) == 1
    assert "actual_mse_median" in summary.columns


def test_run_training_experiment_accepts_explicit_povm_effects():
    rng = np.random.default_rng(456)
    effects = generate_random_rank1_povm(nout=8, dim=2, rng=rng)

    raw, summary = _run_training(
        povm=effects,
        ntr=16,
        repetitions=1,
        actual_noise_trials=1,
    )

    assert len(raw) == 1
    assert set(raw["d"]) == {2}
    assert set(raw["nout"]) == {8}
    assert len(summary) == 1
    assert set(summary["nout"]) == {8}


def test_run_training_experiment_accepts_povm_dictionary_effects():
    rng = np.random.default_rng(456)
    effects = generate_random_rank1_povm(nout=8, dim=2, rng=rng)

    raw, summary = _run_training(
        povm={"effects": effects, "label": "test_povm"},
        repetitions=1,
        actual_noise_trials=1,
    )

    assert len(raw) == 1
    assert set(raw["nout"]) == {8}
    assert len(summary) == 1


def test_run_training_experiment_accepts_random_povm_spec():
    raw, summary = _run_training(
        povm={"kind": "random_rank1", "nout": 8, "dim": 2},
        ntr=16,
        repetitions=1,
        actual_noise_trials=1,
    )

    assert len(raw) == 1
    assert set(raw["d"]) == {2}
    assert set(raw["nout"]) == {8}
    assert len(summary) == 1


def test_run_training_experiment_haar_sample_test_state_shapes():
    raw, summary = _run_training(
        repetitions=2,
        actual_noise_trials=2,
        test_state=("haar_sample", 3),
    )

    assert len(raw) == 2
    assert set(raw["test_state"]) == {"haar_sample"}
    assert set(raw["test_average"]) == {"sampled_haar_states"}
    assert set(raw["num_test_points"]) == {3}
    assert len(summary) == 1
    assert "actual_mse_median" in summary.columns


def test_run_training_experiment_fixed_test_state_vector_shapes():
    psi = np.array([1.0, 1.0j]) / np.sqrt(2.0)

    raw, summary = _run_training(
        repetitions=2,
        actual_noise_trials=2,
        test_state=psi,
    )

    assert len(raw) == 2
    assert set(raw["test_state"]) == {"fixed_state"}
    assert set(raw["test_average"]) == {"fixed_state"}
    assert set(raw["num_test_points"]) == {1}
    assert len(summary) == 1
    assert "actual_mse_median" in summary.columns


def test_run_training_experiment_test_state_vector_is_fixed_state():
    psi = np.array([1.0, 1.0j]) / np.sqrt(2.0)

    raw, summary = _run_training(
        repetitions=1,
        actual_noise_trials=1,
        test_state=psi,
    )

    assert len(raw) == 1
    assert set(raw["test_state"]) == {"fixed_state"}
    assert set(raw["test_average"]) == {"fixed_state"}
    assert len(summary) == 1


def test_run_training_experiment_test_state_dictionary_selector():
    raw, summary = _run_training(
        repetitions=1,
        actual_noise_trials=1,
        test_state={"kind": "haar_sample", "num_points": 3},
    )

    assert len(raw) == 1
    assert set(raw["test_state"]) == {"haar_sample"}
    assert set(raw["num_test_points"]) == {3}
    assert len(summary) == 1


def test_run_training_experiment_fixed_test_state_density_shapes():
    rho = np.array([[1.0, 0.0], [0.0, 0.0]])

    raw, summary = _run_training(
        repetitions=1,
        actual_noise_trials=1,
        test_state=rho,
    )

    assert len(raw) == 1
    assert set(raw["test_state"]) == {"fixed_state"}
    assert set(raw["num_test_points"]) == {1}
    assert len(summary) == 1


def test_run_training_experiment_verbose_progress_does_not_keyerror():
    raw, summary = _run_training(
        repetitions=1,
        actual_noise_trials=1,
        verbose=True,
    )

    assert len(raw) == 1
    assert len(summary) == 1


def test_run_training_experiment_requires_target_observable():
    with pytest.raises(ValueError, match="Target observable is required"):
        _run_training(
            repetitions=1,
            actual_noise_trials=1,
            target_observable=None,
        )


def test_leading_terms_use_dual_test_probabilities_consistently():
    rng = np.random.default_rng(123)
    dataset = QELMQuantumDataset.random_isometry(
        nout=8,
        ntr=12,
        ntest=5,
        dim=2,
        rng=rng,
        rcond=1e-12,
    )
    P = dataset.P_train
    P_test = dataset.P_test
    dual_P_test = dataset.dual_P_test

    diag = tilde_u_correction_operator_diagnostics(P=P, rank=4)
    blocks = diag["blocks"]
    w = blocks["U1"] @ rng.standard_normal(4)

    from_columns = leading_training_bias_variance_terms(
        P=P,
        U1=blocks["U1"],
        U2=blocks["U2"],
        C22_inv_C21=diag["C22_inv_C21"],
        w_observable=w,
        singular_values=blocks["singular_values"],
        N=20,
        approximate_identity=False,
        test_probabilities=P_test,
        dual_test_probabilities=dual_P_test,
    )
    from_moments = leading_training_bias_variance_terms(
        P=P,
        U1=blocks["U1"],
        U2=blocks["U2"],
        C22_inv_C21=diag["C22_inv_C21"],
        w_observable=w,
        singular_values=blocks["singular_values"],
        N=20,
        approximate_identity=False,
        test_second_moment=P_test @ P_test.T / P_test.shape[1],
        dual_test_second_moment=dual_P_test @ dual_P_test.T / dual_P_test.shape[1],
    )

    np.testing.assert_allclose(from_columns["bias_sq"], from_moments["bias_sq"])
    np.testing.assert_allclose(from_columns["variance"], from_moments["variance"])
    np.testing.assert_allclose(from_columns["mse"], from_moments["mse"])


def test_leading_bias_uses_povm_dual_then_training_state_dual():
    rng = np.random.default_rng(123)
    dataset = QELMQuantumDataset.random_isometry(
        nout=8,
        ntr=12,
        ntest=1,
        dim=2,
        rng=rng,
        rcond=1e-12,
    )
    P = dataset.P_train
    state_rows = dataset._train_state_rows
    effect_rows = dataset._effect_rows
    test_state_rows = dataset.test_states._state_rows

    effect_frame = effect_rows.T @ effect_rows.conj()
    povm_dual_rows = (np.linalg.pinv(effect_frame, rcond=1e-12) @ effect_rows.T).T
    state_frame = state_rows.T @ state_rows.conj()
    composed_dual_rows = (np.linalg.pinv(state_frame, rcond=1e-12) @ povm_dual_rows.T).T
    composed_dual_test = (composed_dual_rows.conj() @ test_state_rows.T).real

    old_one_frame_rows = (np.linalg.pinv(state_frame, rcond=1e-12) @ effect_rows.T).T
    old_one_frame_dual_test = (old_one_frame_rows.conj() @ test_state_rows.T).real

    np.testing.assert_allclose(dataset.dual_P_test, composed_dual_test, atol=1e-12)
    assert not np.allclose(composed_dual_test, old_one_frame_dual_test, atol=1e-8)

    diag = tilde_u_correction_operator_diagnostics(P=P, rank=4)
    blocks = diag["blocks"]
    random_operator = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    observable = random_operator + random_operator.conj().T
    w = (povm_dual_rows.conj() @ observable.reshape(-1, 1)).real[:, 0]
    N = 20

    result = leading_training_bias_variance_terms(
        P=P,
        U1=blocks["U1"],
        U2=blocks["U2"],
        C22_inv_C21=diag["C22_inv_C21"],
        w_observable=w,
        singular_values=blocks["singular_values"],
        N=N,
        approximate_identity=False,
        test_probabilities=dataset.P_test,
        dual_test_probabilities=dataset.dual_P_test,
    )

    T = blocks["U1"] @ blocks["U1"].T - blocks["U2"] @ diag["C22_inv_C21"] @ blocks["U1"].T
    t_w = T @ w
    t_g_sigma = T @ composed_dual_test
    covariance_columns = (
        P.T @ (t_w[:, None] * t_g_sigma)
        - (P.T @ t_w)[:, None] * (P.T @ t_g_sigma)
    )
    expected_bias_sq = float((np.sum(covariance_columns[:, 0]) / N) ** 2)

    np.testing.assert_allclose(result["bias_sq"], expected_bias_sq, atol=1e-12)


def test_target_average_leading_terms_match_empirical_target_second_moment():
    rng = np.random.default_rng(123)
    dataset = QELMQuantumDataset.random_isometry(
        nout=8,
        ntr=12,
        ntest=5,
        dim=2,
        rng=rng,
        rcond=1e-12,
    )
    P = dataset.P_train
    P_test = dataset.P_test
    dual_P_test = dataset.dual_P_test

    diag = tilde_u_correction_operator_diagnostics(P=P, rank=4)
    blocks = diag["blocks"]
    target_weights = blocks["U1"] @ rng.standard_normal((4, 7))
    target_second = target_weights @ target_weights.T / target_weights.shape[1]

    averaged = leading_training_bias_variance_terms_target_average(
        P=P,
        U1=blocks["U1"],
        U2=blocks["U2"],
        C22_inv_C21=diag["C22_inv_C21"],
        target_second_moment=target_second,
        singular_values=blocks["singular_values"],
        N=20,
        approximate_identity=False,
        test_probabilities=P_test,
        dual_test_probabilities=dual_P_test,
    )
    fixed_rows = [
        leading_training_bias_variance_terms(
            P=P,
            U1=blocks["U1"],
            U2=blocks["U2"],
            C22_inv_C21=diag["C22_inv_C21"],
            w_observable=target_weights[:, i],
            singular_values=blocks["singular_values"],
            N=20,
            approximate_identity=False,
            test_probabilities=P_test,
            dual_test_probabilities=dual_P_test,
        )
        for i in range(target_weights.shape[1])
    ]

    np.testing.assert_allclose(
        averaged["bias_sq"],
        np.mean([row["bias_sq"] for row in fixed_rows]),
    )
    np.testing.assert_allclose(
        averaged["variance"],
        np.mean([row["variance"] for row in fixed_rows]),
    )
    np.testing.assert_allclose(
        averaged["mse"],
        np.mean([row["mse"] for row in fixed_rows]),
    )


def test_run_training_experiment_haar_target_average_shapes():
    raw, summary = _run_training(
        repetitions=2,
        actual_noise_trials=2,
        target_observable="haar_pure_average",
    )

    assert len(raw) == 2
    assert set(raw["target_kind"]) == {"haar_pure"}
    assert set(raw["target_average"]) == {"exact_haar_second_moment"}
    assert len(summary) == 1
    assert "target_kind" in summary.columns
    assert "actual_mse_median" in summary.columns


def test_run_training_experiment_operator_target_shapes():
    operator = np.array([[1.0, 0.0], [0.0, 0.0]])

    raw, summary = _run_training(
        repetitions=1,
        actual_noise_trials=1,
        target_observable=operator,
    )

    assert len(raw) == 1
    assert set(raw["target_kind"]) == {"operator"}
    assert set(raw["target_average"]) == {"single_observable"}
    assert len(summary) == 1


def test_run_training_experiment_pure_state_vector_target_shapes():
    psi = np.array([1.0, 1.0j]) / np.sqrt(2.0)

    raw, summary = _run_training(
        repetitions=1,
        actual_noise_trials=1,
        target_observable=psi,
    )

    assert len(raw) == 1
    assert set(raw["target_kind"]) == {"pure_state"}
    assert set(raw["target_average"]) == {"single_observable"}
    assert len(summary) == 1


def test_one_schur_correction_trial_seeded_regression():
    row = one_schur_correction_trial(
        d=2,
        nout=8,
        ntr=20,
        noise="gaussian",
        seed=123,
    )

    assert "r" not in row
    assert "q" not in row
    assert "p_kernel" not in row
    assert row["C22_kept_rank"] == 4
    assert row["P_numerical_rank"] == 4

    np.testing.assert_allclose(row["term_op"], 0.31753126755271266)
    np.testing.assert_allclose(row["Gamma_trace"], 0.05849208846508239)
    np.testing.assert_allclose(row["C22_lambda_min"], 0.02109039266191085)
    np.testing.assert_allclose(row["Pi2_diag_mean"], 0.8000000000000002)


def test_run_schur_correction_scaling_experiment_shapes():
    raw, summary, slopes = run_schur_correction_scaling_experiment(
        d=2,
        nout_values=(8, 12),
        ntr_multiplier=10,
        trials=2,
        noise="gaussian",
        seed=7,
        verbose=False,
    )

    assert len(raw) == 4
    assert not raw["failed"].any()
    assert len(summary) == 2
    assert len(slopes) == 5

    expected_summary_columns = {
        "d",
        "nout",
        "ntr",
        "term_op_median",
        "Gamma_trace_median",
        "C22_lambda_min_median",
    }
    assert expected_summary_columns <= set(summary.columns)
    assert {"r", "q", "p_kernel", "q_over_p"}.isdisjoint(summary.columns)


def test_run_schur_complement_approx_experiment_shapes():
    raw, summary = run_schur_complement_approx_experiment(
        d=2,
        nout=8,
        ntr_values=(12, 16),
        trials=2,
        noise="gaussian",
        seed=123,
        verbose=False,
        show_summary=False,
        make_plots=False,
    )

    assert len(raw) == 4
    assert len(summary) == 2

    expected_summary_columns = {
        "d",
        "nout",
        "ntr",
        "empirical_schur_op_median",
        "limit_relative_error_median",
        "xi11_relative_error_median",
        "limit_approx_ok_median",
        "xi11_approx_ok_median",
    }
    assert expected_summary_columns <= set(summary.columns)
    assert {"r", "q", "p_kernel"}.isdisjoint(summary.columns)


def test_run_schur_correction_report_ntr_shapes_without_output():
    raw, summary, slopes = run_schur_correction_report(
        sweep_col="ntr",
        d=2,
        nout=8,
        ntr_values=(12, 16),
        trials=2,
        noise="gaussian",
        seed=123,
        verbose=False,
        show_summary=False,
        make_plots=False,
    )

    assert len(raw) == 4
    assert len(summary) == 2
    assert len(slopes) == 5
