import numpy as np
import pandas as pd
import pytest

import qelm.training as training
import qelm.training_reports as training_reports
import qelm.workflows as workflows
from qelm import (
    QELMDataSpec,
    QELMNoiseSpec,
    QELMTargetRequest,
    QELMTestRequest,
    QELMRun,
    QELMTrainingSpec,
    POVM,
    QuantumStateBatch,
    TrainingReport,
    TrainingStudySpec,
    analyze_qelm_training,
    clear_default_rng,
    compute_qelm_diagnostics,
    compute_qelm_leading_error,
    qubit_mub_povm,
    generate_random_rank1_povm,
    leading_training_bias_variance_terms,
    load_traindata,
    make_qelm_training_context,
    plot_mse_grid_over_N,
    plot_leading_mse_difference_grid_over_N,
    plot_saved_traindata,
    resolve_qelm_target,
    resolve_qelm_test,
    run_qelm_actual_training,
    run_training_experiment,
    run_training_and_report_results,
    set_default_rng,
    summarize_traindata,
    tilde_u_correction_operator_diagnostics,
)


def _small_spec(*, povm=None, target=None, train_states=None, ntr=12):
    if train_states is None:
        train_states = {"kind": "haar_pure", "num_states": ntr}
    if isinstance(povm, np.ndarray):
        povm = {"effects": povm, "label": "test_povm"}
    return QELMTrainingSpec(
        data=QELMDataSpec(d=2, nout=8, povm=povm, train_states=train_states),
        target=QELMTargetRequest(observable=target, normalization="none"),
        test=QELMTestRequest(state=np.array([[1.0, 0.0], [0.0, 0.0]])),
        noise=QELMNoiseSpec(N=20, noise="gaussian", actual_noise_trials=2),
    )


def test_fixed_explicit_povm_is_used_in_training_context():
    rng = np.random.default_rng(123)
    povm = generate_random_rank1_povm(nout=8, dim=2, rng=rng)
    spec = _small_spec(povm=povm)

    context = make_qelm_training_context(spec, rng=np.random.default_rng(456))

    assert context.P_train.shape == (8, 12)
    np.testing.assert_allclose(context.povm.effects, povm)


def test_training_context_pairwise_quantities_are_lazy_and_cached():
    povm = generate_random_rank1_povm(nout=8, dim=2, rng=np.random.default_rng(123))
    spec = _small_spec(povm=povm)

    context = make_qelm_training_context(spec, rng=np.random.default_rng(456))

    assert context._P_train_cache is None
    assert context._dual_effect_rows_cache is None
    assert context._dual_P_train_cache is None
    assert context._test_second_cache is None
    assert context._dual_test_second_cache is None
    assert context.povm._effect_rows_cache is None
    assert context.train_states._state_rows_cache is None

    P_train = context.P_train

    assert context._P_train_cache is P_train
    assert context.P_train is P_train
    assert context.povm._effect_rows_cache is None
    assert context.train_states._state_rows_cache is None
    assert context._dual_effect_rows_cache is None

    dual_effect_rows = context.dual_effect_rows

    assert context._dual_effect_rows_cache is dual_effect_rows
    assert context.dual_effect_rows is dual_effect_rows
    assert context.rcond in context.povm._dual_effect_rows_cache
    assert context.rcond in context.train_states._frame_pinv_cache
    assert context.povm._effect_rows_cache is not None
    assert context.train_states._state_rows_cache is not None

    test_second = context.test_second

    assert context._test_second_cache is test_second
    assert context.test_second is test_second


def test_sampled_test_matrices_are_lazy_and_cached():
    povm = generate_random_rank1_povm(nout=8, dim=2, rng=np.random.default_rng(123))
    spec = QELMTrainingSpec(
        data=QELMDataSpec(
            d=2,
            nout=8,
            povm={"effects": povm, "label": "test_povm"},
            train_states={"kind": "haar_pure", "num_states": 12},
        ),
        target=QELMTargetRequest(observable=np.array([[1.0, 0.0], [0.0, 0.0]])),
        test=QELMTestRequest(state={"kind": "haar_sample", "num_points": 3}),
        noise=QELMNoiseSpec(N=20, noise="gaussian", actual_noise_trials=2),
    )

    context = make_qelm_training_context(spec, rng=np.random.default_rng(456))

    assert isinstance(context.test_states, QuantumStateBatch)
    assert context._P_test_cache is None
    assert context._dual_P_test_cache is None
    assert context._dual_effect_rows_cache is None

    P_test = context.P_test

    assert context._P_test_cache is P_test
    assert context.P_test is P_test
    assert context._dual_P_test_cache is None
    assert context._dual_effect_rows_cache is None

    dual_P_test = context.dual_P_test

    assert context._dual_P_test_cache is dual_P_test
    assert context.dual_P_test is dual_P_test
    assert context._dual_effect_rows_cache is not None


def test_explicit_array_povm_requires_label_for_training_context():
    rng = np.random.default_rng(123)
    povm = generate_random_rank1_povm(nout=8, dim=2, rng=rng)
    spec = QELMTrainingSpec(
        data=QELMDataSpec(
            d=2,
            povm=povm,
            train_states={"kind": "haar_pure", "num_states": 12},
        ),
        target=QELMTargetRequest(observable=np.array([[1.0, 0.0], [0.0, 0.0]])),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(N=20, noise="gaussian", actual_noise_trials=2),
    )

    with pytest.raises(ValueError, match="Explicit POVM specs require a non-empty label"):
        QELMRun(spec, rng=np.random.default_rng(456)).context


def test_explicit_labeled_povm_dictionary_allows_omitting_nout():
    rng = np.random.default_rng(123)
    povm = generate_random_rank1_povm(nout=8, dim=2, rng=rng)
    spec = QELMTrainingSpec(
        data=QELMDataSpec(
            d=2,
            povm={"effects": povm, "label": "test_povm"},
            train_states={"kind": "haar_pure", "num_states": 12},
        ),
        target=QELMTargetRequest(observable=np.array([[1.0, 0.0], [0.0, 0.0]])),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(N=20, noise="gaussian", actual_noise_trials=2),
    )

    run = QELMRun(spec, rng=np.random.default_rng(456))

    assert spec.data.nout == 8
    assert run.context.P_train.shape == (8, 12)
    np.testing.assert_allclose(run.context.povm.effects, povm)


def test_explicit_povm_object_allows_omitting_nout():
    effects = generate_random_rank1_povm(nout=8, dim=2, rng=np.random.default_rng(123))
    povm = POVM.from_effects(effects, dim=2, label="test_povm")
    spec = QELMTrainingSpec(
        data=QELMDataSpec(
            d=2,
            povm=povm,
            train_states={"kind": "haar_pure", "num_states": 12},
        ),
        target=QELMTargetRequest(observable=np.array([[1.0, 0.0], [0.0, 0.0]])),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(N=20, noise="gaussian", actual_noise_trials=2),
    )

    context = make_qelm_training_context(spec, rng=np.random.default_rng(456))

    assert spec.data.nout == 8
    assert context.P_train.shape == (8, 12)


def test_fixed_explicit_training_states_are_used_in_training_context():
    povm = generate_random_rank1_povm(nout=8, dim=2, rng=np.random.default_rng(123))
    train_states = QuantumStateBatch.haar_pure_from_columns(
        num_states=12,
        dim=2,
        rng=np.random.default_rng(456),
    )
    spec = _small_spec(povm=povm, train_states=train_states)

    context = make_qelm_training_context(spec, rng=np.random.default_rng(789))

    expected = POVM.from_effects(povm, dim=2, nout=8).probability_matrix(train_states)
    np.testing.assert_allclose(context.P_train, expected, atol=1e-12)
    assert isinstance(context.train_states, QuantumStateBatch)
    np.testing.assert_allclose(context.train_states.states, train_states.states)


def test_training_state_vector_columns_are_accepted():
    povm = generate_random_rank1_povm(nout=8, dim=2, rng=np.random.default_rng(123))
    train_vectors = np.array(
        [
            [1.0, 0.0, 1.0, 1.0j],
            [0.0, 1.0, 1.0, 1.0],
        ],
        dtype=complex,
    )
    spec = QELMTrainingSpec(
        data=QELMDataSpec(
            d=2,
            nout=8,
            povm={"effects": povm, "label": "test_povm"},
            train_states={
                "kind": "state_vectors",
                "vectors": train_vectors,
                "axis": "columns",
            },
        ),
        target=QELMTargetRequest(
            observable=np.array([[1.0, 0.0], [0.0, 0.0]]),
            normalization="none",
        ),
        test=QELMTestRequest(state=np.array([[1.0, 0.0], [0.0, 0.0]])),
        noise=QELMNoiseSpec(N=20, noise="gaussian", actual_noise_trials=2),
    )

    context = make_qelm_training_context(spec, rng=np.random.default_rng(789))

    assert context.P_train.shape == (8, 4)
    np.testing.assert_allclose(context.P_train.sum(axis=0), 1.0, atol=1e-10)


def test_qelm_data_spec_requires_training_states():
    with np.testing.assert_raises(TypeError):
        QELMDataSpec(d=2, nout=8)


def test_flexible_haar_training_states_require_sweep_count():
    spec = _small_spec(train_states={"kind": "haar_pure"})

    with np.testing.assert_raises(ValueError):
        make_qelm_training_context(spec, rng=np.random.default_rng(123))


@pytest.mark.parametrize(
    "train_states",
    [
        "haar",
        {"kind": "haar", "num_states": 12},
    ],
)
def test_old_haar_training_state_selector_is_rejected(train_states):
    spec = _small_spec(train_states=train_states)

    with np.testing.assert_raises_regex(ValueError, "train_states is required"):
        make_qelm_training_context(spec, rng=np.random.default_rng(123))


def test_random_povm_alias_is_rejected():
    spec = _small_spec(povm={"kind": "random", "nout": 8, "dim": 2})

    with np.testing.assert_raises_regex(ValueError, "Unknown POVM spec kind"):
        make_qelm_training_context(spec, rng=np.random.default_rng(123))


def test_random_rank1_string_povm_spec_builds_valid_context():
    spec = _small_spec(povm="random_rank1")

    context = make_qelm_training_context(spec, rng=np.random.default_rng(123))

    assert context.P_train.shape == (8, 12)
    np.testing.assert_allclose(context.P_train.sum(axis=0), 1.0, atol=1e-10)


def test_random_rank1_dict_povm_spec_builds_valid_context():
    spec = _small_spec(povm={"kind": "random_rank1", "nout": 8, "dim": 2})

    context = make_qelm_training_context(spec, rng=np.random.default_rng(123))

    assert context.P_train.shape == (8, 12)
    np.testing.assert_allclose(context.P_train.sum(axis=0), 1.0, atol=1e-10)


def test_qubit_mub_string_povm_spec_builds_valid_context_without_nout():
    spec = QELMTrainingSpec(
        data=QELMDataSpec(
            d=2,
            povm="qubit_mub",
            train_states={"kind": "haar_pure", "num_states": 12},
        ),
        target=QELMTargetRequest(observable=np.array([[1.0, 0.0], [0.0, 0.0]])),
        test=QELMTestRequest(state=np.array([[1.0, 0.0], [0.0, 0.0]])),
        noise=QELMNoiseSpec(N=20, noise="gaussian", actual_noise_trials=1),
    )

    context = make_qelm_training_context(spec, rng=np.random.default_rng(123))

    assert spec.data.nout == 6
    assert context.povm.label == "qubit_mub"
    np.testing.assert_allclose(context.povm.effects, qubit_mub_povm(), atol=1e-12)
    assert context.P_train.shape == (6, 12)


def test_qubit_mub_dict_povm_spec_rejects_wrong_dimension():
    spec = QELMTrainingSpec(
        data=QELMDataSpec(
            d=3,
            povm={"kind": "qubit_mub"},
            train_states={"kind": "haar_pure", "num_states": 12},
        ),
        target=QELMTargetRequest(observable=np.eye(3)),
        test=QELMTestRequest(state=np.eye(3) / 3),
        noise=QELMNoiseSpec(N=20, noise="gaussian", actual_noise_trials=1),
    )

    with pytest.raises(ValueError, match="dimension d=2"):
        make_qelm_training_context(spec, rng=np.random.default_rng(123))


def test_test_state_alias_is_rejected():
    spec = _small_spec(target=np.array([[1.0, 0.0], [0.0, 0.0]]))
    spec = QELMTrainingSpec(
        data=spec.data,
        target=spec.target,
        noise=spec.noise,
        test=QELMTestRequest(state="haar"),
        numerics=spec.numerics,
    )

    with np.testing.assert_raises_regex(ValueError, "test_state must be"):
        make_qelm_training_context(spec, rng=np.random.default_rng(123))


def test_target_alias_is_rejected():
    spec = _small_spec(target="haar_pure_state_average")
    context = make_qelm_training_context(spec, rng=np.random.default_rng(123))

    with np.testing.assert_raises_regex(ValueError, "Unknown target_observable string"):
        resolve_qelm_target(spec, context, rng=np.random.default_rng(456))


def test_actual_training_returns_weight_shapes_and_optional_fits():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    spec = _small_spec(target=target)

    analysis = analyze_qelm_training(
        spec,
        rng=np.random.default_rng(123),
        return_fits=True,
        return_fit_matrix=True,
    )

    assert analysis.mse.mean_weights.shape == (8,)
    assert analysis.mse.fitted_weights.shape == (2, 8)
    assert analysis.mse.mean_fit_matrix.shape == (8, 8)
    assert analysis.mse.fit_matrices.shape == (2, 8, 8)
    assert np.isfinite(analysis.mse.mse)


def test_qelm_run_caches_resolved_objects_and_analyzes():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    spec = _small_spec(target=target)
    run = QELMRun(spec, rng=np.random.default_rng(123))

    context = run.context
    test = run.test
    diagnostics = run.diagnostics
    target = run.target
    analysis = run.analyze()

    assert run.context is context
    assert run.test is test
    assert run.diagnostics is diagnostics
    assert run.target is target
    assert analysis.context is context
    assert analysis.test is test
    assert analysis.diagnostics is diagnostics
    assert analysis.target is target
    assert np.isfinite(analysis.mse.mse)
    assert np.isfinite(analysis.leading_corrected.mse)
    assert np.isfinite(analysis.leading_identity.mse)


def test_resolved_test_carries_raw_fixed_state_operator():
    fixed_state = np.array([[1.0, 0.0], [0.0, 0.0]])
    spec = _small_spec(target=fixed_state)
    context = make_qelm_training_context(spec, rng=np.random.default_rng(123))

    test = resolve_qelm_test(spec, context, rng=np.random.default_rng(456))

    assert test.states.shape == (1, 2, 2)
    np.testing.assert_allclose(test.states[0], fixed_state)


def test_resolved_training_column_test_carries_raw_state_operator():
    spec = QELMTrainingSpec(
        data=QELMDataSpec(
            d=2,
            nout=8,
            povm={
                "effects": generate_random_rank1_povm(
                    nout=8,
                    dim=2,
                    rng=np.random.default_rng(123),
                ),
                "label": "test_povm",
            },
            train_states={
                "kind": "states",
                "states": np.array(
                    [
                        [[1.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 1.0]],
                    ],
                    dtype=complex,
                ),
            },
        ),
        target=QELMTargetRequest(observable=np.array([[1.0, 0.0], [0.0, 0.0]])),
        test=QELMTestRequest(state="training_column"),
        noise=QELMNoiseSpec(N=20, noise="gaussian", actual_noise_trials=2),
    )
    context = make_qelm_training_context(spec, rng=np.random.default_rng(456))

    test = resolve_qelm_test(spec, context, rng=np.random.default_rng(789))

    assert test.states.shape == (1, 2, 2)
    assert any(np.allclose(test.states[0], state) for state in context.train_states.states)


def test_resolved_target_carries_raw_and_normalized_operator():
    target_operator = np.array([[2.0, 0.0], [0.0, 0.0]])
    spec = _small_spec(
        target=target_operator,
    )
    spec = QELMTrainingSpec(
        data=spec.data,
        target=QELMTargetRequest(observable=target_operator, normalization="euclidean"),
        noise=spec.noise,
        test=spec.test,
        numerics=spec.numerics,
    )
    context = make_qelm_training_context(spec, rng=np.random.default_rng(123))

    target = resolve_qelm_target(spec, context, rng=np.random.default_rng(456))

    np.testing.assert_allclose(target.raw_operator, target_operator)
    np.testing.assert_allclose(target.operator, target_operator / target.scale)
    assert target.weights is None


def test_qelm_run_from_context_reuses_existing_context():
    spec = _small_spec(target=np.array([[1.0, 0.0], [0.0, 0.0]]))
    context = make_qelm_training_context(spec, rng=np.random.default_rng(123))

    run = QELMRun.from_context(spec, context, rng=np.random.default_rng(456))

    assert run.context is context
    assert np.isfinite(run.leading_error().mse)


def test_tilde_u_diagnostics_supports_attributes_mapping_and_transforms():
    spec = _small_spec(target=np.array([[1.0, 0.0], [0.0, 0.0]]))
    context = make_qelm_training_context(spec, rng=np.random.default_rng(123))

    diagnostics = tilde_u_correction_operator_diagnostics(P=context.P_train, rank=4)

    assert diagnostics["C22_inv_C21"] is diagnostics.C22_inv_C21
    assert diagnostics["blocks"] is diagnostics.blocks
    assert dict(diagnostics)["C22_kept_rank"] == diagnostics.C22_kept_rank
    assert diagnostics.U1 is diagnostics.blocks["U1"]
    assert diagnostics.U2 is diagnostics.blocks["U2"]
    assert diagnostics.singular_values is diagnostics.blocks["singular_values"]
    assert diagnostics.training_transform_matrix(approximate_identity=True).shape == (8, 8)
    assert diagnostics.bias_transform_matrix(approximate_identity=False).shape == (8, 8)


def test_actual_training_does_not_store_per_trial_fits_by_default():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    spec = _small_spec(target=target)

    result = run_qelm_actual_training(spec, rng=np.random.default_rng(123))

    assert result.mean_weights.shape == (8,)
    assert result.fitted_weights is None
    assert result.fit_matrices is None


def test_target_average_training_returns_fit_matrix():
    spec = _small_spec(target="haar_pure_average")

    analysis = analyze_qelm_training(spec, rng=np.random.default_rng(123))

    assert analysis.target.is_average
    assert analysis.mse.mean_fit_matrix.shape == (8, 8)
    assert analysis.mse.fit_matrices is None
    assert np.isfinite(analysis.mse.mse)


def test_analyze_qelm_training_uses_default_rng_when_rng_omitted():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    spec = _small_spec(target=target)

    try:
        set_default_rng(123)
        first = analyze_qelm_training(spec)

        set_default_rng(123)
        second = analyze_qelm_training(spec)
    finally:
        clear_default_rng()

    np.testing.assert_allclose(first.context.P_train, second.context.P_train, atol=1e-12)
    np.testing.assert_allclose(first.mse.mse, second.mse.mse, atol=1e-12)


def test_leading_error_matches_existing_low_level_function():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    spec = _small_spec(target=target)
    rng = np.random.default_rng(123)
    context = make_qelm_training_context(spec, rng)
    test = resolve_qelm_test(spec, context, rng)
    diagnostics = compute_qelm_diagnostics(spec, context)
    target = resolve_qelm_target(spec, context, rng)

    result = compute_qelm_leading_error(
        spec,
        context,
        rng,
        corrected=True,
    )
    blocks = diagnostics["blocks"]
    expected = leading_training_bias_variance_terms(
        P=context.P_train,
        U1=blocks["U1"],
        U2=blocks["U2"],
        C22_inv_C21=diagnostics["C22_inv_C21"],
        w_observable=training._target_outcome_weights(target, context),
        singular_values=blocks["singular_values"],
        N=spec.noise.N,
        approximate_identity=False,
        test_probabilities=test.probabilities,
        test_second_moment=test.second_moment,
        dual_test_probabilities=test.dual_probabilities,
        dual_test_second_moment=test.dual_second_moment,
    )

    np.testing.assert_allclose(result.mse, expected["mse"])
    np.testing.assert_allclose(result.bias_sq, expected["bias_sq"])
    np.testing.assert_allclose(result.variance, expected["variance"])


def test_spec_based_tilde_u_experiment_runs():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    study = TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12, 16),
        repetitions=1,
        seed=123,
        verbose=False,
    )

    raw, summary = run_training_experiment(study)

    assert len(raw) == 2
    assert set(raw["ntr"]) == {12, 16}
    assert len(summary) == 2


def test_spec_based_tilde_u_experiment_accepts_haar_pure_string_training_states():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states="haar_pure")
    study = TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12, 16),
        repetitions=1,
        seed=123,
        verbose=False,
    )

    raw, summary = run_training_experiment(study)

    assert len(raw) == 2
    assert set(raw["ntr"]) == {12, 16}
    assert len(summary) == 2


def test_spec_based_tilde_u_N_sweep_can_supply_missing_base_noise_N():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target)
    base = QELMTrainingSpec(
        data=base.data,
        target=base.target,
        noise=QELMNoiseSpec(noise="gaussian", actual_noise_trials=1),
        test=base.test,
        numerics=base.numerics,
    )
    study = TrainingStudySpec(
        base=base,
        sweep_col="N",
        sweep_values=(10, 20),
        repetitions=1,
        seed=123,
        verbose=False,
    )

    raw, summary = run_training_experiment(study)

    assert len(raw) == 2
    assert set(raw["N"]) == {10, 20}
    assert set(summary["N"]) == {10, 20}


def test_tilde_u_report_saves_extensionless_portable_zip(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    output_file = tmp_path / "tilde_u_sweep"
    study = TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12, 16),
        repetitions=1,
        seed=123,
        verbose=False,
        show_summary=False,
        show_slopes=False,
        make_plots=False,
        output_file=output_file,
    )

    raw, summary, slopes = run_training_and_report_results(study)

    saved_path = output_file.with_suffix(".zip")
    assert saved_path.exists()
    loaded = load_traindata(saved_path)
    assert isinstance(loaded, TrainingReport)
    assert set(loaded.datadict) == {"data", "metadata"}
    assert list(loaded.data.columns) == [
        "trial",
        "ntr",
        "leading_bias2",
        "leading_variance",
        "identity_leading_bias2",
        "identity_leading_variance",
        "mse",
        "bias_sq",
        "variance",
    ]
    pd.testing.assert_series_equal(
        loaded.data["mse"],
        raw["mse"],
        check_names=False,
    )
    assert "trial_seed" not in loaded.data.columns
    assert "error" not in loaded.data.columns
    assert "r" not in loaded.data.columns
    assert "p_kernel" not in loaded.data.columns
    assert loaded.metadata["created_at"]
    assert loaded.metadata["completed_at"]
    assert loaded.metadata["elapsed_seconds"] >= 0.0
    assert "format" not in loaded.metadata
    assert "x_col" not in loaded.metadata
    assert "study" not in loaded.metadata
    assert "base_spec" not in loaded.metadata
    assert "concrete_specs" not in loaded.metadata
    assert loaded.metadata["repetitions"] == study.repetitions
    assert loaded.metadata["data"]["d"] == base.data.d
    assert loaded.metadata["data"]["povm"]["kind"] == "random_rank1"
    assert loaded.metadata["data"]["train_states"]["kind"] == "haar_pure"
    assert loaded.metadata["target"]["observable"]["kind"] == "operator"
    assert loaded.metadata["target"]["observable"]["operator"]["shape"] == [2, 2]
    assert loaded.metadata["test"]["state"]["kind"] == "fixed_state"
    assert loaded.metadata["test"]["state"]["state"]["shape"] == [2, 2]
    assert loaded.metadata["sweep_col"] == "ntr"
    assert loaded.metadata["sweep_values"] == [12, 16]


def test_tilde_u_report_rejects_pickle_output(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    study = TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12,),
        repetitions=1,
        seed=123,
        verbose=False,
        show_summary=False,
        show_slopes=False,
        make_plots=False,
        output_file=tmp_path / "tilde_u_sweep.pkl",
    )

    with pytest.raises(ValueError, match=r"must end in \.zip"):
        run_training_and_report_results(study)


def test_tilde_u_report_avoids_overwriting_existing_output(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    output_file = tmp_path / "tilde_u_sweep.zip"
    output_file.write_bytes(b"existing report")
    (tmp_path / "tilde_u_sweep_1.zip").write_bytes(b"existing numbered report")
    study = TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12,),
        repetitions=1,
        seed=123,
        verbose=False,
        show_summary=False,
        show_slopes=False,
        make_plots=False,
        output_file=output_file,
    )

    run_training_and_report_results(study)

    assert output_file.read_bytes() == b"existing report"
    assert (tmp_path / "tilde_u_sweep_1.zip").read_bytes() == b"existing numbered report"
    loaded = load_traindata(tmp_path / "tilde_u_sweep_2.zip")
    assert loaded.metadata["sweep_values"] == [12]


def test_tilde_u_report_overwrite_reuses_requested_output(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    output_file = tmp_path / "tilde_u_sweep.zip"
    output_file.write_bytes(b"existing report")
    study = TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12,),
        repetitions=1,
        seed=123,
        verbose=False,
        show_summary=False,
        show_slopes=False,
        make_plots=False,
        output_file=output_file,
        overwrite=True,
    )

    run_training_and_report_results(study)

    loaded = load_traindata(output_file)
    assert loaded.metadata["sweep_values"] == [12]
    assert not (tmp_path / "tilde_u_sweep_1.zip").exists()


def test_tilde_u_report_saves_portable_parquet_zip(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    output_file = tmp_path / "tilde_u_sweep.zip"
    study = TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12, 16),
        repetitions=1,
        seed=123,
        verbose=False,
        show_summary=False,
        show_slopes=False,
        make_plots=False,
        output_file=output_file,
    )

    raw, summary, slopes = run_training_and_report_results(study)

    loaded = load_traindata(output_file)
    assert isinstance(loaded, TrainingReport)
    assert set(loaded.datadict) == {"data", "metadata"}
    np.testing.assert_allclose(
        loaded.data["leading_bias2"] + loaded.data["leading_variance"],
        raw["leading_mse"],
    )
    assert loaded.metadata["sweep_values"] == [12, 16]


def test_saved_tilde_u_report_data_can_be_summarized_and_plotted(tmp_path, monkeypatch):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    output_file = tmp_path / "tilde_u_sweep.zip"
    study = TrainingStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12, 16),
        repetitions=1,
        seed=123,
        verbose=False,
        show_summary=False,
        show_slopes=False,
        make_plots=False,
        output_file=output_file,
    )
    run_training_and_report_results(study)

    loaded = load_traindata(output_file)
    raw, summary, slopes = summarize_traindata(
        loaded,
        quantile_band=(0.10, 0.90),
    )

    np.testing.assert_allclose(
        raw["leading_mse"],
        loaded.data["leading_bias2"] + loaded.data["leading_variance"],
    )
    np.testing.assert_allclose(
        raw["leading_mse_identity_minus_exact_times_N"],
        raw["N"] * (raw["leading_mse_identity"] - raw["leading_mse"]),
    )
    np.testing.assert_allclose(
        raw["leading_mse_identity_minus_exact_times_N2"],
        raw["N"] ** 2 * (raw["leading_mse_identity"] - raw["leading_mse"]),
    )
    assert set(summary["ntr"]) == {12, 16}
    assert "leading_mse_q10" in summary.columns
    assert "leading_mse_identity_minus_exact_times_N_q10" in summary.columns
    assert "leading_mse_identity_minus_exact_times_N2_q10" in summary.columns
    assert "mse_q90" in summary.columns
    assert set(slopes["x"]) == {"ntr"}

    calls = {}

    def fake_plotter(summary_df, **kwargs):
        calls["summary"] = summary_df
        calls.update(kwargs)

    monkeypatch.setattr(
        training_reports,
        "plot_grouped_mean_median_quantile_summary",
        fake_plotter,
    )

    ax_sentinel = object()
    plotted_raw, plotted_summary, plotted_slopes = plot_saved_traindata(
        output_file,
        plots="mse",
        quantile_band=(0.10, 0.90),
        show_mean=False,
        show_median=False,
        xlim=(10, 20),
        ylim=(0.1, 2.0),
        ax=ax_sentinel,
    )

    assert calls["x_col"] == "ntr"
    expected_mse_plot = workflows.TRAINING_PLOT_SPECS["mse"]
    mse_series, mse_title, mse_ylabel = calls["plots"][0]
    assert mse_series == expected_mse_plot[0]
    assert expected_mse_plot[1] in mse_title
    assert mse_ylabel == expected_mse_plot[2]
    title_lines = mse_title.splitlines()
    assert title_lines[1] == "POVM=random_rank1, nout=8; N=20"
    assert title_lines[2] == "test=fixed; target=fixed; q10-q90; noise=gaussian"
    for text in (
        "POVM=random_rank1, nout=8",
        "N=20",
        "test=fixed",
        "target=fixed",
        "q10-q90",
        "noise=gaussian",
    ):
        assert text in mse_title
    mse_calls = calls.copy()

    delta_raw, delta_summary, _delta_slopes = plot_saved_traindata(
        output_file,
        plots="leading_mse_delta_N2",
        quantile_band=(0.10, 0.90),
        make_plots=False,
    )
    assert "leading_mse_identity_minus_exact_times_N2" in delta_raw.columns
    assert "leading_mse_identity_minus_exact_times_N" not in delta_raw.columns
    assert "leading_mse_identity_minus_exact_times_N2_q10" in delta_summary.columns

    custom_delta = training_reports.MetricExpr(
        "custom_leading_mse_delta",
        ("leading_mse_identity", "leading_mse"),
        lambda df: df["leading_mse_identity"] - df["leading_mse"],
    )
    custom_plot = (
        [(custom_delta, "identity - corrected")],
        "Custom leading-MSE delta",
        "delta",
    )
    custom_raw, custom_summary, _custom_slopes = plot_saved_traindata(
        output_file,
        plots=custom_plot,
        quantile_band=(0.10, 0.90),
    )
    np.testing.assert_allclose(
        custom_raw["custom_leading_mse_delta"],
        custom_raw["leading_mse_identity"] - custom_raw["leading_mse"],
    )
    assert "custom_leading_mse_delta_q10" in custom_summary.columns
    assert calls["plots"][0][0] == [
        ("custom_leading_mse_delta", "identity - corrected")
    ]

    assert "quantiles=" not in mse_title
    assert mse_calls["quantile_band"] == (0.10, 0.90)
    assert mse_calls["show_mean"] is False
    assert mse_calls["show_median"] is False
    assert mse_calls["xlim"] == (10, 20)
    assert mse_calls["ylim"] == (0.1, 2.0)
    assert mse_calls["ax"] is ax_sentinel
    shared_raw_cols = list(plotted_raw.columns.intersection(raw.columns))
    shared_summary_cols = list(plotted_summary.columns.intersection(summary.columns))
    pd.testing.assert_frame_equal(plotted_raw[shared_raw_cols], raw[shared_raw_cols])
    pd.testing.assert_frame_equal(
        plotted_summary[shared_summary_cols],
        summary[shared_summary_cols],
    )
    assert "mse_identity_over_exact" not in plotted_raw.columns
    assert set(plotted_slopes["y"]) <= {
        "leading_mse_median",
        "leading_mse_identity_median",
        "mse_median",
    }


@pytest.mark.parametrize(
    ("x_col", "expected_label"),
    [
        ("ntr", r"$n_{\mathrm{tr}}$"),
        ("nout", r"$n_{\mathrm{out}}$"),
        ("N", r"$N$"),
    ],
)
def test_grouped_tilde_u_plot_uses_latex_axis_labels(monkeypatch, x_col, expected_label):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    summary = pd.DataFrame(
        {
            x_col: [12, 16],
            "mse_median": [1.0, 0.8],
            "mse_mean": [1.1, 0.9],
            "mse_q10": [0.9, 0.7],
            "mse_q90": [1.2, 1.0],
        }
    )

    workflows.plot_grouped_mean_median_quantile_summary(
        summary,
        x_col=x_col,
        plots=[([("mse", "MSE")], "title", "ylabel")],
        quantile_band=(0.10, 0.90),
        logx=False,
        logy=False,
    )

    assert plt.gcf().axes[0].get_xlabel() == expected_label
    plt.close("all")


def test_grouped_tilde_u_plot_can_show_mean_without_median(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    summary = pd.DataFrame(
        {
            "ntr": [12, 16, 100],
            "mse_median": [1.0, 0.8, 50.0],
            "mse_mean": [1.1, 0.9, 60.0],
            "mse_q10": [0.9, 0.7, 40.0],
            "mse_q90": [1.2, 1.0, 70.0],
        }
    )

    workflows.plot_grouped_mean_median_quantile_summary(
        summary,
        x_col="ntr",
        plots=[([("mse", "MSE")], "title", "ylabel")],
        quantile_band=(0.10, 0.90),
        logx=False,
        logy=False,
        show_median=False,
        xlim=(10, 20),
    )

    ax = plt.gcf().axes[0]
    labels = [line.get_label() for line in ax.lines]
    assert labels == ["MSE, mean"]
    assert ax.get_xlim() == (10, 20)
    assert ax.get_ylim()[1] < 2.0
    plt.close("all")


def test_grouped_tilde_u_plot_draws_into_supplied_axes_without_show(monkeypatch):
    import matplotlib.pyplot as plt

    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))
    summary = pd.DataFrame(
        {
            "ntr": [12, 16],
            "mse_median": [1.0, 0.8],
            "mse_mean": [1.1, 0.9],
            "mse_q10": [0.9, 0.7],
            "mse_q90": [1.2, 1.0],
        }
    )
    fig, axes = plt.subplots(1, 2)

    for ax, title in zip(axes, ("left", "right")):
        workflows.plot_grouped_mean_median_quantile_summary(
            summary,
            x_col="ntr",
            plots=[([("mse", "MSE")], title, "ylabel")],
            quantile_band=(0.10, 0.90),
            logx=False,
            logy=False,
            show_mean=False,
            ax=ax,
        )

    assert show_calls == []
    assert [len(ax.lines) for ax in axes] == [1, 1]
    assert [ax.get_title() for ax in axes] == ["left", "right"]
    plt.close(fig)


def test_grouped_tilde_u_plot_explicit_ylim_overrides_visible_x_autoscale(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    summary = pd.DataFrame(
        {
            "ntr": [12, 16, 100],
            "mse_median": [1.0, 0.8, 50.0],
            "mse_mean": [1.1, 0.9, 60.0],
            "mse_q10": [0.9, 0.7, 40.0],
            "mse_q90": [1.2, 1.0, 70.0],
        }
    )

    workflows.plot_grouped_mean_median_quantile_summary(
        summary,
        x_col="ntr",
        plots=[([("mse", "MSE")], "title", "ylabel")],
        quantile_band=(0.10, 0.90),
        logx=False,
        logy=False,
        xlim=(10, 20),
        ylim=(0.25, 3.0),
    )

    ax = plt.gcf().axes[0]
    assert ax.get_xlim() == (10, 20)
    assert ax.get_ylim() == (0.25, 3.0)
    plt.close("all")


def test_live_tilde_u_mse_plot_title_contains_context(monkeypatch):
    base = QELMTrainingSpec(
        data=QELMDataSpec(
            d=2,
            nout=8,
            povm=None,
            train_states={"kind": "haar_pure", "num_states": 12},
        ),
        target=QELMTargetRequest(observable="haar_pure_average"),
        test=QELMTestRequest(state="haar_pure_average"),
        noise=QELMNoiseSpec(noise="multinomial", actual_noise_trials=1),
    )
    study = TrainingStudySpec(
        base=base,
        sweep_col="N",
        sweep_values=(20,),
        repetitions=1,
        seed=123,
        quantile_band=(0.25, 0.75),
        verbose=False,
        show_summary=False,
        show_slopes=False,
        plots="mse",
    )
    calls = {}

    def fake_plotter(summary_df, **kwargs):
        calls["summary"] = summary_df
        calls.update(kwargs)

    monkeypatch.setattr(
        training_reports,
        "plot_grouped_mean_median_quantile_summary",
        fake_plotter,
    )

    run_training_and_report_results(study)

    mse_title = calls["plots"][0][1]
    title_lines = mse_title.splitlines()
    assert title_lines[1] == "POVM=random_rank1, nout=8; ntr=12"
    assert title_lines[2] == "test=haar; target=haar; q25-q75; noise=multinomial"
    for text in (
        "POVM=random_rank1, nout=8",
        "ntr=12",
        "test=haar",
        "target=haar",
        "q25-q75",
        "noise=multinomial",
    ):
        assert text in mse_title
    assert "Haar avg" not in mse_title
    assert "quantiles=" not in mse_title


def test_tilde_u_mse_plot_title_recovers_old_unlabeled_mub_metadata():
    metadata = {
        "sweep_col": "ntr",
        "data": {
            "povm": {
                "kind": "explicit",
                "effects": workflows._array_payload(qubit_mub_povm()),
            },
            "train_state_count": None,
        },
        "test": {"state": {"kind": "haar_pure_average"}},
        "target": {"observable": {"kind": "haar_pure_average"}},
        "noise": {"N": 100, "noise": "multinomial"},
    }

    title = workflows._training_context_title_suffix(
        metadata,
        quantile_band=(0.05, 0.95),
    )
    normalized = workflows._normalize_metadata(metadata)

    lines = title.splitlines()
    assert lines[0] == "POVM=qubit_mub; N=100"
    assert lines[1] == "test=haar; target=haar; q5-q95; noise=multinomial"
    assert normalized["data"]["povm"]["label"] == "qubit_mub"


def test_tilde_u_report_metadata_stores_explicit_povm_and_training_states(tmp_path):
    povm = generate_random_rank1_povm(nout=8, dim=2, rng=np.random.default_rng(123))
    train_states = QuantumStateBatch.haar_pure_from_columns(
        num_states=12,
        dim=2,
        rng=np.random.default_rng(456),
    )
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(povm=povm, target=target, train_states=train_states)
    study = TrainingStudySpec(
        base=base,
        sweep_col="N",
        sweep_values=(20,),
        repetitions=1,
        seed=123,
        verbose=False,
        show_summary=False,
        show_slopes=False,
        make_plots=False,
        output_file=tmp_path / "explicit.zip",
    )

    run_training_and_report_results(study)

    loaded = load_traindata(tmp_path / "explicit.zip")
    assert loaded.metadata["data"]["povm"]["kind"] == "explicit"
    assert loaded.metadata["data"]["povm"]["label"] == "test_povm"
    assert loaded.metadata["data"]["povm"]["effects"]["shape"] == [8, 2, 2]
    assert loaded.metadata["data"]["train_states"]["kind"] == "explicit"
    assert loaded.metadata["data"]["train_states"]["states"]["shape"] == [12, 2, 2]
    assert loaded.metadata["target"]["observable"]["operator"]["shape"] == [2, 2]


def test_tilde_u_plot_selector_uses_short_keys():
    assert workflows._training_plots_from_keys("mse") == [
        workflows.TRAINING_PLOT_SPECS["mse"]
    ]
    assert workflows._training_plots_from_keys(("mse", "correction")) == [
        workflows.TRAINING_PLOT_SPECS["mse"],
        workflows.TRAINING_PLOT_SPECS["correction"],
    ]
    assert workflows._training_plots_from_keys("leading_mse_delta_N") == [
        workflows.TRAINING_PLOT_SPECS["leading_mse_delta_N"]
    ]
    assert workflows._training_plots_from_keys("leading_mse_delta_N2") == [
        workflows.TRAINING_PLOT_SPECS["leading_mse_delta_N2"]
    ]
    assert workflows._training_plots_from_keys("all") == (
        workflows.TRAINING_PLOTS
    )

    with pytest.raises(ValueError, match="Unknown training plot key"):
        workflows._training_plots_from_keys("not_a_plot")


def test_training_report_helpers_are_reexported_for_compatibility():
    assert workflows.plot_saved_traindata is training_reports.plot_saved_traindata
    assert workflows.summarize_traindata is training_reports.summarize_traindata
    assert (
        workflows.load_traindata
        is training_reports.load_traindata
    )
    assert (
        workflows.summarize_dataraw
        is training_reports.summarize_dataraw
    )
    assert (
        workflows.TRAINING_PLOT_SPECS
        is training_reports.TRAINING_PLOT_SPECS
    )
    assert workflows.run_tilde_u_training_approx_experiment is workflows.run_training_experiment
    assert workflows.run_tilde_u_training_approx_report is workflows.run_training_and_report_results
    assert training_reports.load_tilde_u_training_approx_report_data is training_reports.load_traindata
    assert training_reports.plot_saved_training_data is training_reports.plot_saved_traindata
    assert callable(plot_leading_mse_difference_grid_over_N)


def test_mse_grid_allows_autoscaled_ylim(tmp_path):
    import matplotlib.pyplot as plt

    calls = []
    panel_limits = [(0.2, 3.0), (0.5, 50.0)]

    def fake_plotter(path, **kwargs):
        calls.append(kwargs["ylim"])
        ax = kwargs["ax"]
        ax.plot([1, 2], [1, 2], label="line")
        ax.set_ylim(panel_limits[len(calls) - 1])

    fig, axes = plot_mse_grid_over_N(
        tmp_path,
        n_min=1,
        n_max=2,
        ncols=2,
        ylim=None,
        plot_func=fake_plotter,
    )

    assert calls == [None, None]
    assert axes[0].get_ylim() == (0.2, 50.0)
    assert axes[1].get_ylim() == (0.2, 50.0)
    plt.close(fig)


def test_mse_grid_allows_per_panel_autoscaled_ylim(tmp_path):
    import matplotlib.pyplot as plt

    calls = []
    panel_limits = [(0.2, 3.0), (0.5, 50.0)]

    def fake_plotter(path, **kwargs):
        calls.append(kwargs["ylim"])
        ax = kwargs["ax"]
        ax.plot([1, 2], [1, 2], label="line")
        ax.set_ylim(panel_limits[len(calls) - 1])

    fig, axes = plot_mse_grid_over_N(
        tmp_path,
        n_min=1,
        n_max=2,
        ncols=2,
        sharey=False,
        ylim=None,
        plot_func=fake_plotter,
    )

    assert calls == [None, None]
    assert axes[0].get_ylim() == panel_limits[0]
    assert axes[1].get_ylim() == panel_limits[1]
    plt.close(fig)


def test_mse_grid_allows_one_sided_ylim_with_autoscale(tmp_path):
    import matplotlib.pyplot as plt

    def fake_plotter(path, **kwargs):
        ax = kwargs["ax"]
        ax.plot([1, 2], [1, 2], label="line")
        ax.set_ylim(0.2, 3.0)

    fig, axes = plot_mse_grid_over_N(
        tmp_path,
        n_min=1,
        n_max=1,
        ncols=1,
        ylim=(None, 10.0),
        plot_func=fake_plotter,
    )

    assert axes[0].get_ylim() == (0.2, 10.0)
    plt.close(fig)


def test_mse_grid_allows_per_panel_one_sided_ylim(tmp_path):
    import matplotlib.pyplot as plt

    calls = []
    panel_limits = [(0.2, 3.0), (0.5, 50.0)]

    def fake_plotter(path, **kwargs):
        calls.append(kwargs["ylim"])
        ax = kwargs["ax"]
        ax.plot([1, 2], [1, 2], label="line")
        ax.set_ylim(panel_limits[len(calls) - 1])

    fig, axes = plot_mse_grid_over_N(
        tmp_path,
        n_min=1,
        n_max=2,
        ncols=2,
        sharey=False,
        ylim=(None, 10.0),
        plot_func=fake_plotter,
    )

    assert axes[0].get_ylim() == (0.2, 10.0)
    assert axes[1].get_ylim() == (0.5, 10.0)
    plt.close(fig)


def test_actual_training_requires_noise_N_when_not_swept():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target)
    spec = QELMTrainingSpec(
        data=base.data,
        target=base.target,
        noise=QELMNoiseSpec(noise="gaussian", actual_noise_trials=1),
        test=base.test,
        numerics=base.numerics,
    )

    with np.testing.assert_raises_regex(ValueError, "noise.N is required"):
        run_qelm_actual_training(spec, rng=np.random.default_rng(123))
