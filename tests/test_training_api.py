import numpy as np
import pandas as pd
import pytest

import qelm_rank.training as training
import qelm_rank.workflows as workflows
from qelm_rank import (
    QELMDataSpec,
    QELMNoiseSpec,
    QELMTargetRequest,
    QELMTestRequest,
    QELMRun,
    QELMTrainingSpec,
    POVM,
    QuantumStateBatch,
    TildeUTrainingApproxStudySpec,
    analyze_qelm_training,
    clear_default_rng,
    compute_qelm_diagnostics,
    compute_qelm_leading_error,
    generate_random_rank1_povm,
    leading_training_bias_variance_terms,
    load_tilde_u_training_approx_report_data,
    make_qelm_training_context,
    resolve_qelm_target,
    resolve_qelm_test,
    run_qelm_actual_training,
    run_tilde_u_training_approx_experiment,
    run_tilde_u_training_approx_report,
    set_default_rng,
    tilde_u_correction_operator_diagnostics,
)


def _small_spec(*, povm=None, target=None, train_states=None, ntr=12):
    if train_states is None:
        train_states = {"kind": "haar_pure", "num_states": ntr}
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
            povm=povm,
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


def test_explicit_array_povm_allows_omitting_nout():
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

    run = QELMRun(spec, rng=np.random.default_rng(456))

    assert spec.data.nout == 8
    assert run.context.P_train.shape == (8, 12)
    np.testing.assert_allclose(run.context.povm.effects, povm)


def test_explicit_povm_object_allows_omitting_nout():
    effects = generate_random_rank1_povm(nout=8, dim=2, rng=np.random.default_rng(123))
    povm = POVM.from_effects(effects, dim=2)
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
            povm=povm,
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
            povm=generate_random_rank1_povm(nout=8, dim=2, rng=np.random.default_rng(123)),
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
    study = TildeUTrainingApproxStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12, 16),
        repetitions=1,
        seed=123,
        verbose=False,
    )

    raw, summary = run_tilde_u_training_approx_experiment(study)

    assert len(raw) == 2
    assert set(raw["ntr"]) == {12, 16}
    assert len(summary) == 2


def test_spec_based_tilde_u_experiment_accepts_haar_pure_string_training_states():
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states="haar_pure")
    study = TildeUTrainingApproxStudySpec(
        base=base,
        sweep_col="ntr",
        sweep_values=(12, 16),
        repetitions=1,
        seed=123,
        verbose=False,
    )

    raw, summary = run_tilde_u_training_approx_experiment(study)

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
    study = TildeUTrainingApproxStudySpec(
        base=base,
        sweep_col="N",
        sweep_values=(10, 20),
        repetitions=1,
        seed=123,
        verbose=False,
    )

    raw, summary = run_tilde_u_training_approx_experiment(study)

    assert len(raw) == 2
    assert set(raw["N"]) == {10, 20}
    assert set(summary["N"]) == {10, 20}


def test_tilde_u_report_saves_extensionless_portable_zip(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    output_file = tmp_path / "tilde_u_sweep"
    study = TildeUTrainingApproxStudySpec(
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

    raw, summary, slopes = run_tilde_u_training_approx_report(study)

    saved_path = output_file.with_suffix(".zip")
    assert saved_path.exists()
    loaded = load_tilde_u_training_approx_report_data(saved_path)
    assert set(loaded) == {"raw", "metadata"}
    assert list(loaded["raw"].columns) == [
        "trial",
        "ntr",
        "leading_bias_sq",
        "leading_variance",
        "identity_leading_bias_sq",
        "identity_leading_variance",
        "mse",
        "bias_sq",
        "variance",
    ]
    pd.testing.assert_series_equal(
        loaded["raw"]["mse"],
        raw["actual_mse"],
        check_names=False,
    )
    assert "trial_seed" not in loaded["raw"].columns
    assert "error" not in loaded["raw"].columns
    assert "r" not in loaded["raw"].columns
    assert "p_kernel" not in loaded["raw"].columns
    assert loaded["metadata"]["created_at"]
    assert loaded["metadata"]["completed_at"]
    assert loaded["metadata"]["elapsed_seconds"] >= 0.0
    assert "format" not in loaded["metadata"]
    assert "x_col" not in loaded["metadata"]
    assert "study" not in loaded["metadata"]
    assert "base_spec" not in loaded["metadata"]
    assert "concrete_specs" not in loaded["metadata"]
    assert loaded["metadata"]["repetitions"] == study.repetitions
    assert loaded["metadata"]["data"]["d"] == base.data.d
    assert loaded["metadata"]["data"]["povm"]["kind"] == "random_rank1"
    assert loaded["metadata"]["data"]["train_states"]["kind"] == "haar_pure"
    assert loaded["metadata"]["target"]["observable"]["kind"] == "operator"
    assert loaded["metadata"]["target"]["observable"]["operator"]["shape"] == [2, 2]
    assert loaded["metadata"]["test"]["state"]["kind"] == "fixed_state"
    assert loaded["metadata"]["test"]["state"]["state"]["shape"] == [2, 2]
    assert loaded["metadata"]["sweep_col"] == "ntr"
    assert loaded["metadata"]["sweep_values"] == [12, 16]


def test_tilde_u_report_rejects_pickle_output(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    study = TildeUTrainingApproxStudySpec(
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
        run_tilde_u_training_approx_report(study)


def test_tilde_u_report_avoids_overwriting_existing_output(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    output_file = tmp_path / "tilde_u_sweep.zip"
    output_file.write_bytes(b"existing report")
    (tmp_path / "tilde_u_sweep_1.zip").write_bytes(b"existing numbered report")
    study = TildeUTrainingApproxStudySpec(
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

    run_tilde_u_training_approx_report(study)

    assert output_file.read_bytes() == b"existing report"
    assert (tmp_path / "tilde_u_sweep_1.zip").read_bytes() == b"existing numbered report"
    loaded = load_tilde_u_training_approx_report_data(tmp_path / "tilde_u_sweep_2.zip")
    assert loaded["metadata"]["sweep_values"] == [12]


def test_tilde_u_report_overwrite_reuses_requested_output(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    output_file = tmp_path / "tilde_u_sweep.zip"
    output_file.write_bytes(b"existing report")
    study = TildeUTrainingApproxStudySpec(
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

    run_tilde_u_training_approx_report(study)

    loaded = load_tilde_u_training_approx_report_data(output_file)
    assert loaded["metadata"]["sweep_values"] == [12]
    assert not (tmp_path / "tilde_u_sweep_1.zip").exists()


def test_tilde_u_report_saves_portable_parquet_zip(tmp_path):
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(target=target, train_states={"kind": "haar_pure"})
    output_file = tmp_path / "tilde_u_sweep.zip"
    study = TildeUTrainingApproxStudySpec(
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

    raw, summary, slopes = run_tilde_u_training_approx_report(study)

    loaded = load_tilde_u_training_approx_report_data(output_file)
    assert set(loaded) == {"raw", "metadata"}
    np.testing.assert_allclose(
        loaded["raw"]["leading_bias_sq"] + loaded["raw"]["leading_variance"],
        raw["leading_mse_exact"],
    )
    assert loaded["metadata"]["sweep_values"] == [12, 16]


def test_tilde_u_report_metadata_stores_explicit_povm_and_training_states(tmp_path):
    povm = generate_random_rank1_povm(nout=8, dim=2, rng=np.random.default_rng(123))
    train_states = QuantumStateBatch.haar_pure_from_columns(
        num_states=12,
        dim=2,
        rng=np.random.default_rng(456),
    )
    target = np.array([[1.0, 0.0], [0.0, 0.0]])
    base = _small_spec(povm=povm, target=target, train_states=train_states)
    study = TildeUTrainingApproxStudySpec(
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

    run_tilde_u_training_approx_report(study)

    loaded = load_tilde_u_training_approx_report_data(tmp_path / "explicit.zip")
    assert loaded["metadata"]["data"]["povm"]["kind"] == "explicit"
    assert loaded["metadata"]["data"]["povm"]["effects"]["shape"] == [8, 2, 2]
    assert loaded["metadata"]["data"]["train_states"]["kind"] == "explicit"
    assert loaded["metadata"]["data"]["train_states"]["states"]["shape"] == [12, 2, 2]
    assert loaded["metadata"]["target"]["observable"]["operator"]["shape"] == [2, 2]


def test_tilde_u_plot_selector_uses_short_keys():
    assert workflows._tilde_u_training_approx_plots_from_keys("mse") == [
        workflows.TILDE_U_TRAINING_APPROX_PLOT_SPECS["mse"]
    ]
    assert workflows._tilde_u_training_approx_plots_from_keys(("mse", "correction")) == [
        workflows.TILDE_U_TRAINING_APPROX_PLOT_SPECS["mse"],
        workflows.TILDE_U_TRAINING_APPROX_PLOT_SPECS["correction"],
    ]
    assert workflows._tilde_u_training_approx_plots_from_keys("all") == (
        workflows.TILDE_U_TRAINING_APPROX_PLOTS
    )

    with pytest.raises(ValueError, match="Unknown tilde-U plot key"):
        workflows._tilde_u_training_approx_plots_from_keys("not_a_plot")


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
